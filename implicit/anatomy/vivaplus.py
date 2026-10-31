"""Average (idealised) bones from the VIVA+ human body models -> implicit bodies.

    python -m implicit.anatomy.vivaplus --model 50F-standing --out assets/bones/vivaplus-50F

VIVA+ (Chalmers University of Technology and partners) are open finite element
models of an average female (50F) and male (50M), seated and standing:
https://vivaplus.readthedocs.io, doi:10.5281/zenodo.7789362, licensed LGPL-3.0-or-later.
This step reads the model's bone surfaces (cortical shells; femurs: the outer
boundary of their solid elements) and converts each bone once into the same
implicit bone file as implicit.anatomy.convert:

  1. inside/outside on a fine grid by ray parity along x, y and z (majority vote,
     so a small gap in the surface mesh cannot flip a whole column);
  2. signed distance to that mask (implicit.anatomy.convert.bone_sdf);
  3. coordinates: the model frame (x forward, y left, z up, mm) becomes the
     viewer's right-anterior-superior frame (x right, y anterior, z up).
Then landmarks are detected (implicit.anatomy.landmarks) and the vertebrae are
translated so that the rest disc heights take population values
(implicit.anatomy.template; --keep-disc-heights keeps the model's).

The bone files are derived from VIVA+ and stay under LGPL-3.0-or-later; see NOTICE.md.
Requires scipy (this step only).
"""

from __future__ import annotations

import argparse
import io
import json
import re
import urllib.request
import zipfile
from pathlib import Path

import numpy as np

from .convert import SPACING, bone_sdf, save_bone

RECORD = "18338876"  # Zenodo record of VIVA+ v2.0.2
URL = f"https://zenodo.org/api/records/{RECORD}/files/vivaplus-v2.0.2.zip/content"
VERSION = "2.0.2"
ATTRIBUTION = ("VIVA+ human body models v2.0.2 (Chalmers University of Technology et al.), "
               "doi:10.5281/zenodo.7789362, LGPL-3.0-or-later. Bone surfaces converted to "
               "signed distance fields; the derived files are under the same licence.")

# Bone -> (kind, part-name pattern). Vertebrae: cortical shell (+ endplate shells).
BONES = {f"T{n}": ("vertebra", rf"^TX-Spine-T{n}-Cortical-M$") for n in range(1, 13)}
BONES.update({f"L{n}": ("vertebra", rf"^AB-(Spine-)?L{n}-(Cortical|Endplate)") for n in range(1, 6)})
BONES.update({
    "sacrum": ("sacrum", r"^PE-Pelvic-Sacrum-Cortical(-Nulls)?-[LR]$"),
    "hip_left": ("hip", r"^PE-Pelvic-(Ilium|Ischium|Pubis|Acetabulum)-Cortical-L$"),
    "hip_right": ("hip", r"^PE-Pelvic-(Ilium|Ischium|Pubis|Acetabulum)-Cortical-R$"),
    "femur_left": ("femur", r"^LX-Bone-Femur-(head|neck|greater-trochanter|lesser-trochanter|shaft-proximal)-"
                            r"(Cortical|Trabecular)-L$"),
    "femur_right": ("femur", r"^LX-Bone-Femur-(head|neck|greater-trochanter|lesser-trochanter|shaft-proximal)-"
                             r"(Cortical|Trabecular)-R$"),
})
# Rib cage: ribs (cortical shell and its posterior end), sternum, costal cartilages
# (one solid part per side, all ribs; its boundary is closed, the exterior shells are not).
for _n in range(1, 13):
    for _s, _c in (("left", "L"), ("right", "R")):
        BONES[f"rib{_n}_{_s}"] = ("rib", rf"^TX-Ribcage-R{_n}-Cortical(-Posterior-End)?-{_c}$")
BONES["sternum"] = ("sternum", r"^TX-Ribcage-Sternum-Cortical-M$")
for _s, _c in (("left", "L"), ("right", "R")):
    BONES[f"costal_cartilage_{_s}"] = ("cartilage", rf"^TX-Ribcage-Cartilage-Solid-{_c}$")
RIBCAGE = [f"rib{n}_{s}" for n in range(1, 13) for s in ("left", "right")] + [
    "sternum", "costal_cartilage_left", "costal_cartilage_right"]
PRESETS = {"spine": [f"T{n}" for n in range(1, 13)] + [f"L{n}" for n in range(1, 6)]
           + ["sacrum", "hip_left", "hip_right", "femur_left", "femur_right"],
           "lumbar": [f"L{n}" for n in range(1, 6)] + ["sacrum", "hip_left", "hip_right",
                                                      "femur_left", "femur_right"],
           "ribcage": RIBCAGE}
PRESETS["full"] = PRESETS["spine"] + RIBCAGE
MASK_H = {"vertebra": 0.25, "sacrum": 0.4, "hip": 0.6, "femur": 0.4, "rib": 0.3, "sternum": 0.3, "cartilage": 0.4}  # mm, inside/outside grid

# Solid faces (LS-DYNA node order): hexahedron; tetrahedra repeat the 4th node.
HEX_FACES = [(0, 1, 2, 3), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)]


# ---------------------------------------------------------------- reading the model

def fetch(cache):
    """Download the VIVA+ release once; returns the extracted model folder."""
    cache = Path(cache)
    model = cache / f"vivaplus-v{VERSION}" / "model"
    if not model.exists():
        cache.mkdir(parents=True, exist_ok=True)
        print(f"downloading VIVA+ v{VERSION} (~50 MB) from Zenodo")
        with urllib.request.urlopen(URL) as r:
            zipfile.ZipFile(io.BytesIO(r.read())).extractall(cache)
    return model


def _cards(path):
    """(keyword, data line) for each data line of a keyword file (comments skipped)."""
    kw = None
    with open(path, errors="ignore") as f:
        for line in f:
            if line.startswith("*"):
                kw = line.strip().upper()
            elif not line.startswith("$"):
                yield kw, line.rstrip("\n")


def _ints(line, start, n, width=8):
    out = []
    for k in range(n):
        s = line[start + k * width:start + (k + 1) * width].strip()
        out.append(int(s) if s else 0)
    return out


def read_parts(model):
    """{part id: title} from all *PART cards of the common files."""
    parts = {}
    for f in sorted((Path(model) / "common").glob("*.k")):
        kw, k, title = None, 0, None
        for line in f.read_text(errors="ignore").splitlines():
            if line.startswith("*"):
                kw, k = line.strip().upper(), 0
                continue
            if line.startswith("$") or kw is None or not kw.startswith("*PART"):
                continue
            if k == 0:
                title = line.strip()
            elif k == 1:
                try:
                    parts[int(line[:10])] = title
                except ValueError:
                    pass
            k += 1
    return parts


def read_elements(model, pids):
    """Shell quads/triangles and solid elements of the given parts: ({pid: faces}, {pid: solids})."""
    shells, solids = {}, {}
    for f in sorted((Path(model) / "common").glob("*.k")):
        kw_prev, skip = None, False
        for kw, line in _cards(f):
            if kw != kw_prev:  # new keyword block: element lines start again
                kw_prev, skip = kw, False
            if not kw or not kw.startswith("*ELEMENT_"):
                continue
            if skip:  # THICKNESS/BETA variants: every other line is the thickness card
                skip = False
                continue
            if kw.startswith("*ELEMENT_SHELL"):
                skip = "THICKNESS" in kw or "BETA" in kw
                eid, pid, *n = _ints(line, 0, 6)
                if pid in pids:
                    shells.setdefault(pid, []).append(n)
            elif kw.startswith("*ELEMENT_SOLID"):
                vals = _ints(line, 0, 10)
                if vals[1] in pids:
                    solids.setdefault(vals[1], []).append(vals[2:])
    return shells, solids


def read_nodes(model, posture):
    nodes = {}
    for f in sorted((Path(model) / posture).glob("*nodes*.k")):
        for kw, line in _cards(f):
            if kw == "*NODE" and len(line) >= 56:
                try:
                    nodes[int(line[:8])] = (float(line[8:24]), float(line[24:40]), float(line[40:56]))
                except ValueError:
                    pass
    return nodes


def surface(shells, solids):
    """Triangles (node ids) of a bone: its shells, plus the boundary faces of its solids."""
    quads = [q for faces in shells.values() for q in faces]
    count = {}
    for els in solids.values():
        for e in els:
            for fc in HEX_FACES:
                q = tuple(e[i] for i in fc)
                key = tuple(sorted(set(q)))
                if len(key) < 3:
                    continue
                count.setdefault(key, [0, q])[0] += 1
    quads += [list(q) for n, q in count.values() if n == 1]
    tris = []
    for q in quads:
        a, b, c, d = q
        tris.append((a, b, c))
        if d and d != c:
            tris.append((a, c, d))
    return [t for t in tris if len(set(t)) == 3]


# ---------------------------------------------------------------- inside/outside

def _parity(V, h, lo, n, axis):
    """Inside mask by ray parity along one axis (crossings counted below each sample)."""
    a, b = [k for k in range(3) if k != axis]
    hist = np.zeros((n[a], n[b], n[axis] + 1), np.int32)
    P0, P1, P2 = V[:, 0], V[:, 1], V[:, 2]
    for t0, t1, t2 in zip(P0, P1, P2):
        u = np.array([t0[a], t1[a], t2[a]])
        v = np.array([t0[b], t1[b], t2[b]])
        i0 = max(int(np.ceil((u.min() - lo[a]) / h)), 0)
        i1 = min(int(np.floor((u.max() - lo[a]) / h)), n[a] - 1)
        j0 = max(int(np.ceil((v.min() - lo[b]) / h)), 0)
        j1 = min(int(np.floor((v.max() - lo[b]) / h)), n[b] - 1)
        if i1 < i0 or j1 < j0:
            continue
        I, J = np.meshgrid(np.arange(i0, i1 + 1), np.arange(j0, j1 + 1), indexing="ij")
        pu, pv = lo[a] + I * h, lo[b] + J * h
        d = (v[1] - v[2]) * (u[0] - u[2]) + (u[2] - u[1]) * (v[0] - v[2])
        if abs(d) < 1e-12:
            continue
        l0 = ((v[1] - v[2]) * (pu - u[2]) + (u[2] - u[1]) * (pv - v[2])) / d
        l1 = ((v[2] - v[0]) * (pu - u[2]) + (u[0] - u[2]) * (pv - v[2])) / d
        l2 = 1 - l0 - l1
        ok = (l0 >= 0) & (l1 >= 0) & (l2 >= 0)
        if not ok.any():
            continue
        w = l0 * t0[axis] + l1 * t1[axis] + l2 * t2[axis]
        k = np.clip(np.floor((w[ok] - lo[axis]) / h).astype(int) + 1, 0, n[axis])
        np.add.at(hist, (I[ok], J[ok], k), 1)
    inside = (np.cumsum(hist, axis=2)[..., :n[axis]] % 2) == 1
    return np.moveaxis(inside, [0, 1, 2], [a, b, axis])


def inside_mask(V, h, pad=4.0):
    """Majority of x/y/z ray-parity tests on a grid of spacing h: (mask, origin)."""
    pts = V.reshape(-1, 3)
    # A tiny irrational shift keeps grid rays off mesh vertices and edges.
    lo = pts.min(0) - pad + h * np.array([0.1371, 0.2713, 0.3117])
    n = np.ceil((pts.max(0) + pad - lo) / h).astype(int) + 1
    votes = sum(_parity(V, h, lo, n, ax).astype(np.int8) for ax in range(3))
    return votes >= 2, lo


# ---------------------------------------------------------------- conversion

def to_ras(xyz):
    """VIVA+ frame (x forward, y left, z up) -> viewer frame (x right, y anterior, z up)."""
    xyz = np.asarray(xyz, float)
    return np.stack([-xyz[..., 1], xyz[..., 0], xyz[..., 2]], -1)


def convert(model, out, posture="50F-standing", select="spine", log=print):
    model, out = Path(model), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    names = PRESETS.get(select, select.split(",") if isinstance(select, str) else select)
    parts = read_parts(model)
    groups = {b: {p for p, t in parts.items() if re.search(BONES[b][1], t)} for b in names}
    for b, pids in groups.items():
        if not pids:
            raise SystemExit(f"{b}: no parts match {BONES[b][1]!r}")
    shells, solids = read_elements(model, set().union(*groups.values()))
    nodes = read_nodes(model, posture)
    subject = f"vivaplus-{posture.split('-')[0]}"
    for b in names:
        kind = BONES[b][0]
        sh = {p: shells.get(p, []) for p in groups[b]}
        so = {p: solids.get(p, []) for p in groups[b]}
        tris = surface(sh, so)
        V = to_ras(np.array([[nodes[i] for i in t] for t in tris]))
        h = MASK_H[kind]
        mask, lo = inside_mask(V, h)
        affine = np.diag([h, h, h, 1.0])
        affine[:3, 3] = lo
        values, origin, _ = bone_sdf(mask, affine, SPACING[kind])
        meta = {"name": b, "kind": kind, "subject": subject,
                "source": f"VIVA+ v{VERSION} {posture}",
                "source_parts": sorted(parts[p] for p in groups[b]),
                "triangles": len(tris), "attribution": ATTRIBUTION, "license": "LGPL-3.0-or-later"}
        meta = save_bone(out / f"{b}.npz", values, origin, SPACING[kind], meta)
        log(f"  {b}: {len(tris)} triangles -> {meta['volume_mm3'] / 1000:.1f} cm3")
    (out / "ATTRIBUTION.txt").write_text(ATTRIBUTION + "\n")
    return out


def endplate_surfaces(model, posture="50F-standing"):
    """VIVA+'s own lumbar endplate shells: {(level, 'sup'|'inf'): points (viewer frame)}."""
    model = Path(model)
    parts = read_parts(model)
    want = {p: t for p, t in parts.items() if re.search(r"^AB-L[1-5]-Endplate-(superior|inferior)$", t)}
    shells, _ = read_elements(model, set(want))
    nodes = read_nodes(model, posture)
    out = {}
    for p, t in want.items():
        m = re.search(r"L([1-5])-Endplate-(superior|inferior)", t)
        ids = {i for q in shells.get(p, []) for i in q if i}
        out[(f"L{m.group(1)}", "sup" if m.group(2) == "superior" else "inf")] = to_ras(
            np.array([nodes[i] for i in ids if i in nodes]))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="50F-standing", help="50F-standing, 50F-seated, 50M-standing, 50M-seated")
    ap.add_argument("--out", required=True)
    ap.add_argument("--select", default="spine", help="spine, lumbar or a list (L3,L4,sacrum)")
    ap.add_argument("--cache", default="data/vivaplus", help="download folder")
    ap.add_argument("--keep-disc-heights", action="store_true",
                    help="keep the model's disc space heights (default: implicit.anatomy.template)")
    a = ap.parse_args(argv)
    convert(fetch(a.cache), a.out, a.model, a.select)
    from .landmarks import detect
    detect(a.out)
    if not a.keep_disc_heights:
        from .template import normalise
        normalise(a.out)  # rest disc heights to population values; detects landmarks again
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
