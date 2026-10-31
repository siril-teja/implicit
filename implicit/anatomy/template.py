"""Template disc spaces and facet joints: bring a template to population values.

    python -m implicit.anatomy.template assets/bones/vivaplus-50F [--dry-run]

The VIVA+ 50F spine has upper lumbar disc spaces about twice the population height
(implicit.anatomy.discs) and lumbar facet joints whose bones nearly touch (0.3 mm).
This step makes the template representative of the average population:

1. Disc spaces. The vertebrae move rigidly along the spine, without changing any
   orientation:
   - per disc, the shift d along its disc axis (bisector of the two endplate
     normals) that gives the population mean middle height (discs.NORMS);
   - a disc whose thinnest point (discs.thickness, over the whole endplates) would
     be below its floor (discs.thickness_floor) is opened until it is not;
   - the upper vertebra of the disc and everything above it move by d along that
     axis; the shifts add up the spine (the sacrum, hips and femurs stay).
   Only translations: the endplate angles, the sagittal and coronal Cobb angles and
   the pelvic parameters are unchanged. A, M and P change by the same amount, so the
   disc wedge (A - P) is the template's own. Discs without corners (T1-T2 of VIVA+
   50F, see discs.RMS_MAX) are not changed.

2. Joint spaces (``carve_joints``). The facets of the model fit its own disc
   heights, so closing a disc pushes its articular processes into the vertebra
   below; and even at the model's heights its lumbar facet joints are far narrower
   than in people. Every pair of neighbouring bones closer than the joint space g is
   carved back from the surface midway between them (an implicit Boolean cut on
   both bones: f = max(f, (f - f_other + g) / 2), 1-Lipschitz, so the bones end at
   least g apart, each losing only what that needs): g = 2.0 mm for the joints of the
   lumbar spine (T12-L1 to L5-S1: the lower normal limit of the lumbar facet joint
   space on CT, grade 0 of the Weishaupt et al. 1999 classification, as used by
   Tiwari et al. 2021) and 1.3 mm in the thoracic spine (mean thoracic facet joint
   space on CT, 1.27 +- 0.34 mm, 1512 joints, Tiwari et al. 2021). That space holds
   the articular cartilage (mean 0.57 mm a side in the lumbar spine, Woldtvedt et
   al. 2011) and the joint gap. Only bone where the two are less than g apart
   changes (the facets, where they touch); every disc space is thicker than its g.

The bone files are rewritten (shifted origins; carved values), landmarks are
detected again, and everything is recorded in ``template.json`` next to the bones.
Running it again changes nothing (shifts below TOL, joints already open).
implicit.anatomy.vivaplus runs it after converting a spine.

References
  Weishaupt D, Zanetti M, Boos N, Hodler J. MR imaging and CT in osteoarthritis of
    the lumbar facet joints. Skeletal Radiol 1999;28(4):215-9. doi:10.1007/s002560050503
  Tiwari P, Kaur H, Jha V. Is thoracic facet joint arthritis over-reported? It's time
    to review CT grading parameters. Indian J Radiol Imaging 2020;30(4):427-35.
    doi:10.4103/ijri.IJRI_390_20
  Woldtvedt DJ, Womack W, Gadomski BC, Schuldt D, Puttlitz CM. Finite element lumbar
    spine facet contact parameter predictions are affected by the cartilage thickness
    distribution and initial joint gap size. J Biomech Eng 2011;133(6):061009.
    doi:10.1115/1.4004287
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from . import clearance as cl
from . import discs
from .convert import VERTEBRAE, load_bone, load_bones, save_bone

TOL = 0.1    # mm: shifts below this are not applied (corner detection repeats to ~0.05 mm)
SCAN = 0.05  # mm: step of the search when a disc is opened to its thickness floor
FLOOR_MARGIN = 0.05  # mm: discs are opened this far past their thickness floor (sample rounding)
JOINT_SPACE_LUMBAR = 2.0    # mm (Weishaupt 1999 grade 0 lower limit, via Tiwari 2021)
JOINT_SPACE_THORACIC = 1.3  # mm (Tiwari 2021 mean, 1.27 mm)
JOINT_MARGIN = 0.1  # mm: carved this much wider (0.1 mm value quantisation, surface sampling)
LUMBAR = {"L1", "L2", "L3", "L4", "L5", "sacrum"}  # lower bones of the lumbar joints


def _bodies(landmarks):
    out = {}
    for lv, e in landmarks.items():
        corners = {ep: e[ep]["corners"] for ep in ("sup", "inf") if ep in e and "corners" in e[ep]}
        out["sacrum" if lv == "S1" else lv] = (corners or None, discs.IDENTITY)
    return out


def disc_shifts(landmarks, grids):
    """[(upper, lower, d, axis, (A, M, P) before, note)] per measured disc, d in mm;
    ``grids``: {bone name: Grid} of the subject."""
    bodies = _bodies(landmarks)
    samples = {n: cl.surface_samples(g) for n, g in grids.items()}
    out = []
    for u, l, *h in discs.heights(bodies):
        ref = discs.NORMS.get(discs.label(u, l))
        if ref is None:
            continue
        d = ref[1][0] - h[1]
        cu, cl_ = bodies[u][0]["inf"], bodies[l][0]["sup"]
        axis = np.asarray(cl_["normal"], float) - np.asarray(cu["normal"], float)
        axis /= np.linalg.norm(axis)
        note = ""
        # Thinnest point over the whole disc at least the disc's floor (discs.thickness).
        fl = discs.thickness_floor(u, l) + FLOOR_MARGIN
        th = lambda s: discs.thickness({u: (grids[u], samples[u], bodies[u][0], (np.eye(3), s * axis)),
                                        l: (grids[l], samples[l], bodies[l][0], discs.IDENTITY)})[(u, l)]
        if th(d) < fl:
            lo, s = d, d
            while th(s) < fl:
                lo, s = s, s + SCAN
            for _ in range(12):  # th(lo) < fl <= th(s)
                mid = 0.5 * (lo + s)
                lo, s = (mid, s) if th(mid) < fl else (lo, mid)
            note = f"opened to the thickness floor {fl:.2f} mm: {d:+.2f} -> {s:+.2f} mm"
            d = s
        out.append((u, l, d, axis, tuple(h[:3]), note))
    return out


def vertebra_shifts(shifts, names):
    """{vertebra: translation} from the disc shifts: each vertebra moves by the sum of
    the shifts of the discs below it."""
    order = [lv for lv in VERTEBRAE if lv in names]  # cranial to caudal
    disc_below = {u: (d, axis) for u, _, d, axis, *_ in shifts}
    out, acc = {}, np.zeros(3)
    for lv in reversed(order):  # from the lowest vertebra up
        if lv in disc_below:
            d, axis = disc_below[lv]
            acc = acc + d * axis
        out[lv] = acc.copy()
    return out


RIB_TIPS = range(1, 11)  # ribs whose cartilages reach the sternum or the costal margin


def joint_space(upper, lower):
    return JOINT_SPACE_LUMBAR if lower in LUMBAR else JOINT_SPACE_THORACIC


def rib_joints(names):
    """(vertebra, rib) pairs of the costovertebral and costotransverse joints present:
    rib n with T n (head and tubercle) and with T n-1 (upper half of the head)."""
    out = []
    for n in range(1, 13):
        for side in ("left", "right"):
            rib = f"rib{n}_{side}"
            if rib in names:
                out += [(v, rib) for v in (f"T{n - 1}", f"T{n}") if v in names]
    return out


def _carried(grid, controls):
    """``grid`` resampled through the displacement u(x) interpolated (inverse distance
    squared) from ``controls``: [(point, shift)]; the box moves by the mean shift."""
    from ..sdf import Grid
    C = np.array([c for c, _ in controls], float)
    U = np.array([u for _, u in controls], float)
    X = _points(grid) + U.mean(0)  # the new grid's samples
    d2 = ((X[:, None, :] - C[None]) ** 2).sum(-1) + 1.0
    w = 1.0 / d2
    u = (w @ U) / w.sum(1, keepdims=True)
    vals = grid.eval(X - u).reshape(grid.values.shape)
    return Grid(vals, grid.origin + U.mean(0), grid.spacing, grid.name)


def _points(g):
    """World coordinates of every sample of a grid, in the grid's array order."""
    axes = [g.origin[i] + g.spacing * np.arange(g.shape[i]) for i in range(3)]
    return np.stack(np.meshgrid(*axes, indexing="ij"), -1).reshape(-1, 3)


def carve_joints(grids, log=print):
    """Carve every pair of neighbouring bones closer than their joint space (see the
    module docstring, 2.) in place: {(upper, lower): (clearance before, after, mm3
    removed)}."""
    from ..sdf import Grid
    names = set(grids)
    levels = [lv for lv in VERTEBRAE if lv in names]
    pairs = list(zip(levels, levels[1:])) + ([(levels[-1], "sacrum")] if levels and "sacrum" in names else [])
    pairs += [p for p in rib_joints(names)]
    out = {}
    for u, l in pairs:
        gap = joint_space(u, l) + JOINT_MARGIN
        pair = lambda gu, gl: cl.clearance({u: (gu, discs.IDENTITY), l: (gl, discs.IDENTITY)},
                                           {u: cl.surface_samples(gu), l: cl.surface_samples(gl)})[0][2]
        gu0, gl0 = grids[u], grids[l]
        fu, fl = gu0.values, gl0.values
        lu = gl0.eval(_points(gu0)).reshape(fu.shape)  # the neighbour's field at each sample
        ul = gu0.eval(_points(gl0)).reshape(fl.shape)
        # Closer than the joint space: a point on either surface whose two distances add
        # up to less than the space (after a carve they never do, so a rerun does nothing;
        # the sampled clearance reads up to ~0.3 mm low, so it is not the test).
        need = joint_space(u, l) - JOINT_MARGIN  # values are stored to 0.1 mm
        if not (((np.abs(fu) < gu0.spacing) & (fu + lu < need)).any()
                or ((np.abs(fl) < gl0.spacing) & (fl + ul < need)).any()):
            continue
        c0 = pair(gu0, gl0)
        # Medial carving: each bone keeps what is at least gap / 2 on its side of the
        # surface where both fields are equal; (f - f_other + gap) / 2 is 1-Lipschitz, so
        # the carved bones are at least gap apart, and nothing changes where f + f_other >= gap.
        nu, nl = np.maximum(fu, 0.5 * (fu - lu + gap)), np.maximum(fl, 0.5 * (fl - ul + gap))
        removed = float(((fu < 0) & (nu >= 0)).sum() * gu0.spacing ** 3 + ((fl < 0) & (nl >= 0)).sum() * gl0.spacing ** 3)
        grids[u] = Grid(nu, gu0.origin, gu0.spacing, gu0.name)
        grids[l] = Grid(nl, gl0.origin, gl0.spacing, gl0.name)
        c = pair(grids[u], grids[l])
        out[(u, l)] = (c0, c, removed)
        log(f"  joint {u}-{l}: clearance {c0:+.2f} -> {c:.2f} mm (space {joint_space(u, l)} mm), {removed:.0f} mm3 removed")
    return out


def recorded_shifts(bone_dir):
    """{vertebra: total translation} recorded in template.json over all runs."""
    f = Path(bone_dir) / "template.json"
    out = {}
    rec = json.loads(f.read_text()) if f.exists() else None
    while rec:
        for lv, u in rec.get("vertebrae_mm", {}).items():
            if isinstance(u, list) and lv[0] in "TL" and lv[1:].isdigit():
                out[lv] = out.get(lv, np.zeros(3)) + np.asarray(u, float)
        rec = rec.get("previous")
    return out


def normalise(bone_dir, dry_run=False, log=print):
    """Shift the vertebrae of ``bone_dir`` and carve its joints (see the module
    docstring); returns the record."""
    from .landmarks import detect
    bone_dir = Path(bone_dir)
    lm = json.loads((bone_dir / "landmarks.json").read_text())["landmarks"]
    loaded = load_bones(bone_dir)
    grids = {n: g for n, (g, _) in loaded.items()}
    metas = {n: m for n, (_, m) in loaded.items()}
    shifts = disc_shifts(lm, grids)
    moves = vertebra_shifts(shifts, lm)
    record = {"method": "implicit.anatomy.template: vertebrae translated along the disc axes so that the "
                        "rest middle disc heights take the population means (implicit.anatomy.discs.NORMS); "
                        "neighbouring bones carved to the facet joint space",
              "discs": {discs.label(u, l): {"shift_mm": round(d, 3), "before_AMP_mm": [round(x, 2) for x in h],
                                            **({"note": n} if n else {})}
                        for u, l, d, _, h, n in shifts},
              "vertebrae_mm": {lv: [round(float(x), 3) for x in t] for lv, t in moves.items()}}
    for u, l, d, _, h, n in shifts:
        log(f"  {discs.label(u, l):>8}: A/M/P {h[0]:5.1f} {h[1]:5.1f} {h[2]:5.1f} -> shift {d:+6.2f} mm  {n}")
    if dry_run:
        return record
    # Ribs move with their vertebra; the sternum and costal cartilages with the rib ends.
    # A rib added after the vertebrae were shifted (no template_shift_mm yet) first
    # takes the vertebra's shift so far (template.json).
    prior = recorded_shifts(bone_dir)
    for n in range(1, 13):
        for side in ("left", "right"):
            r = f"rib{n}_{side}"
            if r in grids and f"T{n}" in lm:
                u = moves.get(f"T{n}", np.zeros(3))
                if "template_shift_mm" not in metas[r]:
                    u = u + prior.get(f"T{n}", np.zeros(3))
                moves[r] = u
    moved = {lv for lv, t in moves.items() if np.linalg.norm(t) >= TOL}
    tips = []
    for n in RIB_TIPS:
        for side in ("left", "right"):
            r = f"rib{n}_{side}"
            if r in grids:
                S = cl.surface_samples(grids[r])
                tips.append((S[np.argmax(S[:, 1])], moves.get(r, np.zeros(3))))  # most anterior point
    for lv in moved:
        g = grids[lv]
        grids[lv] = type(g).from_quantized(g.q, g.origin + moves[lv], g.spacing, g.name)
    carried = [n for n in ("sternum", "costal_cartilage_left", "costal_cartilage_right") if n in grids]
    if tips and any(np.linalg.norm(u) >= TOL for _, u in tips):
        for n in carried:
            grids[n] = _carried(grids[n], tips)
            moved.add(n)
            moves[n] = np.mean([u for _, u in tips], 0)
            record["vertebrae_mm"][n] = "carried by the rib ends (inverse distance squared)"
    for n, u in moves.items():
        if n.startswith("rib"):
            record["vertebrae_mm"][n] = [round(float(x), 3) for x in u]
    carved = carve_joints(grids, log)
    record["joints"] = {f"{u}-{l}": {"space_mm": joint_space(u, l), "clearance_before_mm": round(c0, 3),
                                     "clearance_after_mm": round(c, 3), "removed_mm3": round(v, 1)}
                        for (u, l), (c0, c, v) in carved.items()}
    changed = moved | {n for pair in carved for n in pair}
    if not changed:
        return record
    for lv in sorted(changed):
        path = bone_dir / f"{lv}.npz"
        g, meta = load_bone(path)
        if lv in carried and lv in moved:
            meta["template_carried"] = "resampled with the rib ends' shifts (implicit.anatomy.template)"
        elif lv in moved:
            prev = meta.get("template_shift_mm", [0.0, 0.0, 0.0])
            meta["template_shift_mm"] = [round(float(a + b), 4) for a, b in zip(prev, moves[lv])]
        cut = [f"{a}-{b}" for a, b in carved if lv in (a, b)]
        if cut:
            meta["template_joints_carved"] = sorted(set(meta.get("template_joints_carved", []) + cut))
        save_bone(path, grids[lv].values, grids[lv].origin, grids[lv].spacing, meta)
    detect(bone_dir, log=lambda *a: None)
    done = bone_dir / "template.json"
    if done.exists():  # keep the history of earlier runs
        record = {**record, "previous": json.loads(done.read_text())}
    done.write_text(json.dumps(record, indent=1) + "\n")
    log(f"wrote {done}")
    return record


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("bone_dir")
    ap.add_argument("--dry-run", action="store_true", help="print the shifts only")
    a = ap.parse_args(argv)
    normalise(a.bone_dir, a.dry_run)


if __name__ == "__main__":
    main()
