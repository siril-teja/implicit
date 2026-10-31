"""Template disc space heights: bring a template's rest disc heights to population values.

    python -m implicit.anatomy.template assets/bones/vivaplus-50F [--dry-run]

The VIVA+ 50F spine has upper lumbar disc spaces about twice the population height
(implicit.anatomy.discs). This step moves the vertebrae rigidly along the spine so
that the disc spaces take the population's middle height, without changing any
orientation:

  - per disc, the shift d along its disc axis (bisector of the two endplate normals)
    that gives the population mean middle height (implicit.anatomy.discs.NORMS);
  - a disc is not closed so far that its two vertebrae come closer than the
    unmodified template's closest pair is at rest (the floor: the smallest clearance
    of any neighbouring pair, implicit.anatomy.clearance; kept in template.json for
    later runs): the facets of the template
    fit its own disc heights, and closing a disc further would push its articular
    processes into the vertebra below. The shift stops there (bisection), so such
    a disc ends between its old height and the population mean;
  - a disc whose thinnest point (discs.thickness, over the whole endplates) would
    be below its floor (discs.thickness_floor) is opened until it is not;
  - the upper vertebra of the disc and everything above it move by d along that
    axis; the shifts add up the spine (the sacrum, hips and femurs stay).

Only translations: every vertebra keeps its orientation, so the endplate angles, the
sagittal and coronal Cobb angles and the pelvic parameters are unchanged. A, M and
P change by the same amount, so the disc wedge (A - P) is the template's own, and
A and P can stay outside the population range where the template's wedge or endplate
shape differs from the population's. Discs without corners (T1-T2 of VIVA+ 50F, see
discs.RMS_MAX) are not changed: the vertebra above moves with the one below.

The bone files are rewritten with shifted origins (the grids themselves are
untouched, so nothing is resampled), landmarks are detected again, and the shifts
are recorded in ``template.json`` next to the bones. Running it again finds shifts below
TOL. implicit.anatomy.vivaplus runs it after converting a spine.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from . import clearance as cl
from . import discs
from .convert import VERTEBRAE, load_bone, load_bones, save_bone

TOL = 0.2    # mm: shifts below this are not applied: corner detection repeats to ~0.05 mm, and
             # near the floor the clearance changes by only ~0.04 mm per mm of shift (L1-L2)
SCAN = 0.05  # mm: step of the search for the first contact when a disc is closed
FLOOR_MARGIN = 0.05  # mm: discs are opened this far past their thickness floor (sample rounding)


def _bodies(landmarks):
    out = {}
    for lv, e in landmarks.items():
        corners = {ep: e[ep]["corners"] for ep in ("sup", "inf") if ep in e and "corners" in e[ep]}
        out["sacrum" if lv == "S1" else lv] = (corners or None, discs.IDENTITY)
    return out


def _pair_clearance(grids, samples, u, l, t):
    """Clearance of vertebra u moved by t against l (unmoved)."""
    bodies = {u: (grids[u], (np.eye(3), t)), l: (grids[l], discs.IDENTITY)}
    return cl.clearance(bodies, {u: samples[u], l: samples[l]})[0][2]


def disc_shifts(landmarks, grids, floor=None):
    """([(upper, lower, d, axis, (A, M, P) before, note)] per measured disc, d in mm;
    the clearance floor); ``grids``: {bone name: Grid} of the subject; ``floor``: the
    clearance floor (default: the smallest clearance of the subject as it is)."""
    bodies = _bodies(landmarks)
    samples = {n: cl.surface_samples(g) for n, g in grids.items()}
    if floor is None:
        rest = cl.clearance({n: (g, discs.IDENTITY) for n, g in grids.items()}, samples)
        floor = min(c for _, _, c, _ in rest)
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
        if d < 0:  # closing: stop at the first contact with the floor
            c = lambda s: _pair_clearance(grids, samples, u, l, s * axis)
            prev, s = 0.0, 0.0
            while s > d:
                s = max(s - SCAN, d)
                if c(s) < floor:
                    lo, hi = s, prev  # c(hi) >= floor > c(lo)
                    for _ in range(12):
                        mid = 0.5 * (lo + hi)
                        lo, hi = (lo, mid) if c(mid) >= floor else (mid, hi)
                    note = f"limited by clearance (floor {floor:.2f} mm): {d:+.2f} -> {hi:+.2f} mm"
                    d = hi
                    break
                prev = s
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
            note = (note + "; " if note else "") + f"opened to the thickness floor {fl:.2f} mm: {d:+.2f} -> {s:+.2f} mm"
            d = s
        out.append((u, l, d, axis, tuple(h[:3]), note))
    return out, floor


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


def normalise(bone_dir, dry_run=False, log=print):
    """Shift the vertebrae of ``bone_dir`` (see the module docstring); returns the record."""
    from .landmarks import detect
    bone_dir = Path(bone_dir)
    lm = json.loads((bone_dir / "landmarks.json").read_text())["landmarks"]
    grids = {n: g for n, (g, _) in load_bones(bone_dir).items()}
    done = bone_dir / "template.json"
    # The floor is the unmodified template's: a later run keeps it, so it finds no shift.
    first = json.loads(done.read_text()) if done.exists() else None
    while first and "previous" in first:
        first = first["previous"]
    shifts, floor = disc_shifts(lm, grids, first["clearance_floor_mm"] if first else None)
    moves = vertebra_shifts(shifts, lm)
    record = {"method": "implicit.anatomy.template: vertebrae translated along the disc axes so that the "
                        "rest middle disc heights take the population means (implicit.anatomy.discs.NORMS), "
                        "no pair closer than the template's closest pair at rest",
              "clearance_floor_mm": round(floor, 4),
              "discs": {discs.label(u, l): {"shift_mm": round(d, 3), "before_AMP_mm": [round(x, 2) for x in h],
                                            **({"note": n} if n else {})}
                        for u, l, d, _, h, n in shifts},
              "vertebrae_mm": {lv: [round(float(x), 3) for x in t] for lv, t in moves.items()}}
    log(f"  clearance floor (closest pair at rest): {floor:.2f} mm")
    for u, l, d, _, h, n in shifts:
        log(f"  {discs.label(u, l):>8}: A/M/P {h[0]:5.1f} {h[1]:5.1f} {h[2]:5.1f} -> shift {d:+6.2f} mm  {n}")
    if dry_run or all(abs(d) < TOL for _, _, d, *_ in shifts):
        return record
    for lv, t in moves.items():
        if np.linalg.norm(t) < TOL:
            continue
        path = bone_dir / f"{lv}.npz"
        g, meta = load_bone(path)
        prev = meta.get("template_shift_mm", [0.0, 0.0, 0.0])
        meta["template_shift_mm"] = [round(float(a + b), 4) for a, b in zip(prev, t)]
        save_bone(path, g.values, g.origin + t, g.spacing, meta)
    detect(bone_dir, log=lambda *a: None)
    if done.exists():  # keep the history of earlier runs
        record = {**record, "previous": json.loads(done.read_text())}
    (bone_dir / "template.json").write_text(json.dumps(record, indent=1) + "\n")
    log(f"wrote {bone_dir / 'template.json'}")
    return record


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("bone_dir")
    ap.add_argument("--dry-run", action="store_true", help="print the shifts only")
    a = ap.parse_args(argv)
    normalise(a.bone_dir, a.dry_run)


if __name__ == "__main__":
    main()
