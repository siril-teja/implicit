"""Feasible range of each disc: how far it may turn before bones collide or the disc
is deformed beyond its rules.

    python -m implicit.anatomy.ranges assets/bones/vivaplus-50F

Every constraint of the constrained distribution (roadmap 1.7) involves one disc
only: the bones that can touch across a disc are the two vertebrae of that disc and
the ribs that move with them (rib n with T n: rib n against T n-1 and against rib
n-1), and the disc's heights and thickness depend on its two endplates alone. So the
feasible set of the whole spine is the product of the feasible sets of the discs,
and each disc's set is found once, from the bone fields, per plane and direction:
the disc (the upper vertebra and its ribs) is turned about its rotation centres
(implicit.anatomy.rotation; sagittal about the left-right axis, coronal about the
antero-posterior axis, axial about the upper vertebra's own axis) away from rest,
until
  - the clearance of a pair across the disc (implicit.anatomy.clearance) falls below
    min(CLEARANCE, its rest clearance): about 1 mm at the facets (the roadmap's
    figure; the joint spaces are 1.3-2.0 mm at rest), or
  - a disc posture rule fails (implicit.anatomy.discs: middle height +-10 %, local
    +-40 %, thinnest point above its floor),
or it reaches half its range of motion (implicit.anatomy.motion; beyond that the
limit is the range itself). The angle is found by a scan of SCAN_STEPS steps, then
BISECT_STEPS bisections of the first failing step (to about 0.002 deg).

Stored per disc with its upper vertebra's landmarks as "range": {plane: [most
negative turn, most positive turn] (deg, signed as the chain model's turns), "why":
{plane+sign: what stops it}}.

The planes are found one at a time; a posture combining two planes near their
limits can reach a little past them (the tests check combined postures on a grid).
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from . import clearance as cl
from . import discs
from .convert import load_bone

CLEARANCE = 1.0     # mm: smallest gap across a disc (or the rest gap, when smaller)
SCAN_STEPS = 8
BISECT_STEPS = 10
PLANES = ("FE", "LB", "AR")
IDENTITY = (np.eye(3), np.zeros(3))


def _corners(lm, name):
    e = lm.get("S1" if name == "sacrum" else name, {})
    return {ep: e[ep]["corners"] for ep in ("sup", "inf") if ep in e and "corners" in e[ep]}


def disc_ranges(bone_dir, doc, log=print):
    """{upper vertebra: range entry} for every disc of the chain model of ``doc``
    (landmarks.json content) whose bones are in ``bone_dir``."""
    from . import alignment as al
    bone_dir = Path(bone_dir)
    lm = doc["landmarks"]
    data = al.chain(lm, doc.get("pelvis"))
    if data is None:
        return {}
    grids, samples = {}, {}

    def bone(n):
        if n not in grids:
            f = bone_dir / f"{n}.npz"
            if not f.exists():
                return None
            grids[n] = load_bone(f)[0]
            samples[n] = cl.surface_samples(grids[n])
        return grids[n]

    ribs = lambda lv: [f"rib{lv[1:]}_{s}" for s in ("left", "right")] if lv[0] == "T" else []
    out, levels = {}, data["levels"]
    for j, disc in enumerate(data["discs"]):
        up, lo = disc["upper"], (levels[j - 1] if j else "sacrum")
        moving = [b for b in [up] + ribs(up) if bone(b) is not None]
        fixed = [b for b in [lo] + (ribs(lo) if lo != "sacrum" else []) if bone(b) is not None]
        if up not in moving or lo not in fixed or "rom" not in disc:
            continue
        names = moving + fixed

        def across(M):
            bodies = {b: (grids[b], M if b in moving else IDENTITY) for b in names}
            return [(a, b, c) for a, b, c, _ in cl.clearance(bodies, {b: samples[b] for b in names})
                    if (a in moving) != (b in moving)]

        rest = {(a, b): c for a, b, c in across(IDENTITY)}
        cu, cd = _corners(lm, up), _corners(lm, lo)
        h0 = discs.heights({up: (cu, IDENTITY), lo: (cd, IDENTITY)}) if cu and cd else []

        def fails(M):
            for a, b, c in across(M):
                if c < min(CLEARANCE, rest[(a, b)]) - 1e-6:
                    return f"clearance {a}-{b}"
            if h0:
                now = discs.heights({up: (cu, M), lo: (cd, IDENTITY)})
                th = discs.thickness({up: (grids[up], samples[up], cu, M), lo: (grids[lo], samples[lo], cd, IDENTITY)})
                flags = discs.posture_flags(now, h0, th)
                if flags:
                    return "disc " + "; ".join(sum(flags.values(), []))
            return None

        axes = [(disc["sagittal"], al.X_AXIS), (disc["coronal"], al.Y_AXIS),
                (data["frames"][up]["center"], data["frames"][up]["axes"][2])]
        entry, why = {}, {}
        for p, (c, ax) in enumerate(axes):
            half = math.radians(disc["rom"][p]) / 2
            ends = []
            for sgn in (-1, 1):
                turn = lambda a: tuple(np.asarray(x, float) for x in al._rot(c, ax, sgn * a))
                end, reason, prev = half, "range of motion", 0.0
                for k in range(1, SCAN_STEPS + 1):
                    a = half * k / SCAN_STEPS
                    r = fails(turn(a))
                    if r:
                        lo_a, hi_a = prev, a
                        for _ in range(BISECT_STEPS):
                            mid = 0.5 * (lo_a + hi_a)
                            if fails(turn(mid)):
                                hi_a = mid
                            else:
                                lo_a = mid
                        end, reason = lo_a, r
                        break
                    prev = a
                ends.append(round(sgn * math.degrees(end), 3))
                why[f"{PLANES[p]}{'+' if sgn > 0 else '-'}"] = reason
            entry[PLANES[p]] = ends
        entry["why"] = why
        out[up] = entry
        tight = {k: v for k, v in why.items() if v != "range of motion"}
        log(f"  {disc['label']:>8}: " + "  ".join(f"{k} {entry[k][0]:+.2f}/{entry[k][1]:+.2f}" for k in PLANES)
            + (f"  ({'; '.join(f'{k}: {v}' for k, v in tight.items())})" if tight else ""))
    return out


def add_ranges(bone_dir, log=print):
    """Compute the disc ranges of ``bone_dir`` and store them in its landmarks.json."""
    f = Path(bone_dir) / "landmarks.json"
    doc = json.loads(f.read_text())
    for lv, entry in disc_ranges(bone_dir, doc, log).items():
        doc["landmarks"][lv]["range"] = entry
    f.write_text(json.dumps(doc, indent=1))
    log(f"wrote {f}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("bone_dir")
    add_ranges(ap.parse_args(argv).bone_dir)


if __name__ == "__main__":
    main()
