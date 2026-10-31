"""Clearance between neighbouring bones: do the posed bones overlap?

    python -m implicit.anatomy.clearance examples/average_spine.json [--sweep]

For each pair of neighbouring bones (PAIRS) the clearance is the smallest value of
one bone's distance field at the other bone's surface points, both ways: negative
where the bones overlap (about the depth of the overlap), else roughly the gap.
Each bone is its own implicit grid moved by the rigid motion its blocks apply (Pose,
Align); smoothing is not included (clearance of the unsmoothed bones).

Surface points: grid points within half a cell of the bone surface, thinned to one
per SAMPLE_STEP cube (the first in C order), so the clearance is found to about
SAMPLE_STEP / 2 on curved surfaces. viewer/clearance.js is the same computation on
the same points and grid values.
"""

from __future__ import annotations

import argparse
import itertools
import json

import numpy as np

from .convert import VERTEBRAE

SAMPLE_STEP = 2.0   # mm, surface point spacing
COLLIDE = 0.0       # mm: clearance below this is a collision


def pairs(names):
    """Neighbouring bone pairs among ``names``: consecutive vertebrae, the lowest one
    with the sacrum, the sacrum with each hip bone, L5 with each hip bone, each hip
    bone with its femur."""
    names = set(names)
    levels = [lv for lv in VERTEBRAE if lv in names]
    out = list(zip(levels, levels[1:]))
    if levels and "sacrum" in names:
        out.append((levels[-1], "sacrum"))
    for side in ("left", "right"):
        hip, femur = f"hip_{side}", f"femur_{side}"
        if hip in names:
            out += [(a, b) for a, b in (("sacrum", hip), ("L5", hip), (hip, femur)) if a in names and b in names]
    return out


def surface_samples(grid):
    """Surface points of a bone grid (see the module docstring), rounded to 0.01 mm."""
    v = grid.values
    idx = np.argwhere(np.abs(v) < 0.5 * grid.spacing)
    P = grid.origin + grid.spacing * idx
    cell = np.floor((P - P.min(0)) / SAMPLE_STEP).astype(np.int64)
    key = (cell[:, 0] * 100003 + cell[:, 1]) * 100003 + cell[:, 2]
    _, first = np.unique(key, return_index=True)
    return np.round(P[np.sort(first)], 2)


def _apply(M, P):
    R, t = M
    return P @ R.T + t


def _inverse(M):
    R, t = M
    return R.T, -R.T @ t


def clearance(bodies, samples):
    """[(a, b, clearance, point)] for the neighbouring pairs of ``bodies``.

    ``bodies``: {name: (grid, (R, t))}, the bone grid and its current rigid motion;
    ``samples``: {name: surface points} in the grid's own (unmoved) frame. ``point``
    is where the smallest value was found (scene frame).
    """
    out = []
    for a, b in pairs(bodies):
        best = (np.inf, None)
        for s, o in ((a, b), (b, a)):
            (_, Ms), (go, Mo) = bodies[s], bodies[o]
            W = _apply(Ms, samples[s])  # moved surface points of s
            d = go.eval(_apply(_inverse(Mo), W))
            i = int(np.argmin(d))
            if d[i] < best[0]:
                best = (float(d[i]), W[i])
        out.append((a, b, best[0], best[1]))
    return out


def scene_bodies(scene):
    """{name: (grid, (R, t))} of the scene's bodies that are moved bones (graph scenes)."""
    out = {}
    for b in scene.bodies:
        rig = getattr(b, "rigid", None)
        if rig is not None:
            grid, motion = rig
            out[b.name] = (grid, motion())
    return out


def report(scene, samples):
    """Clearance of a graph scene at its current parameter values."""
    return clearance(scene_bodies(scene), samples)


def sweep(scene, samples, grid):
    """Collisions over a grid of slider values: {setting: [colliding pairs]}."""
    names = list(grid)
    out = {}
    defaults = {p.name: p.value() for p in scene.params()}
    for combo in itertools.product(*(grid[n] for n in names)):
        scene.set_params({**defaults, **dict(zip(names, combo))})
        hits = [(a, b, round(c, 2)) for a, b, c, _ in report(scene, samples) if c < COLLIDE]
        out[combo] = hits
    scene.set_params(defaults)
    return names, out


def main(argv=None):
    from ..graph import load_graph
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("graph")
    ap.add_argument("--set", action="append", default=[], help="name=value, e.g. spp_LL=80")
    ap.add_argument("--sweep", action="store_true", help="collision counts over a grid of slider values")
    a = ap.parse_args(argv)
    scene = load_graph(a.graph)
    bodies = scene_bodies(scene)
    samples = {n: surface_samples(g) for n, (g, _) in bodies.items()}
    scene.set_params({k: float(v) for k, v in (s.split("=") for s in a.set)})
    for x, y, c, p in report(scene, samples):
        print(f"{x:>10} - {y:<12} {c:7.2f} mm{'  COLLISION' if c < COLLIDE else ''}")
    if a.sweep:
        names = {p.name for p in scene.params()}
        grid = {k: v for k, v in {"spp_LL": [20, 52, 80], "spp_TK": [10, 30, 60], "spp_PT": [0, 17, 35],
                                  "spp_MT": [-40, 0, 40], "spp_TL": [-40, 0, 40]}.items() if k in names}
        keys, res = sweep(scene, samples, grid)
        n = sum(1 for v in res.values() if v)
        print(f"\nsweep over {', '.join(keys)}: {n} of {len(res)} settings with collisions")
        counts = {}
        for hits in res.values():
            for x, y, _ in hits:
                counts[(x, y)] = counts.get((x, y), 0) + 1
        for (x, y), k in sorted(counts.items(), key=lambda kv: -kv[1]):
            print(f"  {x}-{y}: {k}")
        print(json.dumps({"settings": len(res), "colliding": n}))


if __name__ == "__main__":
    main()
