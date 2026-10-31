"""Probe points with NumPy reference values, for the GPU diagnostics page.

For every shader unit of a scene (one body, or bodies sharing a ``share`` key)
the page evaluates the unit's GLSL at these points on the viewer's GPU and
compares with the NumPy value, then traces one ray to the unit's deepest point.
Grid bodies get points deep inside, at the grid centre, on the surface and
outside the grid; every unit also gets points sampled across its bounds.
Template instances (data-texture driven) are not probed.
"""

from __future__ import annotations

import numpy as np

from .sdf import Grid, GridSmooth


def _grid_points(g: Grid):
    q = g.q
    picks = [np.unravel_index(np.argmin(q), q.shape), tuple(np.array(q.shape) // 2)]
    surface = np.argwhere(q == 128)
    if len(surface):
        picks.append(tuple(surface[len(surface) // 2]))
    pts = [g.origin + g.spacing * np.array(i, float) for i in picks]
    lo, hi = g.bounds()
    pts.append(lo - 4.0)                                   # outside, below the grid
    pts.append(np.array([hi[0] + 6.0, (lo[1] + hi[1]) / 2, (lo[2] + hi[2]) / 2]))
    return pts


def probes(scene, n_random=8, seed=0):
    """Probe batches: slider defaults, then every slider moved away from its default.

    {"batches": [{"params": {name: value}, "units": [{"unit", "name", "points": [[x, y, z, d]],
    "target": [x, y, z]}]}]}. The second batch exercises parameter-dependent code
    paths (e.g. blending between smoothing levels) that defaults never reach.
    """
    params = scene.params()
    defaults = {p.name: p.value() for p in params}
    moved = {p.name: round(p.lo + 0.37 * (p.hi - p.lo), 4) for p in params
             if not getattr(p, "solver", None)}  # computed values follow their inputs
    batches = []
    try:
        for values in ({}, moved):
            scene.set_params({**defaults, **values})
            batches.append({"params": values, "units": _units(scene, n_random, seed),
                            "clearance": _clearance(scene), "discs": _discs(scene)})
    finally:
        scene.set_params(defaults)
    return {"batches": batches}


def _clearance(scene):
    """Clearance of neighbouring bones (graph scenes with moved bones): [[a, b, mm]]."""
    from .anatomy.clearance import report, surface_samples, scene_bodies
    bodies = scene_bodies(scene)
    if not bodies:
        return []
    cache = scene.__dict__.setdefault("_samples", {})
    for n, (g, _) in bodies.items():
        if n not in cache:
            cache[n] = surface_samples(g)
    return [[a, b, round(c, 6)] for a, b, c, _ in report(scene, cache)]


def _discs(scene):
    """Disc space heights of neighbouring vertebrae: [[upper, lower, A, M, P, L, R]] (mm)."""
    from .anatomy.discs import report
    return [[u, l, *(round(x, 6) for x in h)] for u, l, *h in report(scene)[0]]


def _now(body):
    """Body bounds at the current parameter values when known (graph bodies), else its bounds."""
    f = getattr(body, "bounds_now", None)
    b = f() if f else None
    return b if b is not None else body.bounds


def _units(scene, n_random, seed):
    rng = np.random.default_rng(seed)
    out = []
    for k, unit in enumerate(scene._units()):
        bodies = [scene.bodies[i] for i in unit]
        sdfs = [b.sdf for b in bodies]
        bounds = [_now(b) for b in bodies]
        lo, hi = (np.array(x) for x in (bounds[0] or scene.bounds))
        for b in bounds[1:]:
            if b:
                lo, hi = np.minimum(lo, b[0]), np.maximum(hi, b[1])
        pts = [lo + (hi - lo) * rng.random(3) for _ in range(n_random)]
        for s in sdfs:
            if isinstance(s, Grid):
                pts += _grid_points(s)
            elif isinstance(s, GridSmooth):
                pts += _grid_points(s.grid)
        # Deepest point on a coarse lattice: the ray-probe target.
        axes = [np.linspace(lo[i], hi[i], 24) for i in range(3)]
        lattice = np.stack(np.meshgrid(*axes, indexing="ij"), -1).reshape(-1, 3)
        dl = np.min([s.eval(lattice) for s in sdfs], axis=0)
        pts.append(lattice[np.argmin(np.abs(dl))])           # near the surface
        target = lattice[np.argmin(dl)] if dl.min() < 0 else None
        P = np.array(pts)
        d = np.min([s.eval(P) for s in sdfs], axis=0)
        out.append({"unit": k, "name": ", ".join(b.name for b in bodies),
                    "points": np.column_stack([P, d]).round(5).tolist(),
                    "target": None if target is None else target.round(5).tolist()})
    return out
