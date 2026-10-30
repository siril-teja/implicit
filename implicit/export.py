"""Mesh bodies of a scene to STL at chosen parameter values.

    python -m implicit.export examples/average_spine.json --body L3 --set L3_smooth_sigma=1.5 --voxel 0.4 --out out/
    python -m implicit.export examples/average_spine.json --out out/            # every visible body

Each body is meshed with narrow-band marching tetrahedra at spacing ``--voxel``
(mm): closed, manifold, outward-oriented triangles (see implicit.mesh.mesh_sdf).
The report lists triangles, volume, area and whether every edge is shared by
exactly two triangles.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np

from .mesh import mesh_sdf, mesh_volume, write_stl
from .project import load_scene


def mesh_report(v, t):
    a, b, c = v[t[:, 0]], v[t[:, 1]], v[t[:, 2]]
    area = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1).sum()
    e = np.sort(np.concatenate([t[:, [0, 1]], t[:, [1, 2]], t[:, [2, 0]]]), axis=1)
    _, counts = np.unique(e, axis=0, return_counts=True)
    return {"triangles": len(t), "volume_mm3": float(mesh_volume(v, t)), "area_mm2": float(area),
            "closed": bool(np.all(counts == 2))}


def mesh_body(scene, name, voxel=0.5):
    """Closed mesh of one scene body at the scene's current parameter values."""
    body = scene.body(name)
    now = getattr(body, "bounds_now", None)  # graph bodies: bounds at the current values
    lo, hi = (now() if now else None) or body.bounds or scene.bounds
    return mesh_sdf(body.sdf.eval, np.array(lo) - 2 * voxel, np.array(hi) + 2 * voxel, voxel)


def export(script, out, bodies=None, values=None, voxel=0.5):
    scene = load_scene(script)
    if values:
        unknown = set(values) - {p.name for p in scene.params()}
        if unknown:
            raise SystemExit(f"unknown parameters: {', '.join(sorted(unknown))}")
        scene.set_params(values)
    names = bodies or [b.name for b in scene.bodies if b.visible and b.template is None]
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    reports = {}
    for name in names:
        v, t = mesh_body(scene, name, voxel)
        if not len(t):
            print(f"  {name}: empty, skipped")
            continue
        path = out / (re.sub(r"[^A-Za-z0-9_.-]+", "_", name) + ".stl")
        write_stl(path, v, t, name)
        r = reports[name] = mesh_report(v, t)
        print(f"  {path}: {r['triangles']} triangles, {r['volume_mm3'] / 1000:.2f} cm3, "
              f"{r['area_mm2'] / 100:.1f} cm2, {'closed' if r['closed'] else 'NOT closed'}")
    return reports


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("script", help="scene script or block graph, e.g. examples/average_spine.json")
    ap.add_argument("--body", action="append", help="body to mesh (repeatable; default: all visible)")
    ap.add_argument("--set", action="append", default=[], metavar="NAME=VALUE", help="parameter value")
    ap.add_argument("--voxel", type=float, default=0.5, help="sampling grid spacing (mm)")
    ap.add_argument("--out", default="out", help="output folder")
    a = ap.parse_args(argv)
    values = {}
    for kv in a.set:
        k, _, v = kv.partition("=")
        values[k] = float(v)
    export(a.script, a.out, a.body, values, a.voxel)


if __name__ == "__main__":
    main()
