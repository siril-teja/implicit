"""Iso-surface extraction (naive surface nets) and STL export, NumPy only.

Surface nets places one vertex per sign-changing grid cell (at the mean of the
edge crossings) and one quad per sign-changing grid edge. Output is a closed,
consistently oriented (outward) triangle mesh when the grid encloses the body.
This is a surface for inspection / STL; FE volume meshing is a later stage.
"""

from __future__ import annotations

import struct

import numpy as np

_CORNERS = np.array([[i, j, k] for i in (0, 1) for j in (0, 1) for k in (0, 1)])
_EDGES = [
    (a, b) for a in range(8) for b in range(a + 1, 8)
    if np.abs(_CORNERS[a] - _CORNERS[b]).sum() == 1
]


def surface_nets(f: np.ndarray, origin, h):
    """Return (vertices[V, 3], triangles[T, 3]) for the zero level set of ``f``."""
    f = np.pad(np.asarray(f, dtype=float), 1, constant_values=1e10)
    origin = np.asarray(origin, dtype=float) - h
    inside = f < 0
    n = np.array(f.shape)
    cs = tuple(n - 1)  # cell grid shape

    def corner(c):
        dx, dy, dz = _CORNERS[c]
        return f[dx:dx + cs[0], dy:dy + cs[1], dz:dz + cs[2]]

    acc = np.zeros(cs + (3,))
    cnt = np.zeros(cs)
    for a, b in _EDGES:
        fa, fb = corner(a), corner(b)
        m = (fa < 0) != (fb < 0)
        if not m.any():
            continue
        t = np.zeros(cs)
        t[m] = fa[m] / (fa[m] - fb[m])
        pos = _CORNERS[a] + t[..., None] * (_CORNERS[b] - _CORNERS[a])
        acc[m] += pos[m]
        cnt[m] += 1

    active = cnt > 0
    vid = -np.ones(cs, dtype=np.int64)
    vid[active] = np.arange(active.sum())
    cell_idx = np.argwhere(active)
    verts = origin + h * (cell_idx + acc[active] / cnt[active][:, None])

    tris = []
    for ax in range(3):
        b, c = (ax + 1) % 3, (ax + 2) % 3
        lo_sl = [slice(None)] * 3
        hi_sl = [slice(None)] * 3
        lo_sl[ax], hi_sl[ax] = slice(0, -1), slice(1, None)
        lo, hi = inside[tuple(lo_sl)], inside[tuple(hi_sl)]
        ch = lo != hi
        idx = np.argwhere(ch)
        ok = (idx[:, b] >= 1) & (idx[:, c] >= 1) & (idx[:, b] <= n[b] - 2) & (idx[:, c] <= n[c] - 2)
        idx = idx[ok]
        outward_pos = lo[ch][ok]  # inside on the low side -> normal along +ax
        quad = []
        for ob, oc in ((-1, -1), (0, -1), (0, 0), (-1, 0)):  # CCW about +ax
            cell = idx.copy()
            cell[:, b] += ob
            cell[:, c] += oc
            quad.append(vid[cell[:, 0], cell[:, 1], cell[:, 2]])
        q = np.stack(quad, axis=1)
        q[~outward_pos] = q[~outward_pos][:, ::-1]
        tris.append(q[:, [0, 1, 2]])
        tris.append(q[:, [0, 2, 3]])
    return verts, np.concatenate(tris).astype(np.int64)


def mesh_volume(v, t):
    a, b, c = v[t[:, 0]], v[t[:, 1]], v[t[:, 2]]
    return np.einsum("ij,ij->i", a, np.cross(b, c)).sum() / 6.0


def write_stl(path, v, t, name="implicit"):
    a, b, c = v[t[:, 0]], v[t[:, 1]], v[t[:, 2]]
    nrm = np.cross(b - a, c - a)
    nrm /= np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-30)
    rec = np.zeros(len(t), dtype=[("n", "<f4", 3), ("v", "<f4", (3, 3)), ("attr", "<u2")])
    rec["n"] = nrm
    rec["v"] = np.stack([a, b, c], axis=1)
    with open(path, "wb") as fh:
        fh.write(struct.pack("<80sI", name.encode()[:80], len(t)))
        fh.write(rec.tobytes())
