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


# ---------------------------------------------------------------- robust meshing
# Cube corners (x, y, z) and the 6 tetrahedra around the main diagonal 0-6. All
# cubes use the same diagonal, so neighbouring cubes split their shared faces
# identically and the surface is closed and manifold by construction.
_CUBE = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
                  [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]])
_TETS = np.array([[0, 5, 1, 6], [0, 1, 2, 6], [0, 2, 3, 6],
                  [0, 3, 7, 6], [0, 7, 4, 6], [0, 4, 5, 6]])


def _tet_triangles(ids, vals):
    """Marching tetrahedra on (T, 4) global node ids and values.

    Returns (edges (F, 3, 2) node-id pairs per triangle vertex, (F,) inside-node,
    (F,) outside-node) for orientation.
    """
    inside = vals < 0
    n_in = inside.sum(1)
    tris, ins, outs = [], [], []
    order = np.argsort(~inside, axis=1, kind="stable")  # inside nodes first
    srt = np.take_along_axis(ids, order, 1)
    # one inside node: triangle on its three edges
    m = n_in == 1
    if m.any():
        a, b, c, d = srt[m].T
        tris.append(np.stack([np.stack([a, b], -1), np.stack([a, c], -1), np.stack([a, d], -1)], 1))
        ins.append(a)
        outs.append(b)
    # three inside nodes: triangle on the edges to the single outside node
    m = n_in == 3
    if m.any():
        a, b, c, d = srt[m].T
        tris.append(np.stack([np.stack([a, d], -1), np.stack([b, d], -1), np.stack([c, d], -1)], 1))
        ins.append(a)
        outs.append(d)
    # two inside: quad a-c, a-d, b-d, b-c split into two triangles
    m = n_in == 2
    if m.any():
        a, b, c, d = srt[m].T
        ac, ad, bd, bc = (np.stack(e, -1) for e in ((a, c), (a, d), (b, d), (b, c)))
        tris.append(np.stack([ac, ad, bd], 1))
        tris.append(np.stack([ac, bd, bc], 1))
        ins += [a, a]
        outs += [c, c]
    if not tris:
        return np.zeros((0, 3, 2), np.int64), np.zeros(0, np.int64), np.zeros(0, np.int64)
    return np.concatenate(tris), np.concatenate(ins), np.concatenate(outs)


def mesh_sdf(f, lo, hi, h, block=8, chunk=4096):
    """Closed, manifold triangle mesh of {f < 0} inside the box [lo, hi].

    ``f`` maps (N, 3) points to signed distances and must be 1-Lipschitz (any
    SDF of this library). Narrow band: f is evaluated on block corners first
    and only blocks that can contain the surface (|f| <= block diagonal at some
    corner) are sampled at spacing ``h``. Returns (vertices, triangles).
    """
    lo = np.asarray(lo, float)
    hi = np.asarray(hi, float)
    nb = np.maximum(np.ceil((hi - lo) / (h * block)).astype(int), 1)
    n = nb * block + 1                                   # fine nodes per axis
    stride = np.array([1, n[0], n[0] * n[1]], np.int64)  # global node id = i + nx*(j + ny*k)
    # coarse pass on block corners
    axes = [lo[i] + h * block * np.arange(nb[i] + 1) for i in range(3)]
    C = f(np.stack(np.meshgrid(*axes, indexing="ij"), -1).reshape(-1, 3)).reshape(nb + 1)
    reach = np.sqrt(3) * h * block
    corners = np.stack([C[dx:dx + nb[0], dy:dy + nb[1], dz:dz + nb[2]]
                        for dx in (0, 1) for dy in (0, 1) for dz in (0, 1)], -1)
    active = np.argwhere((corners.min(-1) <= reach) & (corners.max(-1) >= -reach))
    local = np.argwhere(np.ones((block + 1,) * 3, bool))            # (b+1)^3 node offsets
    cubes = np.argwhere(np.ones((block,) * 3, bool))                # b^3 cube origins
    lid = lambda ijk: ijk[..., 0] * (block + 1) ** 2 + ijk[..., 1] * (block + 1) + ijk[..., 2]
    tet_local = lid(cubes[:, None, None, :] + _CUBE[_TETS][None])   # (b^3, 6, 4) local node ids
    tet_local = tet_local.reshape(-1, 4)
    E, I, O, G, V = [], [], [], [], []
    for s in range(0, len(active), chunk):
        blk = active[s:s + chunk]                                   # (B, 3) block indices
        nodes = blk[:, None, :] * block + local[None]               # (B, (b+1)^3, 3) global ijk
        pts = lo + h * nodes.reshape(-1, 3)
        vals = f(pts).reshape(len(blk), -1)
        gid = (nodes * stride).sum(-1)                              # (B, (b+1)^3)
        tv = vals[:, tet_local]                                     # (B, T, 4)
        ti = gid[:, tet_local]
        mixed = (tv.min(-1) < 0) & (tv.max(-1) >= 0)
        e, i_, o_ = _tet_triangles(ti[mixed], tv[mixed])
        E.append(e); I.append(i_); O.append(o_)
        if len(e):  # keep the values of the nodes the triangles refer to
            flat_g, flat_v = gid.reshape(-1), vals.reshape(-1)
            sel = np.isin(flat_g, np.unique(e.reshape(-1)))
            G.append(flat_g[sel]); V.append(flat_v[sel])
    if not E or not sum(len(e) for e in E):
        return np.zeros((0, 3)), np.zeros((0, 3), np.int64)
    E, I, O = np.concatenate(E), np.concatenate(I), np.concatenate(O)
    keys = np.sort(E.reshape(-1, 2), axis=1)
    ukeys, inv = np.unique(keys, axis=0, return_inverse=True)
    G, first = np.unique(np.concatenate(G), return_index=True)
    V = np.concatenate(V)[first]
    gval = lambda g: V[np.searchsorted(G, g)]
    gpos = lambda g: lo + h * np.stack([g % n[0], (g // n[0]) % n[1], g // (n[0] * n[1])], -1)
    fa, fb = gval(ukeys[:, 0]), gval(ukeys[:, 1])
    t = np.clip(fa / np.where(fa != fb, fa - fb, 1.0), 0.0, 1.0)[:, None]
    verts = gpos(ukeys[:, 0]) + t * (gpos(ukeys[:, 1]) - gpos(ukeys[:, 0]))
    tris = inv.reshape(-1, 3)
    # orient: normal points from the tet's inside node towards its outside node
    a, b, c = verts[tris[:, 0]], verts[tris[:, 1]], verts[tris[:, 2]]
    nrm = np.cross(b - a, c - a)
    flip = (nrm * (gpos(O) - gpos(I))).sum(1) < 0
    tris[flip] = tris[flip][:, ::-1]
    return verts, tris
