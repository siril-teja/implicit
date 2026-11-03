"""Intervertebral discs, built live from the two vertebrae as shown.

    python -m implicit.anatomy.intervertebral assets/bones/vivaplus-50F   # inputs, rim areas

As built for FE in other tools (nTop): the endplate rims are traced on the bone, the
two rim profiles are joined across the disc space, the disc is extended a little into
both bones (so it covers any endplate concavity) and the bones cut it, so the disc and
each bone share their interface surface (conforming meshes, common nodes).

Here the Disc block (graph.py; sdf.LoftDisc) does this in the shader, every frame,
from the upper and lower vertebrae as shown and their moved landmarks: whatever moves
or reshapes the bones, the disc is rebuilt from them, and its interface is the bone
surface itself in every posture, whatever the motion limits. This module gives the
block's inputs per level (``disc_params``: body half depth, rim ellipse, extension into
each bone, from the landmarks and the bones at rest), rim areas and the annulus fibre
field.

Shape (disc frame: axis w, the bisector of the two outward endplate normals;
anterior a, right r):
  - rims: each vertebral body's outline SECTION mm inside its endplate (the bone's own
    field as shown, sampled on that plane along w), behind the posterior body wall cut off (pedicles),
    within the ellipse of the body's depth and width, 10 % larger, and at most 15 %
    beyond the opposite endplate's extent (from its corners, else its body: the
    sacrum's section there includes its alae; S1 takes L5's body dimensions). No endplate corners are needed (T1's inferior endplate fits
    poorly and has none);
  - wall: the two rims blended by height between the endplate planes (the spine curve
    is straight across one disc; its bulge beyond the rims is not modelled);
  - extension: beyond each endplate plane the profile shrinks by TAPER mm per mm (no
    collar around the waisted vertebral bodies) up to EXT mm into the bone (each
    endplate's deepest concavity below its plane plus 1 mm, measured on the bone more
    than INNER mm inside the rim, where the waisted side walls do not reach; at least
    EXT_MIN mm, as the bones cut it anyway).
Endplate rim area: the area inside the rim at rest (``disc_params``), compared with
population values (lumbar: Louie et al. 2018, women 1,819 +- 502 mm2 over L1-S1).

Annulus fibres (``fibres``, for FE): 20 lamellae between the outer wall and the
nucleus, each with its fibres inclined to the transverse plane, alternately up and
down, from 28 deg at the outer wall to 45 deg at the nucleus (Cassidy et al. 1989;
Marchand & Ahmed 1990: 15-25 lamellae).

References
  Cassidy JJ, Hiltner A, Baer E. Hierarchical structure of the intervertebral disc.
    Connect Tissue Res 1989;23(1):75-88. doi:10.3109/03008208909103905
  Marchand F, Ahmed AM. Investigation of the laminate structure of lumbar disc anulus
    fibrosus. Spine 1990;15(5):402-10. doi:10.1097/00007632-199005000-00011
  Louie PK, Espinoza Orias AA, Fogg LF, et al. Changes in lumbar endplate area and
    concavity associated with disc degeneration. Spine 2018;43(19):E1127-34.
    doi:10.1097/BRS.0000000000002657
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .convert import VERTEBRAE, load_bone

SECTION = 1.5          # mm: each rim is the body's outline this far inside its endplate
FOOTPRINT = 1.1        # ... within the ellipse of the body's depth and width, this much larger
PARTNER = 1.15         # ... and at most this many times the opposite body's
TAPER = 1.0            # mm the profile shrinks per mm beyond an endplate plane
EXT_MARGIN = 1.0       # mm beyond the deepest endplate concavity the disc reaches into the bone
EXT_MIN = 3.0          # mm: at least this far (the bones cut it in the graph: reaching further costs nothing)
INNER = 3.0            # mm: concavity measured this far inside the rim (the side walls rise just inside it)
LUMBAR_AREA = (1819.0, 502.0)  # mm2, mean, SD: women, L1-S1 (Louie et al. 2018)
LAMELLAE = 20                  # annulus lamellae (Marchand & Ahmed 1990: 15-25)
FIBRE_OUTER, FIBRE_INNER = 28.0, 45.0  # deg from the transverse plane (Cassidy et al. 1989)


def _unit(v):
    v = np.asarray(v, float)
    return v / np.linalg.norm(v)


def disc_frame(upper, lower):
    """(centre, axis w, anterior a, right r) of the disc between landmark entries."""
    pu, nu = np.asarray(upper["inf"]["point"], float), _unit(upper["inf"]["normal"])  # nu points down
    pl, nl = np.asarray(lower["sup"]["point"], float), _unit(lower["sup"]["normal"])  # nl points up
    w = _unit(nl - nu)
    c = 0.5 * (pu + pl)
    R = np.asarray(lower["R"], float)
    a = R[:, 1] - R[:, 1].dot(w) * w  # the lower vertebra's anterior axis, normal to w
    a = _unit(a)
    return c, w, a, np.cross(a, w)


def _body(entry, dims):
    """Body centre, anterior and right axes, half depth and half width of a vertebra
    (``dims``: the entry with depth and width; L5's for the sacrum)."""
    R = np.asarray(entry["R"], float)
    bc = np.asarray(entry.get("body_center", entry["sup"]["point"] if "sup" in entry else entry["inf"]["point"]), float)
    return bc, R[:, 1], R[:, 0], 0.5 * dims["depth"], 0.5 * dims["width"]


def _rim(g, p, n, w, body, partner):
    """The body section of grid ``g`` on the plane through p (normal n, SECTION mm inside
    the endplate), as a field of points: sampled at each point's projection along w onto
    that plane; behind the posterior body wall nothing (pedicles), and within the ellipse
    of the body's depth and width, FOOTPRINT times larger, at most PARTNER times the
    opposite endplate's ``partner`` half extents (the sacrum's section reaches its alae)."""
    bc, ax, ar, hd, hw = body
    pd, pw = partner  # the opposite endplate's half depth and width
    ra, rr = min(FOOTPRINT * hd, PARTNER * pd), min(FOOTPRINT * hw, PARTNER * pw)

    def f(X):
        d = (X - p) @ n
        P = X - np.outer(d / (w @ n), w)
        u, v = (P - bc) @ ax, (P - bc) @ ar
        foot = (np.sqrt((u / ra) ** 2 + (v / rr) ** 2) - 1.0) * min(ra, rr)
        return np.maximum.reduce([g.eval(P), -(u + hd), foot])
    return f


def _extent(entry, ep, body):
    """Half depth and half width of an endplate: from its corners when it has them,
    else its body's."""
    c = entry[ep].get("corners")
    if c:
        d = lambda k1, k2: 0.5 * float(np.linalg.norm(np.subtract(c[k1], c[k2])))
        return d("anterior", "posterior"), d("right", "left")
    return body[3], body[4]


def _concavity(g, entry, ep, rim):
    """Deepest point of the endplate surface beyond its plane (into the bone) more than
    INNER mm inside the rim (the waisted side walls rise just inside it), mm."""
    p, n = np.asarray(entry[ep]["point"], float), _unit(entry[ep]["normal"])
    S = g.origin + g.spacing * np.argwhere(np.abs(g.values) < 0.5 * g.spacing)
    h = (S - p) @ (-n)  # + into the bone
    S, h = S[(h > -3.0) & (h < 10.0)], h[(h > -3.0) & (h < 10.0)]
    inner = rim(S) < -INNER
    return float(h[inner].max()) if inner.any() else 0.0


def _rim_points(rim, p, n, a, step=0.5):
    """Sample points inside a rim outline (its field on the plane through p, normal n)."""
    s = np.arange(-45, 45 + 1e-9, step)
    U, V = np.meshgrid(s, s, indexing="ij")
    u = _unit(a - a.dot(n) * n)
    v = np.cross(n, u)
    X = p + U.reshape(-1, 1) * u + V.reshape(-1, 1) * v
    return X[rim(X) < 0]


def rim_area(rim, p, n, a, r, step=0.5):
    """Area (mm2) inside a rim outline (its field on the plane through p, normal n)."""
    return float(len(_rim_points(rim, p, n, a, step)) * step * step)


def rim_ellipse(rim, p, n, body, step=0.5):
    """The rim's ellipse for the nucleus: centre (cu, cv) and semi-axes (ea, er) in the
    body's anterior and right axes (as the Disc block measures them), from the rim's
    centroid and second moments, scaled to the rim's area."""
    bc, ax, ar = body[:3]
    X = _rim_points(rim, p, n, ax, step)
    u, v = (X - bc) @ ax, (X - bc) @ ar
    area = len(X) * step * step
    ea, er = 2.0 * np.sqrt(np.var(u)), 2.0 * np.sqrt(np.var(v))
    k = np.sqrt(area / (np.pi * ea * er))
    return float(u.mean()), float(v.mean()), float(k * ea), float(k * er)


def disc_params(gu, gl, upper, lower, dims_u, dims_l):
    """Inputs of the Disc block (LoftDisc) between two landmark entries at rest, and
    rim areas: per side the body's half depth (hd), the rim ellipse (ra, rr: FOOTPRINT
    times the body, at most PARTNER times the opposite endplate's extent) and the
    extension into the bone (ext: the deepest endplate concavity plus EXT_MARGIN, at
    least EXT_MIN). ``gu``, ``gl``: the two bone grids; ``dims_u``, ``dims_l``: entries
    with each body's depth and width (L5's for the sacrum)."""
    c, w, a, r = disc_frame(upper, lower)
    pu, nu = np.asarray(upper["inf"]["point"], float), _unit(upper["inf"]["normal"])
    pl, nl = np.asarray(lower["sup"]["point"], float), _unit(lower["sup"]["normal"])
    bu, bl = _body(upper, dims_u), _body(lower, dims_l)
    pdu, pwu = _extent(lower, "sup", bl)
    pdl, pwl = _extent(upper, "inf", bu)
    rim_u = _rim(gu, pu - SECTION * nu, nu, w, bu, (pdu, pwu))
    rim_l = _rim(gl, pl - SECTION * nl, nl, w, bl, (pdl, pwl))
    out = {}
    for side, body, (pd, pw), g, e, ep, rim in (("u", bu, (pdu, pwu), gu, upper, "inf", rim_u),
                                                ("l", bl, (pdl, pwl), gl, lower, "sup", rim_l)):
        out[f"hd_{side}"] = round(body[3], 3)
        out[f"ra_{side}"] = round(min(FOOTPRINT * body[3], PARTNER * pd), 3)
        out[f"rr_{side}"] = round(min(FOOTPRINT * body[4], PARTNER * pw), 3)
        out[f"ext_{side}"] = round(max(_concavity(g, e, ep, rim) + EXT_MARGIN, EXT_MIN), 2)
        q, n = np.asarray(e[ep]["point"], float), _unit(e[ep]["normal"])
        for name, x in zip(("cu", "cv", "ea", "er"), rim_ellipse(rim, q - SECTION * n, n, body)):
            out[f"{name}_{side}"] = round(x, 3)
    areas = {"upper": rim_area(rim_u, pu - SECTION * nu, nu, a, r),
             "lower": rim_area(rim_l, pl - SECTION * nl, nl, a, r)}
    return out, areas


def levels(lm):
    """(upper, lower) landmark names of every disc: neighbouring vertebrae with endplates
    (and L5-S1)."""
    order = [lv for lv in VERTEBRAE if lv in lm] + (["S1"] if "S1" in lm else [])
    return [(u, l) for u, l in zip(order, order[1:]) if "inf" in lm[u] and "sup" in lm[l]]


def radial_position(g, centre, axis, P, iters=40):
    """For points P of a disc (grid g, centre, unit axis): the in-plane scale s at which
    each point lies on the disc's outline scaled about the centre (normal to the axis,
    as the Disc block scales the nucleus): 0 at the centre, 1 on the outer wall; the
    nucleus is s < sqrt(fraction). Bisection on s (the outline is star-shaped about the
    centre)."""
    c, w = np.asarray(centre, float), _unit(axis)
    d = np.asarray(P, float) - c
    h = d @ w
    inplane = d - np.outer(h, w)
    lo, hi = np.full(len(d), 1e-3), np.full(len(d), 4.0)
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        inside = g.eval(c + inplane / mid[:, None] + np.outer(h, w)) < 0
        hi = np.where(inside, mid, hi)
        lo = np.where(inside, lo, mid)
    return 0.5 * (lo + hi)


def fibres(g, centre, axis, P, nucleus, lamellae=LAMELLAE, h=0.3):
    """Annulus fibre field at points P (FE material directions): (lamella index, 0 at
    the outer wall, -1 in the nucleus; unit fibre direction). Lamellae are equal steps of
    the radial position between the outer wall (1) and the nucleus (sqrt(nucleus)); the
    fibres lie in the lamella (along the outline's tangent, normal to the axis, and the
    axis), inclined to the transverse plane by FIBRE_OUTER at the wall to FIBRE_INNER at
    the nucleus, the sign alternating between lamellae (Cassidy et al. 1989: 62 to 45 deg
    from the spine's axis, i.e. 28 to 45 deg from the transverse plane; Marchand & Ahmed
    1990: 15-25 lamellae)."""
    c, w = np.asarray(centre, float), _unit(axis)
    P = np.asarray(P, float)
    s = radial_position(g, c, w, P)
    k = np.sqrt(nucleus)
    t = np.clip((1.0 - s) / (1.0 - k), 0.0, 1.0)  # 0 at the outer wall, 1 at the nucleus
    lam = np.where(s < k, -1, np.minimum((t * lamellae).astype(int), lamellae - 1))
    # Outline normal at the corresponding point of the outer wall (in plane), its tangent.
    d = P - c
    hh = d @ w
    B = c + (d - np.outer(hh, w)) / s[:, None] + np.outer(hh, w)
    grad = np.stack([(g.eval(B + h * e) - g.eval(B - h * e)) / (2 * h) for e in np.eye(3)], 1)
    grad -= np.outer(grad @ w, w)
    grad /= np.linalg.norm(grad, axis=1, keepdims=True)
    tangent = np.cross(w, grad)
    alpha = np.radians(FIBRE_OUTER + (FIBRE_INNER - FIBRE_OUTER) * t)
    sign = np.where(lam % 2 == 0, 1.0, -1.0)
    f = np.cos(alpha)[:, None] * tangent + (sign * np.sin(alpha))[:, None] * w
    return lam, f


def report(bone_dir, log=print):
    """Disc inputs and rim areas of every level of ``bone_dir`` (what the graph builder
    passes to the Disc blocks); returns {label: (params, areas)}."""
    bone_dir = Path(bone_dir)
    lm = json.loads((bone_dir / "landmarks.json").read_text())["landmarks"]
    out = {}
    for up, lo in levels(lm):
        lo_bone = "sacrum" if lo == "S1" else lo
        gu, gl = load_bone(bone_dir / f"{up}.npz")[0], load_bone(bone_dir / f"{lo_bone}.npz")[0]
        prm, areas = disc_params(gu, gl, lm[up], lm[lo], lm[up], lm["L5"] if lo == "S1" else lm[lo])
        out[f"{up}-{lo}"] = (prm, areas)
        log(f"  disc {up}-{lo}: rims {areas['upper']:.0f} / {areas['lower']:.0f} mm2, into the bones "
            f"{prm['ext_u']:.1f} / {prm['ext_l']:.1f} mm")
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("bone_dir")
    report(ap.parse_args(argv).bone_dir)


if __name__ == "__main__":
    main()
