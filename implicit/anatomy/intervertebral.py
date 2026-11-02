"""Intervertebral discs of a template, built from the posed endplates of its vertebrae.

    python -m implicit.anatomy.intervertebral assets/bones/vivaplus-50F

Each disc is an implicit body between the inferior endplate of the upper vertebra U
and the superior endplate of the lower one L (the sacrum for L5-S1), at rest (the
template's posture), written as a bone file ``disc_<U>-<L>.npz`` (kind "disc") next to
the vertebrae; the Disc block splits it into nucleus and annulus and the polyrigid
field carries it with the posture, exactly rigid at both endplates.

Shape at rest, in the disc frame (axis w: the bisector of the two outward endplate
normals, centre c: midway between the endplate centres):
  - rims: each vertebra's body section SECTION mm inside its endplate (its own field
    sampled on that plane: the rim's outline), behind the posterior corner cut off
    (pedicles) and kept within the ellipse through the endplate's four corners, 10 %
    larger, its semi-axes at most 15 % beyond the opposite endplate's (the sacrum's
    section there includes its alae, and its detected left and right corners reach
    them: 80 mm apart on VIVA+ 50F against 38 mm for L5's inferior endplate);
  - wall: the two rims joined by a straight loft along w (the disc spans rim to rim;
    its bulge beyond the rims is not modelled);
  - endplates: the disc ends at both bones (max with -f_U, -f_L), so it fills the
    endplate concavities and shares their surfaces;
  - height: limited to the disc space: beyond each endplate plane only inside the rim,
    from 0 at the rim to at most CAP mm (1 mm per 3 mm inward: endplate concavities),
    so the disc never wraps around the waisted vertebral bodies.
Annulus fibres (``fibres``, for FE): 20 lamellae between the outer wall and the
nucleus, each with its fibres inclined to the transverse plane, alternately up and
down, from 28 deg at the outer wall to 45 deg at the nucleus (Cassidy et al. 1989;
Marchand & Ahmed 1990: 15-25 lamellae).

Endplate rim area: the area inside the rim outline on each endplate plane
(``rim_area``), compared with population values (lumbar: Louie et al. 2018, women
1,819 +- 502 mm2 over L1-S1).

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

from .convert import VERTEBRAE, load_bone, save_bone

SECTION = 1.5   # mm: rim outline sampled this far inside each endplate
FOOTPRINT = 1.1  # the rim stays within the ellipse through the endplate corners, this much larger
PARTNER = 1.15   # ... and within this much of the opposite endplate's semi-axes
CAP = 4.0       # mm: the disc extends at most this far beyond each endplate plane
CAP_SLOPE = 1.0 / 3.0  # ... starting from 0 at the rim: this much per mm inward
SPACING = 0.6   # mm: disc grid spacing (as the vertebrae)
MARGIN = 3.0    # mm: grid padding around the disc
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
    ant = np.asarray(lower["sup"]["corners"]["anterior"], float) - np.asarray(lower["sup"]["corners"]["posterior"], float)
    a = _unit(ant - ant.dot(w) * w)
    return c, w, a, np.cross(a, w)


def _footprint(corners, a, partner=None):
    """Field of the ellipse through an endplate's four corners, FOOTPRINT times larger
    (antero-posterior and left-right semi-axes; a: anterior axis), of points in or near
    its plane (an approximate distance, exact on the axes)."""
    c = {k: np.asarray(v, float) for k, v in corners.items()}
    ap, lr = c["anterior"] - c["posterior"], c["right"] - c["left"]
    ra, rr = FOOTPRINT * 0.5 * np.linalg.norm(ap), FOOTPRINT * 0.5 * np.linalg.norm(lr)
    if partner is not None:  # no wider than the opposite endplate allows (S1's corners reach the alae)
        q = {k: np.asarray(v, float) for k, v in partner.items()}
        ra = min(ra, PARTNER * 0.5 * np.linalg.norm(q["anterior"] - q["posterior"]))
        rr = min(rr, PARTNER * 0.5 * np.linalg.norm(q["right"] - q["left"]))
    ca, cr = 0.5 * (c["anterior"] + c["posterior"]), 0.5 * (c["right"] + c["left"])
    ua, ur = _unit(ap), _unit(lr)

    def f(X):
        u, v = (X - ca) @ ua / ra, (X - cr) @ ur / rr
        return (np.sqrt(u * u + v * v) - 1.0) * min(ra, rr)
    return f


def _rim(g, p, n, w, corners, a, partner):
    """The body section of grid ``g`` on the plane through p with normal n (SECTION mm
    inside the endplate), as a field of points x: sampled at x's projection along w
    onto that plane, within the endplate's footprint (no sacral alae, no pedicles:
    nothing behind the posterior corner)."""
    post, foot = np.asarray(corners["posterior"], float), _footprint(corners, a, partner)

    def f(X):
        d = (X - p) @ n
        P = X - np.outer(d / (w @ n), w)  # along w onto the plane
        return np.maximum.reduce([g.eval(P), -((P - post) @ a), foot(P)])
    return f


def disc_field(gu, gl, upper, lower):
    """f(X) of the disc at rest between grids ``gu`` (upper vertebra), ``gl`` (lower)."""
    c, w, a, _ = disc_frame(upper, lower)
    pu, nu = np.asarray(upper["inf"]["point"], float), _unit(upper["inf"]["normal"])
    pl, nl = np.asarray(lower["sup"]["point"], float), _unit(lower["sup"]["normal"])
    cu, cl = upper["inf"]["corners"], lower["sup"]["corners"]
    rim_u = _rim(gu, pu - SECTION * nu, nu, w, cu, a, cl)
    rim_l = _rim(gl, pl - SECTION * nl, nl, w, cl, a, cu)
    hu, hl = (pu - c) @ w, (pl - c) @ w  # endplate heights along w (hu > 0 > hl)

    def f(X):
        s = (X - c) @ w
        t = np.clip((s - hl) / (hu - hl), 0.0, 1.0)
        wall = (1 - t) * rim_l(X) + t * rim_u(X)
        # Beyond the endplate planes only inside the rim (the concavities): none at the
        # rim, CAP_SLOPE mm more per mm inward, at most CAP (no collar around the bodies).
        allow = np.clip(-wall * CAP_SLOPE, 0.0, CAP)
        cap = np.maximum(s - hu - allow, hl - allow - s)
        return np.maximum.reduce([wall, cap, -gu.eval(X), -gl.eval(X)])
    return f


def rim_area(g, entry, ep, w, a, partner=None, step=0.5):
    """Area (mm2) inside the rim outline of endplate ``ep`` of a vertebra (its section
    SECTION mm inside the endplate, posterior elements cut off)."""
    p, n = np.asarray(entry[ep]["point"], float), _unit(entry[ep]["normal"])
    q = p - SECTION * n
    u = _unit(np.cross(n, a))
    v = np.cross(u, n)
    s = np.arange(-40, 40 + 1e-9, step)
    U, V = np.meshgrid(s, s, indexing="ij")
    X = q + U.reshape(-1, 1) * u + V.reshape(-1, 1) * v
    inside = ((g.eval(X) < 0) & ((X - np.asarray(entry[ep]["corners"]["posterior"], float)) @ a >= 0)
              & (_footprint(entry[ep]["corners"], a, partner)(X) < 0))
    return float(inside.sum() * step * step)


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


def build(bone_dir, log=print):
    """Write disc_<U>-<L>.npz for every pair of neighbouring vertebrae (and L5-S1) of
    ``bone_dir`` with endplate corners; returns {label: info}."""
    bone_dir = Path(bone_dir)
    lm = json.loads((bone_dir / "landmarks.json").read_text())["landmarks"]
    order = [lv for lv in VERTEBRAE if lv in lm] + (["S1"] if "S1" in lm else [])
    out = {}
    for up, lo in zip(order, order[1:]):
        u, l = lm[up], lm[lo]
        if "corners" not in u.get("inf", {}) or "corners" not in l.get("sup", {}):
            log(f"  {up}-{lo}: no endplate corners, no disc")
            continue
        lo_bone = "sacrum" if lo == "S1" else lo
        gu, gl = load_bone(bone_dir / f"{up}.npz")[0], load_bone(bone_dir / f"{lo_bone}.npz")[0]
        f = disc_field(gu, gl, u, l)
        c, w, a, r = disc_frame(u, l)
        # Grid: around the rim outlines (both endplates' corners) and the disc space.
        P = np.array([u["inf"]["corners"][k] for k in ("anterior", "posterior", "left", "right")]
                     + [l["sup"]["corners"][k] for k in ("anterior", "posterior", "left", "right")], float)
        lo_c, hi_c = P.min(0) - 25.0, P.max(0) + 25.0
        n = np.ceil((hi_c - lo_c) / SPACING).astype(int) + 1
        axes = [lo_c[i] + SPACING * np.arange(n[i]) for i in range(3)]
        X = np.stack(np.meshgrid(*axes, indexing="ij"), -1).reshape(-1, 3)
        vals = f(X).reshape(n)
        idx = np.argwhere(vals < 0)
        if not len(idx):
            log(f"  {up}-{lo}: empty")
            continue
        i0 = np.maximum(idx.min(0) - int(MARGIN / SPACING), 0)
        i1 = np.minimum(idx.max(0) + int(MARGIN / SPACING) + 1, n)
        vals = vals[i0[0]:i1[0], i0[1]:i1[1], i0[2]:i1[2]]
        origin = lo_c + SPACING * i0
        label = f"{up}-{lo}"
        areas = {"inferior_" + up: rim_area(gu, u, "inf", w, a, l["sup"]["corners"]),
                 "superior_" + lo: rim_area(gl, l, "sup", w, a, u["inf"]["corners"])}
        meta = {"name": f"disc_{label}", "kind": "disc", "subject": bone_dir.name,
                "source": f"implicit.anatomy.intervertebral: between {up} (inferior endplate) and {lo_bone} "
                          f"(superior endplate)", "upper": up, "lower": lo_bone,
                "centre": c.round(3).tolist(), "axis": w.round(6).tolist(), "anterior": a.round(6).tolist(),
                "rim_area_mm2": {k: round(v, 1) for k, v in areas.items()}}
        attrib = json.loads(json.dumps(load_bone(bone_dir / f"{up}.npz")[1].get("attribution", "")))
        if attrib:
            meta["attribution"], meta["license"] = attrib, load_bone(bone_dir / f"{up}.npz")[1].get("license", "")
        meta = save_bone(bone_dir / f"disc_{label}.npz", vals, origin, SPACING, meta)
        out[label] = meta
        log(f"  disc {label}: {meta['volume_mm3'] / 1000:.1f} cm3, rim areas "
            + ", ".join(f"{k} {v:.0f} mm2" for k, v in areas.items()))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("bone_dir")
    build(ap.parse_args(argv).bone_dir)


if __name__ == "__main__":
    main()
