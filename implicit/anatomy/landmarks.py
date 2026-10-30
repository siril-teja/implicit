"""Vertebral landmarks from implicit bones: body frames and endplate planes.

    python -m implicit.anatomy.landmarks assets/bones/s0751

Reads the per-bone files written by ``implicit.anatomy.convert`` (never the
masks) and writes ``landmarks.json`` next to them, in the same scanner frame.

Per vertebra (C2 to L5, plus the S1 superior endplate)
  1. isolate the vertebral body: drop points behind the posterior wall (found
     by marching anteriorly from the spinal-cord centroid, when the cord file
     exists), morphological opening (removes pedicles and processes), largest
     remaining component;
  2. local frame: axial axis w, anterior a (scanner +y made orthogonal to w),
     lateral l = a x w. w starts as the spine tangent through the neighbouring
     vertebrae and is then refined to the bisector of the two endplate normals
     (step 3) until it settles, so a wedge-shaped body (L5) or one with large
     processes does not tilt the frame;
  3. endplates: march from the body centre along +-w through central columns to
     the bone surface (sub-voxel zero crossing), robust least-squares plane.
     The frame origin is the midpoint of the two endplate centres.
  S1 is found by marching from L5's inferior endplate down through the disc
  space; C2's inferior endplate from C3.

Requires scipy.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .convert import VERTEBRAE, load_bones

OPEN_RADIUS = {"C": 4.5, "T": 4.5, "L": 5.5, "S": 5.5}  # mm, body isolation


def _ndi():
    from scipy import ndimage
    return ndimage


class Field:
    """Trilinear sampler for a world-aligned grid (preprocessing helper)."""

    def __init__(self, values, origin, h):
        self.v, self.o, self.h = values, np.asarray(origin, float), float(h)

    def __call__(self, pts):
        ndi = _ndi()
        q = (np.asarray(pts) - self.o) / self.h
        return ndi.map_coordinates(self.v, q.T, order=1, mode="nearest")

    def inside_points(self):
        return self.o + self.h * np.argwhere(self.v < 0)


def posterior_wall(f: Field, canal, tangent, reach=45.0, step=0.25):
    """Midline posterior wall: march anteriorly from the canal centre to the bone."""
    a = np.array([0.0, 1.0, 0.0]) - tangent[1] * tangent
    a /= np.linalg.norm(a)
    ts = np.arange(0, reach, step)
    vals = f(canal + ts[:, None] * a)
    hit = np.flatnonzero(vals < 0)
    if not len(hit):
        return None, a
    return canal + ts[hit[0]] * a, a


def isolate_body(f: Field, region, canal=None, tangent=None):
    """Points of the vertebral body.

    Morphological opening removes thin processes; when the canal is known, all
    points behind the midline posterior wall are dropped (pedicles, arch), then
    the largest remaining component is the body.
    """
    ndi = _ndi()
    inside = f.v < 0
    if canal is not None and tangent is not None:
        wall, a = posterior_wall(f, canal, tangent)
        if wall is not None:
            grid_pts = f.o + f.h * np.indices(f.v.shape).reshape(3, -1).T
            inside = inside & ((grid_pts - wall) @ a > -1.0).reshape(f.v.shape)
    r = OPEN_RADIUS[region]
    for _ in range(4):
        core = (f.v < -r) & inside
        if core.any():
            opened = (ndi.distance_transform_edt(~core) * f.h <= r) & inside
            lab, n = ndi.label(opened)
            best, best_size = None, 0
            for i in range(1, n + 1):
                comp = lab == i
                size = int(comp.sum())
                pts_c = f.o + f.h * np.argwhere(comp).mean(0)
                if canal is not None and pts_c[1] < canal[1]:
                    continue  # posterior to the cord: not the body
                if size > best_size:
                    best, best_size = comp, size
            if best is not None and best_size * f.h ** 3 > 300:
                return f.o + f.h * np.argwhere(best)
        r *= 0.75
    raise RuntimeError("could not isolate the vertebral body")


def frame_from_axis(w):
    """Columns (l, a, w): lateral (right), anterior, axial (cranial) for axial axis w."""
    w = w / np.linalg.norm(w)
    a = np.array([0.0, 1.0, 0.0])
    a = a - a.dot(w) * w
    a /= np.linalg.norm(a)
    return np.stack([np.cross(a, w), a, w], axis=1)


def central_columns(local_uv, frac=0.3, h=1.0):
    """(u, v) column centres in the central part of a 2-D footprint."""
    ndi = _ndi()
    lo = np.floor(local_uv.min(0)) - 2
    ij = np.round((local_uv - lo) / h).astype(int)
    fp = np.zeros(ij.max(0) + 3, dtype=bool)
    fp[ij[:, 0], ij[:, 1]] = True
    fp = ndi.binary_closing(fp, iterations=2)
    fp = ndi.binary_fill_holes(fp)
    d = ndi.distance_transform_edt(fp) * h
    sel = np.argwhere(d >= frac * d.max())
    return lo + h * sel, fp, lo


def march(f: Field, c, R, uv, direction, w0, w_max, step=0.25, enter=False):
    """First crossing along w for each (u, v) column.

    ``enter=False``: start inside, find the exit (value goes >= 0).
    ``enter=True``: start outside, find the entry (value goes < 0).
    Returns local (u, v, w) points of the crossings found.
    """
    ws = w0 + direction * np.arange(0, w_max, step)
    U = np.repeat(uv[:, :1], len(ws), 1)
    V = np.repeat(uv[:, 1:], len(ws), 1)
    W = np.broadcast_to(ws, U.shape)
    local = np.stack([U, V, W], -1).reshape(-1, 3)
    vals = f(c + local @ R.T).reshape(U.shape)
    hit = vals < 0 if enter else vals >= 0
    start_ok = (vals[:, 0] >= 0) if enter else (vals[:, 0] < 0)
    k = np.argmax(hit, axis=1)
    ok = hit.any(1) & start_ok & (k > 0)
    k, vals, uv = k[ok], vals[ok], uv[ok]
    v0, v1 = vals[np.arange(len(k)), k - 1], vals[np.arange(len(k)), k]
    t = v0 / (v0 - v1)
    w = w0 + direction * step * (k - 1 + t)
    return np.column_stack([uv, w])


def fit_plane(local, iters=4):
    """Robust fit w = a u + b v + c; returns (a, b, c, inlier mask)."""
    keep = np.ones(len(local), bool)
    for _ in range(iters):
        A = np.column_stack([local[keep, 0], local[keep, 1], np.ones(keep.sum())])
        coef, *_ = np.linalg.lstsq(A, local[keep, 2], rcond=None)
        res = local[:, 2] - (coef[0] * local[:, 0] + coef[1] * local[:, 1] + coef[2])
        mad = np.median(np.abs(res[keep] - np.median(res[keep]))) + 1e-6
        keep = np.abs(res) < 3.0 * 1.4826 * mad + 0.25
    return coef, keep


def plane_world(c, R, coef, uv0, sign):
    a, b, k = coef
    w0 = a * uv0[0] + b * uv0[1] + k
    point = c + R @ np.array([uv0[0], uv0[1], w0])
    n = R @ np.array([-a, -b, 1.0])
    n = sign * n / np.linalg.norm(n)
    return point, n


def _endplates(f: Field, level, body, c, R):
    local = (body - c) @ R
    uv, fp, lo = central_columns(local[:, :2], frac=0.3, h=f.h)
    uv0 = local[:, :2].mean(0)
    height = np.ptp(local[:, 2])
    out = {"level": level, "R": R, "height": float(height),
           "width": float(np.ptp(local[:, 0])), "depth": float(np.ptp(local[:, 1])),
           "footprint_uv": uv}
    for name, sign in (("sup", 1.0), ("inf", -1.0)):
        pts = march(f, c, R, uv, sign, 0.0, 0.9 * height + 5.0)
        if len(pts) < 10:
            raise RuntimeError(f"{level}: endplate {name} not found")
        coef, keep = fit_plane(pts)
        p, n = plane_world(c, R, coef, uv0, sign)
        out[name] = {"point": p, "normal": n, "n_pts": int(keep.sum()),
                     "rms": float(np.sqrt(np.mean((pts[keep, 2] - (pts[keep, :2] @ coef[:2] + coef[2])) ** 2)))}
    return out


def vertebra_landmarks(f: Field, level, tangent, canal, iters=6):
    """Body frame and endplates; the axial axis converges to the endplate-normal bisector."""
    body = isolate_body(f, level[0], canal, tangent)
    c, w = body.mean(0), tangent / np.linalg.norm(tangent)
    for _ in range(iters):
        out = _endplates(f, level, body, c, frame_from_axis(w))
        bis = out["sup"]["normal"] - out["inf"]["normal"]
        bis /= np.linalg.norm(bis)
        settled = np.degrees(np.arccos(np.clip(bis @ w, -1, 1))) < 0.5
        w = bis
        if settled:
            break
    out = _endplates(f, level, body, c, frame_from_axis(w))
    out["R"] = frame_from_axis(w)
    out["body_center"] = 0.5 * (out["sup"]["point"] + out["inf"]["point"])
    out["body_pts"] = body
    return out


def opposing_endplate(f_target: Field, ref, ref_ep, sign):
    """Fit the endplate of a neighbour across the disc from a reliable reference.

    March from ``ref``'s ``ref_ep`` endplate (``"inf"`` going caudally, ``"sup"``
    going cranially) through the disc until entering ``f_target``, over the
    reference body's central columns. Used for S1 (from L5) and C2 (from C3),
    whose own bodies are hard to isolate (sacral alae, dens/lateral masses).
    Returns (point, normal) with the normal pointing back towards ``ref``.
    """
    c, R, uv = ref["body_center"], ref["R"], ref["footprint_uv"]
    direction = -1.0 if ref_ep == "inf" else 1.0
    w_start = float((ref[ref_ep]["point"] - c) @ R[:, 2]) + direction * 0.5
    pts = march(f_target, c, R, uv, direction, w_start, 25.0, enter=True)
    if len(pts) < 10:
        raise RuntimeError("opposing endplate not found")
    coef, keep = fit_plane(pts)
    p, n = plane_world(c, R, coef, uv.mean(0), -direction)
    return {"point": p, "normal": n, "n_pts": int(keep.sum()),
            "rms": float(np.sqrt(np.mean((pts[keep, 2] - (pts[keep, :2] @ coef[:2] + coef[2])) ** 2)))}


def s1_landmarks(f_sacrum: Field, l5):
    """S1 superior endplate, found from L5 (footprint and frame borrowed from L5)."""
    sup = opposing_endplate(f_sacrum, l5, "inf", -1)
    p, n = sup["point"], sup["normal"]
    a = l5["R"][:, 1] - l5["R"][:, 1].dot(n) * n
    a /= np.linalg.norm(a)
    R1 = np.stack([np.cross(a, n), a, n], axis=1)
    return {"level": "S1", "body_center": p - n * 10.0, "R": R1, "height": 0.0, "sup": sup}


# ---------------------------------------------------------------- endplate shape (rim + surface)

RIM_SLOPE = 35.0   # deg: steeper than this (after smoothing) is the vertebral wall, not the endplate
RIM_SMOOTH = 1.0   # mm: height-map smoothing before the slope test (CT voxel steps)
RIM_HARMONICS = 8  # Fourier harmonics of the rim outline
POLY_DEG = 4       # endplate surface polynomial degree
POLY_SCALE = 20.0  # mm: (u, v) are divided by this in the polynomial (conditioning)


def _plane_axes(n, R):
    """Orthonormal (ex, ey) in the plane with normal n, close to the frame's right/anterior axes."""
    ey = R[:, 1] - (R[:, 1] @ n) * n
    ey /= np.linalg.norm(ey)
    return np.cross(ey, n) * (1 if np.cross(ey, n) @ R[:, 0] > 0 else -1), ey


def height_map(f: Field, p, n, R, ext=40.0, h=0.5):
    """Bone surface height along n over a grid in the endplate plane (NaN where no bone)."""
    ex, ey = _plane_axes(n, R)
    us = np.arange(-ext, ext + 1e-9, h)
    U, V = np.meshgrid(us, us, indexing="ij")
    ws = np.arange(15.0, -25.0, -0.25)
    pts = p + U[..., None, None] * ex + V[..., None, None] * ey + ws[None, None, :, None] * n
    vals = f(pts.reshape(-1, 3)).reshape(U.shape + ws.shape)
    inside = vals < 0
    hit = inside.any(-1) & (vals[..., 0] >= 0)
    k = np.clip(np.argmax(inside, -1), 1, len(ws) - 1)
    v0 = np.take_along_axis(vals, (k - 1)[..., None], -1)[..., 0]
    v1 = np.take_along_axis(vals, k[..., None], -1)[..., 0]
    H = ws[k - 1] - 0.25 * v0 / (v0 - v1 + 1e-12)
    H[~hit] = np.nan
    return U, V, H, (ex, ey)


def _footprint(points, p, ex, ey, U, h, grow):
    """Grid mask of points projected onto the plane (filled, grown by ``grow`` mm)."""
    ndi = _ndi()
    m = np.zeros(U.shape, bool)
    q = points - p
    i = np.round((q @ ex - U[0, 0]) / h).astype(int)
    j = np.round((q @ ey - U[0, 0]) / h).astype(int)
    ok = (i >= 0) & (j >= 0) & (i < U.shape[0]) & (j < U.shape[1])
    m[i[ok], j[ok]] = True
    m = ndi.binary_fill_holes(ndi.binary_closing(m, iterations=3))
    return ndi.binary_dilation(m, iterations=max(1, int(round(grow / h))))


def endplate_shape(f: Field, p, n, R, origin, guide_pts=None, guide_grow=2.0, h=0.5):
    """Endplate region and rim from the bone surface, in the vertebra frame.

    The endplate is where the (smoothed) surface seen along the endplate normal is
    flatter than RIM_SLOPE and within 8 mm of the endplate plane, connected to its
    centre; the rim is its outline, where the surface turns into the vertebral wall.
    ``guide_pts`` (e.g. the isolated vertebral body) limits it to their footprint so
    pedicles, transverse processes or the sacral ala at the same level stay out.

    Returns the rim outline (centre and Fourier coefficients of the radius) and a
    polynomial of the surface height w(u, v), both in the frame (origin, R), plus
    the rim points (frame coordinates) and quality numbers.
    """
    ndi = _ndi()
    U, V, H, (ex, ey) = height_map(f, p, n, R, h=h)
    hit = np.isfinite(H)
    hv = np.where(hit, H, 0.0)
    wgt = ndi.gaussian_filter(hit.astype(float), RIM_SMOOTH / h)
    sm = ndi.gaussian_filter(hv, RIM_SMOOTH / h) / np.maximum(wgt, 1e-6)
    sm[~hit] = -30.0
    gu, gv = np.gradient(sm, h)
    ok = hit & (np.degrees(np.arctan(np.hypot(gu, gv))) < RIM_SLOPE) & (np.abs(np.nan_to_num(H, nan=99)) < 8)
    if guide_pts is not None:
        ok &= _footprint(guide_pts, p, ex, ey, U, h, guide_grow)
    ok = ndi.binary_opening(ok, iterations=int(round(2.0 / h)))
    lab, nl = ndi.label(ok)
    if nl == 0:
        raise RuntimeError("endplate region not found")
    c = lab[U.shape[0] // 2, U.shape[1] // 2]
    if c == 0:  # the centre fell outside: take the component nearest to it
        d = np.where(lab > 0, np.hypot(U, V), np.inf)
        c = lab[np.unravel_index(np.argmin(d), U.shape)]
    mask = ndi.binary_fill_holes(lab == c)
    # Surface and rim points to the vertebra frame.
    world = lambda m: p + U[m][:, None] * ex + V[m][:, None] * ey + H[m][:, None] * n
    inner = mask & hit
    surf = (world(inner) - origin) @ R
    rim_px = mask & ~ndi.binary_erosion(mask)
    rim = (world(rim_px & hit) - origin) @ R
    cu, cv = surf[:, 0].mean(), surf[:, 1].mean()
    # Rim outline: outermost rim radius per angle bin, then a Fourier fit r(theta).
    th = np.arctan2(rim[:, 1] - cv, rim[:, 0] - cu)
    r = np.hypot(rim[:, 0] - cu, rim[:, 1] - cv)
    bins = np.linspace(-np.pi, np.pi, 181)
    idx = np.clip(np.digitize(th, bins) - 1, 0, 179)
    rb = np.full(180, np.nan)
    for b_ in range(180):
        sel = r[idx == b_]
        if len(sel):
            rb[b_] = sel.max()
    tb = 0.5 * (bins[:-1] + bins[1:])
    good = np.isfinite(rb)
    rb = np.interp(tb, tb[good], rb[good], period=2 * np.pi)
    A = np.column_stack([np.ones_like(tb)] + [g(k * tb) for k in range(1, RIM_HARMONICS + 1) for g in (np.cos, np.sin)])
    rc, *_ = np.linalg.lstsq(A, rb, rcond=None)
    # Surface polynomial w(u, v) by trimmed least squares.
    s = (surf[:, :2] - [cu, cv]) / POLY_SCALE
    terms = [(i, j) for i in range(POLY_DEG + 1) for j in range(POLY_DEG + 1 - i)]
    P = np.column_stack([s[:, 0] ** i * s[:, 1] ** j for i, j in terms])
    keep = np.ones(len(s), bool)
    for _ in range(3):
        wc, *_ = np.linalg.lstsq(P[keep], surf[keep, 2], rcond=None)
        res = surf[:, 2] - P @ wc
        keep = np.abs(res) < 3 * np.std(res[keep]) + 0.2
    return {"center": [float(cu), float(cv)], "rim": rc.tolist(), "poly": wc.tolist(),
            "poly_terms": terms, "area_mm2": float(inner.sum() * h * h),
            "rim_fit_rms": float(np.sqrt(np.mean((A @ rc - rb) ** 2))),
            "surface_rms": float(np.sqrt(np.mean(res[keep] ** 2))),
            "rim_points": rim[:, :2]}


def detect(bone_dir, log=print):
    """Landmarks for every vertebra present in ``bone_dir``; writes landmarks.json."""
    bone_dir = Path(bone_dir)
    bones = load_bones(bone_dir)
    fields = {n: Field(g.values, g.origin, g.spacing) for n, (g, _) in bones.items()}
    levels = [lv for lv in VERTEBRAE if lv in fields]
    centroids = {lv: fields[lv].inside_points().mean(0) for lv in levels}
    cord_pts = fields["spinal_cord"].inside_points() if "spinal_cord" in fields else None
    marks = {}
    for i, lv in enumerate(levels):
        if lv == "C1":
            continue
        # Initial axis through the neighbouring vertebrae (one-sided at the ends;
        # never the sacrum or skull, whose centroids are far off the column).
        up = centroids[levels[i - 1]] if i > 0 else centroids[lv]
        dn = centroids[levels[i + 1]] if i + 1 < len(levels) else centroids[lv]
        tangent = (up - dn) / np.linalg.norm(up - dn)
        canal = None
        if cord_pts is not None:
            near = cord_pts[np.abs(cord_pts[:, 2] - centroids[lv][2]) < 6]
            if len(near) > 20:
                canal = near.mean(0)
        try:
            marks[lv] = vertebra_landmarks(fields[lv], lv, tangent, canal)
            e = marks[lv]
            log(f"  {lv}: body {e['width']:.0f}x{e['depth']:.0f}x{e['height']:.0f} mm, "
                f"endplate rms {e['sup']['rms']:.2f}/{e['inf']['rms']:.2f} mm")
        except RuntimeError as err:
            log(f"  {lv}: {err}")
    if "L5" in marks and "sacrum" in fields:
        marks["S1"] = s1_landmarks(fields["sacrum"], marks["L5"])
        log(f"  S1: superior endplate rms {marks['S1']['sup']['rms']:.2f} mm (from L5)")
    if "C2" in marks and "C3" in marks:
        marks["C2"]["inf"] = opposing_endplate(fields["C2"], marks["C3"], "sup", 1)
        marks["C2"]["inf_from"] = "C3"
        log(f"  C2: inferior endplate rms {marks['C2']['inf']['rms']:.2f} mm (from C3)")
    # Endplate shapes (rim outline and surface) for the disc blocks.
    for lv, e in marks.items():
        f = fields["sacrum" if lv == "S1" else lv]
        for ep in ("sup", "inf"):
            if ep not in e:
                continue
            if lv == "S1":  # no isolated body: stay within L5's lower rim (+5 mm)
                l5 = marks.get("L5", {}).get("inf_shape")
                guide = None
                if l5 is not None:
                    rp = l5["rim_points"]
                    guide = marks["L5"]["body_center"] + np.column_stack([rp, np.zeros(len(rp))]) @ marks["L5"]["R"].T
                grow = 5.0
            else:
                guide, grow = e.get("body_pts"), 2.0
            try:
                e[f"{ep}_shape"] = endplate_shape(f, e[ep]["point"], e[ep]["normal"], e["R"], e["body_center"],
                                                  guide, grow)
                sh = e[f"{ep}_shape"]
                log(f"  {lv} {ep} endplate: {sh['area_mm2'] / 100:.1f} cm2, rim fit {sh['rim_fit_rms']:.2f} mm, "
                    f"surface fit {sh['surface_rms']:.2f} mm")
            except RuntimeError as err:
                log(f"  {lv} {ep} endplate shape: {err}")
    drop = {"footprint_uv", "body_pts"}
    out = {lv: _jsonable({k: ({kk: vv for kk, vv in v.items() if kk != "rim_points"} if k.endswith("_shape") else v)
                          for k, v in e.items() if k not in drop})
           for lv, e in marks.items()}
    (bone_dir / "landmarks.json").write_text(json.dumps(
        {"frame": "scanner RAS, mm", "landmarks": out}, indent=1))
    log(f"wrote {bone_dir / 'landmarks.json'}")
    return marks


def _jsonable(x):
    if isinstance(x, dict):
        return {k: _jsonable(v) for k, v in x.items()}
    if isinstance(x, np.ndarray):
        return np.round(x, 4).tolist()
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    return x


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("bone_dir", help="folder of <bone>.npz files from implicit.anatomy.convert")
    detect(ap.parse_args(argv).bone_dir)


if __name__ == "__main__":
    main()
