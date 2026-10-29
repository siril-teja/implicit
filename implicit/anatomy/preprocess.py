"""Bone masks -> signed distance grids + vertebral landmarks (frames, endplates).

    python -m implicit.anatomy.preprocess data/totalseg/s0011 --out data/models/s0011

Steps per bone
  1. keep the largest connected component of the mask;
  2. signed distance via two Euclidean distance transforms (half-voxel corrected),
     light Gaussian smoothing to remove the voxel staircase;
  3. resample onto a world-axis-aligned grid (RAS: x right, y anterior, z cranial).

Steps per vertebra (C2 to L5, plus the S1 superior endplate)
  1. isolate the vertebral body by morphological opening (removes pedicles and
     processes) and keep the largest component anterior to the spinal cord;
  2. local frame: axial axis w from the body's smallest principal axis (or the
     spine tangent when that disagrees), anterior a, lateral l = a x w;
  3. endplates: march from the body centre along +-w through central columns to
     the bone surface (sub-voxel zero crossing), robust least-squares plane.
  S1 is found by marching from L5's inferior endplate down through the disc.

Requires scipy and nibabel (preprocessing only; scenes need just NumPy).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .totalseg import ATTRIBUTION, LEVELS

OPEN_RADIUS = {"C": 4.5, "T": 4.5, "L": 5.5, "S": 5.5}  # mm, body isolation
FEMUR_KEEP = 110.0  # mm of proximal femur kept


def _ndi():
    from scipy import ndimage
    return ndimage


def load_mask(path):
    import nibabel as nib
    im = nib.as_closest_canonical(nib.load(str(path)))
    return np.asarray(im.dataobj) > 0, im.affine


def largest_component(m):
    ndi = _ndi()
    lab, n = ndi.label(m)
    if n <= 1:
        return m
    sizes = ndi.sum(m, lab, range(1, n + 1))
    return lab == (1 + int(np.argmax(sizes)))


def mask_to_grid(mask, affine, h, pad=6.0, sigma=0.6):
    """Signed distance (mm, negative inside) on a world-aligned grid of spacing h.

    Returns (values[nx, ny, nz], origin[3]).
    """
    ndi = _ndi()
    zooms = np.sqrt((affine[:3, :3] ** 2).sum(0))
    idx = np.argwhere(mask)
    lo, hi = idx.min(0), idx.max(0) + 1
    m = int(np.ceil(pad / zooms.min())) + 2
    crop = np.pad(mask[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]], m)
    din = ndi.distance_transform_edt(crop, sampling=zooms)
    dout = ndi.distance_transform_edt(~crop, sampling=zooms)
    hv = float(zooms.mean())
    sdf = np.where(crop, -(din - 0.5 * hv), dout - 0.5 * hv).astype(np.float32)
    if sigma:
        sdf = ndi.gaussian_filter(sdf, sigma)
    off = lo - m
    n0 = np.array(crop.shape) - 1
    corners = np.array([[i, j, k] for i in (0, n0[0]) for j in (0, n0[1]) for k in (0, n0[2])]) + off
    world = corners @ affine[:3, :3].T + affine[:3, 3]
    wlo, whi = world.min(0) + 0.5 * hv, world.max(0) - 0.5 * hv
    n = np.floor((whi - wlo) / h).astype(int) + 1
    axes = [wlo[i] + h * np.arange(n[i]) for i in range(3)]
    pts = np.stack(np.meshgrid(*axes, indexing="ij"), -1).reshape(-1, 3)
    inv = np.linalg.inv(affine)
    vox = pts @ inv[:3, :3].T + inv[:3, 3] - off
    vals = ndi.map_coordinates(sdf, vox.T, order=1, mode="nearest")
    return vals.reshape(n).astype(np.float32), wlo


class CTContext:
    """CT intensities + a nearest-bone partition over a region, for surface refinement.

    The bone surface is placed on the CT iso-surface (sub-voxel, cubic
    interpolation); labels only decide which bone owns a voxel. See bone_grid().
    """

    def __init__(self, ct, affine, label_masks: dict, margin_mm=15.0, smooth=0.7):
        ndi = _ndi()
        zooms = np.sqrt((affine[:3, :3] ** 2).sum(0))
        union = np.zeros(ct.shape, bool)
        for m in label_masks.values():
            union |= m
        idx = np.argwhere(union)
        mv = np.ceil(margin_mm / zooms).astype(int)
        self.lo = np.maximum(idx.min(0) - mv, 0)
        self.hi = np.minimum(idx.max(0) + 1 + mv, ct.shape)
        sl = tuple(slice(a, b) for a, b in zip(self.lo, self.hi))
        ct_c = ndi.gaussian_filter(ct[sl].astype(np.float32), smooth)  # tame CT noise
        self.coef = ndi.spline_filter(ct_c, order=3)  # prefiltered once for cubic sampling
        self.ct = ct_c
        lab = np.zeros(ct_c.shape, np.int16)
        self.ids = {}
        for k, (name, m) in enumerate(label_masks.items(), start=1):
            lab[m[sl]] = k
            self.ids[name] = k
        _, nearest = ndi.distance_transform_edt(lab == 0, sampling=zooms, return_indices=True)
        self.owner = lab[tuple(nearest)]
        self.lab = lab
        self.affine = affine

    def threshold(self, name):
        """Half-way between soft tissue just outside and cortex just inside the label."""
        ndi = _ndi()
        m = self.lab == self.ids[name]
        out = ndi.binary_dilation(m, iterations=2) & (self.lab == 0)
        rim = m & ~ndi.binary_erosion(m, iterations=1)
        return 0.5 * (float(np.median(self.ct[out])) + float(np.median(self.ct[rim])))

    def sample(self, vox_abs, name):
        """Cubic CT and ownership at absolute (canonical) voxel coordinates [3, N]."""
        ndi = _ndi()
        v = vox_abs - self.lo[:, None]
        ct = ndi.map_coordinates(self.coef, v, order=3, mode="nearest", prefilter=False)
        own = ndi.map_coordinates((self.owner == self.ids[name]).astype(np.float32), v,
                                  order=1, mode="constant", cval=0.0) > 0.5
        return ct, own


def bone_grid(mask, affine, h, ctx: CTContext | None = None, name=None, band=2.0, pad=6.0, sigma=0.7):
    """Signed distance grid (world-aligned, spacing h) for one bone.

    Without CT: distance to the binary mask, smoothed (voxel-limited).
    With CT: occupancy = deep inside the label (> band) OR (within band of the
    label surface AND owned by this bone AND CT > threshold), holes filled, then
    re-distanced on the fine grid. The label therefore only guards the surface
    to +-band; its exact position comes from the CT.
    """
    if ctx is None:
        return mask_to_grid(mask, affine, h, pad=pad)
    ndi = _ndi()
    zooms = np.sqrt((affine[:3, :3] ** 2).sum(0))
    idx = np.argwhere(mask)
    lo, hi = idx.min(0), idx.max(0) + 1
    m = int(np.ceil((pad + band) / zooms.min())) + 2
    crop = np.pad(mask[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]], m)
    din = ndi.distance_transform_edt(crop, sampling=zooms)
    dout = ndi.distance_transform_edt(~crop, sampling=zooms)
    hv = float(zooms.mean())
    s_mask = np.where(crop, -(din - 0.5 * hv), dout - 0.5 * hv).astype(np.float32)
    off = lo - m
    n0 = np.array(crop.shape) - 1
    corners = np.array([[i, j, k] for i in (0, n0[0]) for j in (0, n0[1]) for k in (0, n0[2])]) + off
    world = corners @ affine[:3, :3].T + affine[:3, 3]
    wlo, whi = world.min(0) + hv, world.max(0) - hv
    n = np.floor((whi - wlo) / h).astype(int) + 1
    axes = [wlo[i] + h * np.arange(n[i]) for i in range(3)]
    pts = np.stack(np.meshgrid(*axes, indexing="ij"), -1).reshape(-1, 3)
    inv = np.linalg.inv(affine)
    vox = (pts @ inv[:3, :3].T + inv[:3, 3]).T
    sm = ndi.map_coordinates(s_mask, vox - off[:, None], order=1, mode="nearest")
    ct, own = ctx.sample(vox, name)
    T = ctx.threshold(name)
    occ = (sm < -band) | ((sm < band) & own & (ct > T))
    occ = largest_component(ndi.binary_fill_holes(occ.reshape(n)))
    din = ndi.distance_transform_edt(occ, sampling=h)
    dout = ndi.distance_transform_edt(~occ, sampling=h)
    sdf = np.where(occ, -(din - 0.5 * h), dout - 0.5 * h).astype(np.float32)
    if sigma:
        sdf = ndi.gaussian_filter(sdf, sigma)
    return sdf, wlo


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


def body_frame(body_pts, tangent):
    """Columns (l, a, w): lateral (right), anterior, axial (cranial)."""
    c = body_pts.mean(0)
    evals, evecs = np.linalg.eigh(np.cov((body_pts - c).T))
    w = evecs[:, 0]
    if np.dot(w, tangent) < 0:
        w = -w
    if np.degrees(np.arccos(np.clip(np.dot(w, tangent), -1, 1))) > 30:
        w = tangent
    a = np.array([0.0, 1.0, 0.0])
    a = a - a.dot(w) * w
    a /= np.linalg.norm(a)
    l = np.cross(a, w)
    return c, np.stack([l, a, w], axis=1)


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


def vertebra_landmarks(f: Field, level, tangent, canal):
    body = isolate_body(f, level[0], canal, tangent)
    c, R = body_frame(body, tangent)
    local = (body - c) @ R
    uv, fp, lo = central_columns(local[:, :2], frac=0.3, h=f.h)
    uv0 = local[:, :2].mean(0)
    height = np.ptp(local[:, 2])
    out = {"level": level, "body_center": c, "R": R, "height": float(height),
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


def process(subject_dir, out_dir, levels=None, sacrum=True, pelvis=True, use_ct=True,
            h_vert=1.0, h_sacrum=None, h_other=1.5, log=print):
    """Process a subject. ``levels``: vertebrae to output (default all present).

    With ``use_ct`` and a ``ct.nii.gz`` next to the masks, bone surfaces come from
    the CT iso-surface (see bone_grid); otherwise from the masks alone.
    """
    subject_dir, out_dir = Path(subject_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    levels = levels or LEVELS[:-1]
    h_sacrum = h_sacrum or h_vert

    def load(name):
        p = subject_dir / f"{name}.nii.gz"
        if not p.exists():
            return None, None
        m, aff = load_mask(p)
        return (m if m.any() else None), aff

    # Masks: requested levels, their immediate neighbours (for the spine tangent
    # and the bone partition), sacrum + S1, hips.
    order = LEVELS[:-1]
    wanted = [lv for lv in order if lv in levels]
    lo_i, hi_i = order.index(wanted[0]), order.index(wanted[-1])
    context = order[max(lo_i - 1, 0):hi_i + 2]
    masks, aff = {}, None
    for lv in context:
        mk, a = load(f"vertebrae_{lv}")
        if mk is not None:
            masks[lv], aff = largest_component(mk), a
    sac, a1 = load("sacrum")
    s1, a2 = load("vertebrae_S1")
    if sac is not None or s1 is not None:
        mk = (sac if sac is not None else np.zeros_like(s1)) | (s1 if s1 is not None else False)
        masks["sacrum"] = largest_component(mk)
        aff = aff if aff is not None else (a1 if a1 is not None else a2)
    for name in ("hip_left", "hip_right"):
        mk, a = load(name)
        if mk is not None:
            masks[name] = largest_component(mk)

    ctx = None
    ct_path = subject_dir / "ct.nii.gz"
    if use_ct and ct_path.exists():
        import nibabel as nib
        ct_im = nib.as_closest_canonical(nib.load(str(ct_path)))
        if not np.allclose(ct_im.affine, aff, atol=1e-3):
            raise RuntimeError("CT and mask grids differ")
        log("  CT: refining surfaces from intensities")
        ctx = CTContext(np.asarray(ct_im.dataobj), aff, masks)

    grids, fields, centroids = {}, {}, {}
    for lv in context:
        if lv not in masks:
            continue
        if lv in wanted:
            v, o = bone_grid(masks[lv], aff, h_vert, ctx, lv)
            grids[lv] = ("vertebra", v, o, h_vert)
            fields[lv] = Field(v, o, h_vert)
            centroids[lv] = fields[lv].inside_points().mean(0)
            log(f"  {lv}: grid {v.shape} @ {h_vert} mm"
                + (f", CT threshold {ctx.threshold(lv):.0f} HU" if ctx else ""))
        else:  # neighbour: centroid only
            centroids[lv] = np.argwhere(masks[lv]).mean(0) @ aff[:3, :3].T + aff[:3, 3]
    if sacrum and "sacrum" in masks:
        v, o = bone_grid(masks["sacrum"], aff, h_sacrum, ctx, "sacrum")
        grids["sacrum"] = ("sacrum", v, o, h_sacrum)
        fields["sacrum"] = Field(v, o, h_sacrum)
        log(f"  sacrum: grid {v.shape} @ {h_sacrum} mm")
    if pelvis:
        for name in ("hip_left", "hip_right", "femur_left", "femur_right"):
            mk = masks.get(name)
            if mk is None:
                mk, a = load(name)
                if mk is None:
                    continue
                mk = largest_component(mk)
            if name.startswith("femur"):
                zs = np.argwhere(mk)[:, 2]
                zoom = np.sqrt((aff[:3, :3] ** 2).sum(0))[2]
                mk[:, :, :max(int(zs.max() - FEMUR_KEEP / zoom), 0)] = False
            v, o = bone_grid(mk, aff, h_other, ctx if name in masks else None, name)
            grids[name] = (name.split("_")[0], v, o, h_other)
            log(f"  {name}: grid {v.shape} @ {h_other} mm")

    cord, aff_c = load("spinal_cord")
    cord_pts = None
    if cord is not None:
        cord_pts = np.argwhere(cord) @ aff_c[:3, :3].T + aff_c[:3, 3]

    present = [lv for lv in order if lv in centroids]
    marks = {}
    for lv in wanted:
        if lv == "C1" or lv not in fields:
            continue
        i = present.index(lv)
        up = centroids[present[i - 1]] if i > 0 else centroids[lv]
        dn = centroids[present[i + 1]] if i + 1 < len(present) else centroids[lv]
        tangent = (up - dn) / np.linalg.norm(up - dn)
        canal = None
        if cord_pts is not None:
            z = centroids[lv][2]
            near = cord_pts[np.abs(cord_pts[:, 2] - z) < 6]
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

    # Origin at the S1 superior endplate centre (or the lowest landmark).
    ref = marks["S1"]["sup"]["point"] if "S1" in marks else next(iter(marks.values()))["body_center"]
    ref = np.round(ref, 1)
    save = {}
    meta = {"bones": [], "landmarks": {}, "attribution": ATTRIBUTION,
            "subject": subject_dir.name, "frame": "RAS mm, origin at S1 superior endplate centre"}
    from ..sdf import Grid
    for name, (kind, v, o, h) in grids.items():
        g = Grid(v, o - ref, h, name)
        save[f"{name}__q"] = g.q
        meta["bones"].append({"name": name, "kind": kind, "origin": (o - ref).tolist(),
                              "spacing": h, "shape": list(g.shape)})
    for lv, e in marks.items():
        d = {k: (v - ref if k in ("body_center",) else v) for k, v in e.items()
             if k != "footprint_uv"}
        for ep in ("sup", "inf"):
            if ep in d:
                d[ep] = {**d[ep], "point": d[ep]["point"] - ref}
        meta["landmarks"][lv] = _jsonable(d)
    np.savez_compressed(out_dir / "bones.npz", **save)
    (out_dir / "model.json").write_text(json.dumps(meta, indent=1))
    log(f"wrote {out_dir}/bones.npz, model.json")
    return out_dir


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
    ap.add_argument("subject_dir")
    ap.add_argument("--out", required=True)
    ap.add_argument("--levels", default="all", help='"lumbar", "all" or a comma list, e.g. L3,L4,L5')
    ap.add_argument("--no-pelvis", action="store_true", help="skip hips and femurs")
    ap.add_argument("--no-ct", action="store_true", help="ignore ct.nii.gz; use masks only")
    ap.add_argument("--h-vert", type=float, default=1.0, help="grid spacing for vertebrae (mm)")
    ap.add_argument("--h-sacrum", type=float, default=None, help="grid spacing for the sacrum (mm)")
    ap.add_argument("--h-other", type=float, default=1.5, help="grid spacing for hips/femurs (mm)")
    a = ap.parse_args(argv)
    levels = {"all": None, "lumbar": LUMBAR}.get(a.levels) or a.levels.split(",")
    process(a.subject_dir, a.out, levels, pelvis=not a.no_pelvis, use_ct=not a.no_ct,
            h_vert=a.h_vert, h_sacrum=a.h_sacrum, h_other=a.h_other)


if __name__ == "__main__":
    main()
