"""Vertebral landmarks from implicit bones: body frames and endplate planes.

    python -m implicit.anatomy.landmarks assets/bones/s0751

Reads the per-bone files written by ``implicit.anatomy.convert`` (never the
masks) and writes ``landmarks.json`` next to them, in the same scanner frame.

Per vertebra (C2 to L5, plus the S1 superior endplate)
  1. isolate the vertebral body: drop points behind the posterior wall (found
     by marching anteriorly from the spinal-cord centroid, or from the centre of
     the vertebral foramen when there is no cord file), morphological opening (removes pedicles and processes), largest
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
With femurs, hip bones and sacrum present, the pelvic landmarks (femoral head and
sacroiliac joint centres, see implicit.anatomy.alignment) are stored as "pelvis".

Requires scipy.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .convert import VERTEBRAE, load_bones

OPEN_RADIUS = {"C": 4.5, "T": 4.5, "L": 5.5, "S": 5.5, "L5": 7.0}  # mm, body isolation
# (by region; L5 by level: its transverse processes are thick enough to survive 5.5)


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


def foramen_centre(f: Field, z, slab=4.0):
    """Centre of the vertebral foramen near height ``z``, or None.

    Used when there is no spinal-cord file: the bone inside a thin axial slab is
    projected onto the axial plane and the largest enclosed hole is the canal.
    """
    ndi = _ndi()
    pts = f.inside_points()
    pts = pts[np.abs(pts[:, 2] - z) < slab]
    if len(pts) < 50:
        return None
    lo = pts[:, :2].min(0) - 2 * f.h
    ij = np.round((pts[:, :2] - lo) / f.h).astype(int)
    img = np.zeros(ij.max(0) + 3, dtype=bool)
    img[ij[:, 0], ij[:, 1]] = True
    img = ndi.binary_closing(img, iterations=2)
    holes = ndi.binary_fill_holes(img) & ~img
    lab, n = ndi.label(holes)
    if n == 0:
        return None
    sizes = ndi.sum(holes, lab, range(1, n + 1))
    k = int(np.argmax(sizes)) + 1
    if sizes[k - 1] * f.h ** 2 < 40:  # mm^2; smaller holes are foramina of vessels
        return None
    uv = lo + f.h * np.argwhere(lab == k).mean(0)
    return np.array([uv[0], uv[1], z])


def isolate_body(f: Field, level, canal=None, tangent=None):
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
    r = OPEN_RADIUS.get(level, OPEN_RADIUS[level[0]])
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
    body = isolate_body(f, level, canal, tangent)
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
        else:
            canal = foramen_centre(fields[lv], centroids[lv][2])
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
    from .rotation import add_centres
    add_centres(marks)  # physiological rotation centres of each disc
    out = {lv: _jsonable({k: v for k, v in e.items() if k != "footprint_uv"})
           for lv, e in marks.items()}
    doc = {"frame": "scanner RAS, mm", "landmarks": out}
    from .alignment import pelvis_landmarks
    pelvis = pelvis_landmarks(bone_dir, log)
    if pelvis:
        doc["pelvis"] = _jsonable({k: np.asarray(v) for k, v in pelvis.items()})
    (bone_dir / "landmarks.json").write_text(json.dumps(doc, indent=1))
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
