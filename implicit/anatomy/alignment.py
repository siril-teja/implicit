"""Spinopelvic alignment: pelvic landmarks, sagittal parameters and a chain model.

    python -m implicit.anatomy.alignment assets/bones/vivaplus-50F

Pelvic landmarks (stored as "pelvis" in landmarks.json by implicit.anatomy.landmarks):
  - femoral head centres: sphere fitted to the proximal femur surface around its
    most proximal-medial point, refitted to the points within 1.5 mm of the sphere;
    the hip axis centre H is their midpoint;
  - sacroiliac joint centres: centroid of the sacrum surface within 2 mm of its
    closest approach to each hip bone.

Sagittal parameters (degrees; frame x right, y anterior, z cranial; vectors projected
onto the y-z plane; the tilt of an upward normal n is atan2(n_y, n_z), + anterior):
  SS  sacral slope: tilt of the S1 superior endplate normal
  PT  pelvic tilt: angle of the line from the S1 endplate centre m to H with the
      vertical, + when H is anterior to m
  PI  pelvic incidence: PT + SS
  LL  lumbar lordosis: Cobb angle L1 superior to S1 superior endplate
  TK  thoracic kyphosis: Cobb angle T4 superior to T12 inferior endplate

Chain model (``solve``): every segment moves rigidly in the sagittal plane (rotation
about the x axis plus a y-z translation). Above the sacrum, each disc is a hinge at
its centre (midpoint of the two endplate centres); the vertebra above it and
everything higher turn with it.
  - PT: pelvis, sacrum and spine turn about the hip axis (femurs stay);
  - PI: the sacrum and spine turn about the sacroiliac axis relative to the hip
    bones (a morph of the pelvis; the turn is solved so PI hits the target, then
    PT is set by the pelvis turn). The sacroiliac surfaces are not refitted;
  - LL: the change is spread over the discs L1-L2 ... L5-S1 in proportion to their
    lordosis (at least 1 deg each), so the curve keeps its shape;
  - TK: likewise over T4-T5 ... T11-T12 in proportion to their kyphosis.
The spine curve is a Catmull-Rom spline through the moved S1 endplate centre and
body centres (``curve`` in the result).
Cobb angles change exactly by the turns of the discs between their endplates, so
the targets are met exactly (relative to the subject's values rounded to 0.1 deg,
which are the knobs' starting values and move nothing). viewer/solvers.js is the same computation.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from ..expr import SOLVERS
from .convert import VERTEBRAE, load_bone

CURVE_STEPS = 4                                    # spine curve samples per span between centres
LORDOSIS = [f"L{i}" for i in range(1, 6)]          # upper vertebrae of the LL discs
KYPHOSIS = [f"T{i}" for i in range(4, 12)]         # upper vertebrae of the TK discs
KNOBS = {"PI": (30.0, 90.0), "PT": (-10.0, 45.0), "LL": (0.0, 90.0), "TK": (0.0, 80.0)}


# ---------------------------------------------------------------- pelvic landmarks

def surface_points(g):
    """Grid points within half a cell of the zero level."""
    idx = np.argwhere(np.abs(g.values) < 0.5 * g.spacing)
    return g.origin + g.spacing * idx


def fit_sphere(P):
    A = np.column_stack([2 * P, np.ones(len(P))])
    x, *_ = np.linalg.lstsq(A, (P ** 2).sum(1), rcond=None)
    c = x[:3]
    return c, float(np.sqrt(x[3] + c @ c))


def femoral_head(g, side):
    """(centre, radius, rms) of the femoral head of a proximal femur ("left": x < 0)."""
    P = surface_points(g)
    medial = P[:, 0] if side == "left" else -P[:, 0]
    p0 = P[np.argmax(P[:, 2] + 0.7 * medial)]
    sel = P[np.linalg.norm(P - p0, axis=1) < 35.0]
    c, r = fit_sphere(sel)
    for _ in range(8):
        d = np.linalg.norm(P - c, axis=1)
        sel = P[(np.abs(d - r) < 1.5)]
        c, r = fit_sphere(sel)
    res = np.linalg.norm(sel - c, axis=1) - r
    return c, r, float(np.sqrt(np.mean(res ** 2)))


def sacroiliac_centre(g_sacrum, g_hip):
    """Centroid of the sacrum surface within 2 mm of its closest approach to the hip
    bone (models may leave a gap for the joint cartilage), or None if far apart."""
    P = surface_points(g_sacrum)
    d = g_hip.eval(P)
    if d.min() > 8.0:
        return None
    near = P[d < d.min() + 2.0]
    return near.mean(0) if len(near) > 20 else None


def pelvis_landmarks(bone_dir, log=print):
    """{"hip_left", "hip_right", "si_left", "si_right"} (points) from the bone files, or None."""
    bone_dir = Path(bone_dir)
    need = ["femur_left", "femur_right", "hip_left", "hip_right", "sacrum"]
    if not all((bone_dir / f"{n}.npz").exists() for n in need):
        return None
    g = {n: load_bone(bone_dir / f"{n}.npz")[0] for n in need}
    out = {}
    for side in ("left", "right"):
        c, r, rms = femoral_head(g[f"femur_{side}"], side)
        out[f"hip_{side}"] = c.tolist()
        log(f"  femoral head {side}: radius {r:.1f} mm, fit rms {rms:.2f} mm")
        si = sacroiliac_centre(g["sacrum"], g[f"hip_{side}"])
        if si is None:
            return None
        out[f"si_{side}"] = si.tolist()
    return out


# ---------------------------------------------------------------- sagittal geometry

def tilt(n):
    """Sagittal tilt of an upward normal (deg, + anterior)."""
    return math.degrees(math.atan2(n[1], n[2]))


def _up(e, ep):
    n = np.asarray(e[ep]["normal"], float)
    return n if ep == "sup" else -n


def chain(landmarks, pelvis, offset=(0.0, 0.0, 0.0)):
    """Data of the chain model (scene frame: positions minus ``offset``), or None.

    Points and normals are (y, z) pairs. Discs run from L5-S1 upwards; disc j is
    below ``levels[j]``.
    """
    if pelvis is None or "S1" not in landmarks:
        return None
    off = np.asarray(offset, float)
    yz = lambda p: [float(p[1]), float(p[2])]
    pt = lambda p: yz(np.asarray(p, float) - off)
    levels = [lv for lv in reversed(VERTEBRAE) if lv in landmarks and "sup" in landmarks[lv] and "inf" in landmarks[lv]]
    s1 = landmarks["S1"]["sup"]
    discs, below = [], landmarks["S1"]
    for lv in levels:
        e = landmarks[lv]
        c = 0.5 * (np.asarray(e["inf"]["point"], float) + np.asarray(below["sup"]["point"], float))
        discs.append({"upper": lv, "center": pt(c), "angle": tilt(_up(below, "sup")) - tilt(_up(e, "inf"))})
        below = e
    H = 0.5 * (np.asarray(pelvis["hip_left"]) + np.asarray(pelvis["hip_right"]))
    SI = 0.5 * (np.asarray(pelvis["si_left"]) + np.asarray(pelvis["si_right"]))
    # Spine curve through the S1 endplate centre and the body centres (3-D, scene frame).
    curve = [{"segment": "sacrum", "point": (np.asarray(s1["point"], float) - off).tolist()}]
    curve += [{"segment": lv, "point": (np.asarray(landmarks[lv]["body_center"], float) - off).tolist()}
              for lv in levels]
    data = {"hip": pt(H), "si": pt(SI), "s1_point": pt(s1["point"]), "s1_normal": yz(s1["normal"]),
            "levels": levels, "discs": discs, "curve": curve}
    data["measured"] = measure_chain(data, landmarks)
    # The knobs start at these (0.1 deg); targets are taken relative to them, so the
    # starting values move nothing.
    data["base"] = {k: round(v, 1) for k, v in data["measured"].items()}
    return data


def _pt_ss(m, n, H):
    pt = math.degrees(math.atan2(H[0] - m[0], m[1] - H[1]))
    return pt, math.degrees(math.atan2(n[0], n[1]))


def measure_chain(data, landmarks):
    pt, ss = _pt_ss(data["s1_point"], data["s1_normal"], data["hip"])
    out = {"PI": pt + ss, "PT": pt, "SS": ss}
    if "L1" in landmarks:
        out["LL"] = tilt(landmarks["S1"]["sup"]["normal"]) - tilt(_up(landmarks["L1"], "sup"))
    if "T4" in landmarks and "T12" in landmarks:
        out["TK"] = tilt(_up(landmarks["T4"], "sup")) - tilt(_up(landmarks["T12"], "inf"))
    return out


# ---------------------------------------------------------------- chain model

def _rot(c, a):
    """Rigid motion (angle rad, translation) of a turn by ``a`` about point c (y, z)."""
    ca, sa = math.cos(a), math.sin(a)
    return a, (c[0] - (ca * c[0] - sa * c[1]), c[1] - (sa * c[0] + ca * c[1]))


def _compose(A, B):
    """A after B."""
    a, (ty, tz) = A
    b, (uy, uz) = B
    ca, sa = math.cos(a), math.sin(a)
    return a + b, (ca * uy - sa * uz + ty, sa * uy + ca * uz + tz)


def _apply(T, p):
    a, (ty, tz) = T
    ca, sa = math.cos(a), math.sin(a)
    return (ca * p[0] - sa * p[1] + ty, sa * p[0] + ca * p[1] + tz)


def _pi_after(data, beta):
    """PI and PT after turning the sacrum by ``beta`` (rad) about the sacroiliac centre."""
    T = _rot(data["si"], beta)
    m = _apply(T, data["s1_point"])
    n = data["s1_normal"]
    cb, sb = math.cos(beta), math.sin(beta)
    pt, ss = _pt_ss(m, (cb * n[0] - sb * n[1], sb * n[0] + cb * n[1]), data["hip"])
    return pt + ss, pt


def _weights(discs, uppers, sign):
    w = [max(sign * d["angle"], 1.0) if d["upper"] in uppers else 0.0 for d in discs]
    s = sum(w)
    return [x / s if s > 0 else 0.0 for x in w]


def solve(data, PI, PT, LL, TK):
    """{segment: [cos a, sin a, ty, tz]}: the rigid sagittal motion of each segment
    ("pelvis", "sacrum" and the levels) as p -> R(a) p + t in the (y, z) plane."""
    base, meas = data["base"], data["measured"]
    PI += meas["PI"] - base["PI"]  # knob values are relative to the rounded starting values
    PT += meas["PT"] - base["PT"]
    # PI: sacrum turn about the sacroiliac axis (bisection; PI is monotonic in it).
    lo, hi = -math.radians(60), math.radians(60)
    f_lo = _pi_after(data, lo)[0] - PI
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        f_mid = _pi_after(data, mid)[0] - PI
        if (f_mid > 0) == (f_lo > 0):
            lo, f_lo = mid, f_mid
        else:
            hi = mid
    beta = 0.5 * (lo + hi)
    if abs(PI - meas["PI"]) < 1e-9:
        beta = 0.0
    alpha = math.radians(PT - _pi_after(data, beta)[1])
    pelvis = _rot(data["hip"], alpha)
    T = _compose(pelvis, _rot(data["si"], beta))
    out = {"pelvis": pelvis, "sacrum": T}
    d_ll = math.radians(LL - base.get("LL", LL))
    d_tk = math.radians(TK - base.get("TK", TK))
    wl = _weights(data["discs"], LORDOSIS, 1.0)
    wk = _weights(data["discs"], KYPHOSIS, -1.0)
    for d, a, b in zip(data["discs"], wl, wk):
        T = _compose(T, _rot(d["center"], d_ll * a - d_tk * b))
        out[d["upper"]] = T
    res = {k: [math.cos(a), math.sin(a), t[0], t[1]] for k, (a, t) in out.items()}
    res["curve"] = _curve([[c["point"][0], *_apply(out[c["segment"]], c["point"][1:])] for c in data["curve"]])
    return res


def _curve(P):
    """Catmull-Rom spline through points P (end points repeated), CURVE_STEPS samples per
    span plus the last point; flat [x, y, z, x, y, z, ...]."""
    n, out = len(P), []
    for i in range(n - 1):
        p0, p1, p2, p3 = P[max(i - 1, 0)], P[i], P[i + 1], P[min(i + 2, n - 1)]
        for k in range(CURVE_STEPS):
            t = k / CURVE_STEPS
            t2, t3 = t * t, t * t * t
            out += [0.5 * (2 * b + (c - a) * t + (2 * a - 5 * b + 4 * c - d) * t2 + (3 * b - a - 3 * c + d) * t3)
                    for a, b, c, d in zip(p0, p1, p2, p3)]
    return out + list(P[-1])


SOLVERS["spinopelvic"] = solve  # the Spinopelvic alignment block (implicit.graph)


def measure(bone_dir):
    """Sagittal parameters of a subject (landmarks.json with its "pelvis" entry)."""
    d = json.loads((Path(bone_dir) / "landmarks.json").read_text())
    data = chain(d["landmarks"], d.get("pelvis"))
    if data is None:
        raise SystemExit(f"{bone_dir}: no pelvic landmarks (needs femurs, hip bones and sacrum)")
    return data


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("bone_dir")
    data = measure(ap.parse_args(argv).bone_dir)
    print("  ".join(f"{k} {v:.1f}" for k, v in data["base"].items()))
    print("discs (deg, + lordotic): " + "  ".join(f"{d['upper']}: {d['angle']:.1f}" for d in data["discs"]))


if __name__ == "__main__":
    main()
