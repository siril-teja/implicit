"""Spine alignment: pelvic landmarks, sagittal and coronal parameters, a chain model
of rigid segments and a thin-plate spline (TPS) for everything between them.

    python -m implicit.anatomy.alignment assets/bones/vivaplus-50F

Pelvic landmarks (stored as "pelvis" in landmarks.json by implicit.anatomy.landmarks):
  - femoral head centres: sphere fitted to the proximal femur surface around its
    most proximal-medial point, refitted to the points within 1.5 mm of the sphere;
    the hip axis centre H is their midpoint;
  - sacroiliac joint centres: centroid of the sacrum surface within 2 mm of its
    closest approach to each hip bone.

Parameters (degrees unless noted; frame x right, y anterior, z cranial). Sagittal
values use vectors projected onto the y-z plane, coronal ones onto x-z; the tilt of
an upward normal n is atan2(n_y, n_z) (+ anterior), its coronal tilt atan2(n_x, n_z)
(+ right).
  SS  sacral slope: tilt of the S1 superior endplate normal
  PT  pelvic tilt: angle of the line from the S1 endplate centre m to H with the
      vertical, + when H is anterior to m
  PI  pelvic incidence: PT + SS
  LL  lumbar lordosis: Cobb angle L1 superior to S1 superior endplate
  TK  thoracic kyphosis: Cobb angle T4 superior to T12 inferior endplate
  MT, TL  coronal Cobb angles of the main thoracic and thoracolumbar/lumbar curves
      (upper end vertebra superior to lower end vertebra inferior endplate; + convex
      to the right), between chosen end vertebrae, with a chosen apex
  CB  coronal balance (mm): T1 body centre right of the S1 endplate centre
  PO  pelvic obliquity: the pelvis turned about the antero-posterior axis through
      H, + right side up
  AR  apical rotation gain: axial rotation of a curve's apex vertebra per degree of
      Cobb change (toward the convexity), tapering linearly to 0 at the end vertebrae

Chain model (``solve``): every segment ("legs", "pelvis", "sacrum", the vertebrae)
moves rigidly. Above the sacrum each disc is a hinge at its centre (midpoint of the
two endplate centres); the vertebra above it and everything higher turn with it.
  - PO: everything, femurs included, turns about the antero-posterior axis through H;
  - PT: pelvis, sacrum and spine turn about the hip axis (femurs stay);
  - PI: the sacrum and spine turn about the sacroiliac axis relative to the hip
    bones (a morph of the pelvis; the turn is solved so PI hits the target, then
    PT is set by the pelvis turn). The sacroiliac surfaces are not refitted;
  - LL: the change is spread over the discs L1-L2 ... L5-S1 in proportion to their
    lordosis (at least 1 deg each), so the curve keeps its shape;
  - TK: likewise over T4-T5 ... T11-T12 in proportion to their kyphosis;
  - MT, TL: half of a curve's Cobb change goes to the discs between its lower end
    and apex, half to those between apex and upper end (equal shares), all bending
    the same way (lateral bending about the antero-posterior axis); the vertebra
    farthest from the line through the end vertebrae is the apex, or a neighbour
    when the two halves have very different numbers of discs;
  - CB: the discs below the lowest curve (the lumbosacral fractional curve) bend
    so that T1 sits at the target offset (bisection);
  - AR: each curve vertebra also turns about its own axial axis, locally (the turn
    does not carry the vertebrae above).
Sagittal Cobb angles change exactly by the sagittal turns of the discs between
their endplates, coronal ones by the lateral turns; when both planes change, the
projections interact slightly (second order). Knob values are taken relative to the
subject's values rounded to 0.1 (their starting values, which move nothing).

TPS: control points on every segment (body centre, +-15 mm along its axial axis,
15 mm right and anterior; femoral head centres on the pelvis) are moved with their
segment, and a 3-D thin-plate spline (kernel |r|, plus an affine part) is fitted
from the moved positions back to the original ones: the inverse warp, which is what
evaluating a deformed implicit body needs. Bones stay rigid; soft structures (the
spine curve now) are their baseline shape carried by the warp.
viewer/solvers.js is the same computation.
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
KNOBS = {"PI": (30.0, 90.0), "PT": (-10.0, 45.0), "LL": (0.0, 90.0), "TK": (0.0, 80.0),
         "MT": (-90.0, 90.0), "TL": (-90.0, 90.0), "AR": (0.0, 1.0), "PO": (-20.0, 20.0), "CB": (-80.0, 80.0)}
CURVES = {"MT": ("T5", "T8", "T12"), "TL": ("T12", "L1", "L4")}  # default (upper end, apex, lower end)
TPS_ARM = 15.0                                     # mm, control point offsets from a body centre


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


# ---------------------------------------------------------------- geometry

def tilt(n):
    """Sagittal tilt of an upward normal (deg, + anterior)."""
    return math.degrees(math.atan2(n[1], n[2]))


def ctilt(n):
    """Coronal tilt of an upward normal (deg, + right)."""
    return math.degrees(math.atan2(n[0], n[2]))


def _up(e, ep):
    n = np.asarray(e[ep]["normal"], float)
    return n if ep == "sup" else -n


def _axes(e):
    """Orthonormal frame columns (right, anterior, cranial) of a landmark entry."""
    u, _, vt = np.linalg.svd(np.array(e["R"], float))
    return (u @ vt).T.tolist()


def chain(landmarks, pelvis, offset=(0.0, 0.0, 0.0)):
    """Data of the chain model (scene frame: positions minus ``offset``), or None.

    Discs run from L5-S1 upwards; disc j is below ``levels[j]``.
    """
    if pelvis is None or "S1" not in landmarks:
        return None
    off = np.asarray(offset, float)
    pt = lambda p: (np.asarray(p, float) - off).tolist()
    levels = [lv for lv in reversed(VERTEBRAE) if lv in landmarks and "sup" in landmarks[lv] and "inf" in landmarks[lv]]
    s1 = landmarks["S1"]
    discs, below = [], s1
    for lv in levels:
        e = landmarks[lv]
        c = 0.5 * (np.asarray(e["inf"]["point"], float) + np.asarray(below["sup"]["point"], float))
        discs.append({"upper": lv, "center": pt(c), "angle": tilt(_up(below, "sup")) - tilt(_up(e, "inf"))})
        below = e
    frames = {lv: {"center": pt(landmarks[lv]["body_center"]), "axes": _axes(landmarks[lv])} for lv in levels}
    frames["sacrum"] = {"center": pt(s1["sup"]["point"]), "axes": _axes(s1)}
    hips = {k: pt(pelvis[k]) for k in ("hip_left", "hip_right", "si_left", "si_right")}
    H = (0.5 * (np.array(hips["hip_left"]) + np.array(hips["hip_right"]))).tolist()
    SI = (0.5 * (np.array(hips["si_left"]) + np.array(hips["si_right"]))).tolist()
    # TPS control points: five per segment, femoral head centres on the pelvis.
    tps = [{"segment": "pelvis", "point": hips[k]} for k in ("hip_left", "hip_right")]
    for seg in ["sacrum"] + levels:
        c, (l, a, w) = np.array(frames[seg]["center"]), np.array(frames[seg]["axes"])
        tps += [{"segment": seg, "point": q.tolist()}
                for q in (c, c + TPS_ARM * w, c - TPS_ARM * w, c + TPS_ARM * l, c + TPS_ARM * a)]
    data = {"hip": H, "si": SI, "s1_point": pt(s1["sup"]["point"]), "s1_normal": list(s1["sup"]["normal"]),
            "levels": levels, "discs": discs, "frames": frames, "tps_points": tps,
            "curves": {k: list(c) for k, c in CURVES.items() if all(lv in levels for lv in c)},
            "curve": _curve([frames["sacrum"]["center"]] + [frames[lv]["center"] for lv in levels])}
    data["measured"] = measure_chain(data, landmarks)
    # The knobs start at these (0.1 deg / mm); targets are taken relative to them, so
    # the starting values move nothing.
    data["base"] = {k: round(v, 1) for k, v in data["measured"].items()}
    return data


def _pt_ss(m, n, H):
    """PT and SS from the S1 endplate centre m, its normal n and H, as (y, z) pairs."""
    pt = math.degrees(math.atan2(H[0] - m[0], m[1] - H[1]))
    return pt, math.degrees(math.atan2(n[0], n[1]))


def coronal_cobb(landmarks, upper, lower):
    """Coronal Cobb angle (+ convex right): upper end superior to lower end inferior endplate."""
    return ctilt(_up(landmarks[lower], "inf")) - ctilt(_up(landmarks[upper], "sup"))


def measure_chain(data, landmarks):
    yz = lambda v: (v[1], v[2])
    pt, ss = _pt_ss(yz(data["s1_point"]), yz(data["s1_normal"]), yz(data["hip"]))
    out = {"PI": pt + ss, "PT": pt, "SS": ss}
    if "L1" in landmarks:
        out["LL"] = tilt(landmarks["S1"]["sup"]["normal"]) - tilt(_up(landmarks["L1"], "sup"))
    if "T4" in landmarks and "T12" in landmarks:
        out["TK"] = tilt(_up(landmarks["T4"], "sup")) - tilt(_up(landmarks["T12"], "inf"))
    for k, (u, _, lo) in data["curves"].items():
        if u in landmarks and lo in landmarks:
            out[k] = coronal_cobb(landmarks, u, lo)
    top = data["levels"][-1]
    out["CB"] = landmarks[top]["body_center"][0] - landmarks["S1"]["sup"]["point"][0]
    return out


# ---------------------------------------------------------------- rigid motions
# A motion is (R, t), R a 3x3 row list: p -> R p + t. Plain Python (as solvers.js).

X_AXIS, Y_AXIS = (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)
IDENTITY = ([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], [0.0, 0.0, 0.0])


def _mv(R, v):
    return [R[i][0] * v[0] + R[i][1] * v[1] + R[i][2] * v[2] for i in range(3)]


def _rot(c, axis, a):
    """Turn by ``a`` (rad) about the axis through point c (Rodrigues)."""
    x, y, z = axis
    ca, sa = math.cos(a), math.sin(a)
    k = 1.0 - ca
    R = [[ca + x * x * k, x * y * k - z * sa, x * z * k + y * sa],
         [y * x * k + z * sa, ca + y * y * k, y * z * k - x * sa],
         [z * x * k - y * sa, z * y * k + x * sa, ca + z * z * k]]
    Rc = _mv(R, c)
    return R, [c[i] - Rc[i] for i in range(3)]


def _compose(A, B):
    """A after B."""
    (Ra, ta), (Rb, tb) = A, B
    R = [[sum(Ra[i][k] * Rb[k][j] for k in range(3)) for j in range(3)] for i in range(3)]
    u = _mv(Ra, tb)
    return R, [u[i] + ta[i] for i in range(3)]


def _apply(T, p):
    u = _mv(T[0], p)
    return [u[i] + T[1][i] for i in range(3)]


def _pi_after(data, beta):
    """PI and PT after turning the sacrum by ``beta`` (rad) about the sacroiliac axis."""
    m = _apply(_rot(data["si"], X_AXIS, beta), data["s1_point"])
    n = _mv(_rot(data["si"], X_AXIS, beta)[0], data["s1_normal"])
    pt, ss = _pt_ss((m[1], m[2]), (n[1], n[2]), (data["hip"][1], data["hip"][2]))
    return pt + ss, pt


def _bisect(f, lo, hi, target):
    """x in [lo, hi] with f(x) = target, f monotonic (60 halvings)."""
    f_lo = f(lo) - target
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        f_mid = f(mid) - target
        if (f_mid > 0) == (f_lo > 0):
            lo, f_lo = mid, f_mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def _weights(discs, uppers, sign):
    w = [max(sign * d["angle"], 1.0) if d["upper"] in uppers else 0.0 for d in discs]
    s = sum(w)
    return [x / s if s > 0 else 0.0 for x in w]


def _curve_plan(levels, curves):
    """Per curve: (disc shares of its Cobb change, axial-rotation shape per level)."""
    idx = {lv: i for i, lv in enumerate(levels)}
    plans = {}
    for k, (u, a, lo) in curves.items():
        if not all(v in idx for v in (u, a, lo)):
            raise ValueError(f"{k} curve: {u}, {a}, {lo} must all be vertebrae of this spine")
        iu, ia, il = idx[u], idx[a], idx[lo]
        if not iu > ia > il:
            raise ValueError(f"{k} curve: need upper end above apex above lower end ({u}, {a}, {lo})")
        low, up = [j for j in range(il + 1, ia + 1)], [j for j in range(ia + 1, iu + 1)]
        share = [0.0] * len(levels)
        for half, other in ((low, up), (up, low)):
            for j in half:
                share[j] = (0.5 if other else 1.0) / len(half)
        shape = [0.0] * len(levels)
        for i in range(il, iu + 1):
            span = (ia - il) if i <= ia else (iu - ia)
            shape[i] = 1.0 - abs(i - ia) / span
        plans[k] = (share, shape, il)
    return plans


def solve(data, PI, PT, LL, TK, MT=0.0, TL=0.0, AR=0.0, PO=0.0, CB=0.0):
    """{segment: [R columns (9), t (3)]}: the rigid motion p -> R p + t of each segment
    ("legs", "pelvis", "sacrum" and the levels), plus "tps": the inverse warp table
    (see tps_table)."""
    base, meas = data["base"], data["measured"]
    rel = lambda k, v: v + meas[k] - base[k] if k in base else v  # relative to the rounded starting values
    PI, PT, CB = rel("PI", PI), rel("PT", PT), rel("CB", CB)
    delta = lambda k, v: math.radians(v - base.get(k, v))
    levels, discs = data["levels"], data["discs"]
    # Pelvis: obliquity and tilt about the hips; PI by the sacrum about the SI axis.
    beta = 0.0 if abs(PI - meas["PI"]) < 1e-9 else _bisect(lambda b: _pi_after(data, b)[0], -math.radians(60),
                                                          math.radians(60), PI)
    alpha = math.radians(PT - _pi_after(data, beta)[1])
    legs = _rot(data["hip"], Y_AXIS, -math.radians(PO))
    pelvis = _compose(legs, _rot(data["hip"], X_AXIS, alpha))
    sacrum = _compose(pelvis, _rot(data["si"], X_AXIS, beta))
    # Disc turns: sagittal (LL, TK) and coronal (curves, then the balance below them).
    d_ll, d_tk = delta("LL", LL), delta("TK", TK)
    wl, wk = _weights(discs, LORDOSIS, 1.0), _weights(discs, KYPHOSIS, -1.0)
    sag = [d_ll * a - d_tk * b for a, b in zip(wl, wk)]
    plans = _curve_plan(levels, data["curves"])
    cor, axial = [0.0] * len(discs), [0.0] * len(levels)
    targets = {"MT": MT, "TL": TL}  # curves whose vertebrae the subject lacks are not in the plan
    for k, (share, shape, _) in plans.items():
        d = delta(k, targets[k])
        for j in range(len(discs)):
            cor[j] -= d * share[j]  # convex right: each disc bends the spine above to the left
            axial[j] -= AR * d * shape[j]  # the body turns toward the convexity
    lowest = min(il for _, _, il in plans.values()) if plans else -1
    frac = [j for j in range(len(discs)) if j <= lowest]

    def build(bal):
        T, out = sacrum, {}
        for j, d in enumerate(discs):
            c = d["center"]
            turn = cor[j] + (bal / len(frac) if j in frac else 0.0)
            T = _compose(T, _compose(_rot(c, X_AXIS, sag[j]), _rot(c, Y_AXIS, turn)))
            out[d["upper"]] = T
        return out

    def offset(bal):
        out = build(bal)
        return _apply(out[levels[-1]], data["frames"][levels[-1]]["center"])[0] - _apply(sacrum, data["s1_point"])[0]

    moved_cor = any(abs(v) > 0 for v in cor) or abs(PO) > 0 or abs(alpha) > 0 or abs(beta) > 0 or any(sag)
    bal = 0.0
    if frac and (moved_cor or abs(CB - meas["CB"]) > 1e-9):
        bal = _bisect(offset, -math.radians(30), math.radians(30), CB)
    out = build(bal)
    for i, lv in enumerate(levels):  # axial rotation about the vertebra's own axis
        f = data["frames"][lv]
        out[lv] = _compose(out[lv], _rot(f["center"], f["axes"][2], axial[i]))
    out.update(legs=legs, pelvis=pelvis, sacrum=sacrum)
    res = {k: [R[0][0], R[1][0], R[2][0], R[0][1], R[1][1], R[2][1], R[0][2], R[1][2], R[2][2], *t]
           for k, (R, t) in out.items()}
    res["tps"] = tps_table([(q["point"], _apply(out[q["segment"]], q["point"])) for q in data["tps_points"]])
    return res


def tps_table(pairs):
    """Inverse TPS from (original, moved) point pairs, as a flat RGBA table 3 texels wide:
    row 0 the affine part (texel c: coefficients of x, y, z and the constant for output
    component c), row i > 0 the moved point i (texel 0) and its weight (texel 1).
    The warp is f(p) = A p + b + sum_i w_i |p - y_i|."""
    X = np.array([a for a, _ in pairs], float)
    Y = np.array([b for _, b in pairs], float)
    n = len(Y)
    M = np.zeros((n + 4, n + 4))
    M[:n, :n] = np.linalg.norm(Y[:, None] - Y[None], axis=2)
    M[:n, n] = M[n, :n] = 1.0
    M[:n, n + 1:] = Y
    M[n + 1:, :n] = Y.T
    rhs = np.zeros((n + 4, 3))
    rhs[:n] = X
    sol = np.linalg.solve(M, rhs)
    w, a = sol[:n], sol[n:]
    rows = [[*a[1:, c], a[0, c]] for c in range(3)]
    for i in range(n):
        rows += [[*Y[i], 0.0], [*w[i], 0.0], [0.0, 0.0, 0.0, 0.0]]
    return [float(v) for r in rows for v in r]


def tps_eval(table, P):
    """The warp of tps_table at points P (n, 3)."""
    T = np.asarray(table, float).reshape(-1, 3, 4)
    A = T[0]
    out = P @ A[:, :3].T + A[:, 3]
    Y, W = T[1:, 0, :3], T[1:, 1, :3]
    return out + np.linalg.norm(P[:, None] - Y[None], axis=2) @ W


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


SOLVERS["spinopelvic"] = solve  # the Spine alignment block (implicit.graph)


def measure(bone_dir):
    """Chain data of a subject (landmarks.json with its "pelvis" entry)."""
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
