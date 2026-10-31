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
moves rigidly. Above the sacrum each disc is a joint turning about its physiological
rotation centres (implicit.anatomy.rotation: one point for sagittal turns, one for
coronal turns, stored as "rotation" with the upper vertebra's landmarks); the
vertebra above it and everything higher turn with it.
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
    so that T1 sits at the target offset, solved on the lean of the S1-T1 line (which
    grows steadily with the bend, unlike the offset) on its monotonic branch around no
    bend (_branch_solve, else the closest the bend can bring it); the bend, in all,
    is exact up to
    BAL_KNEE (20 deg) and eases to at most BAL_MAX (30 deg) beyond (_soft_limit), so
    a balance that needs more is not reached and the posture still changes smoothly
    with the sliders (e.g. TL below about -35 deg on VIVA+ 50F, whose TL curve ends
    at L4, leaving two discs to balance it);
  - AR: each curve vertebra also turns about its own axial axis, locally (the turn
    does not carry the vertebrae above).
Sagittal Cobb angles change exactly by the sagittal turns of the discs between
their endplates, coronal ones by the lateral turns; when both planes change, the
projections interact slightly (second order). Knob values are taken relative to the
subject's values rounded to 0.1 (their starting values, which move nothing).

Soft-tissue field (``field`` in the data): what carries everything that is not a
bone. Two kinds, both exactly rigid at the bones:
TPS: control points on every segment (body centre, +-15 mm along its axial axis,
15 mm right and anterior; femoral head centres on the pelvis) are moved with their
segment, and a 3-D thin-plate spline (kernel |r|, plus an affine part) is fitted
from the moved positions back to the original ones: the inverse warp, which is what
evaluating a deformed implicit body needs. Bones stay rigid; soft structures (the
spine curve now) are their baseline shape carried by the warp.
Polyrigid (log-Euclidean, along the chain): the endplate planes split space into
vertebral bodies and disc spaces (planes S1 superior, L5 inferior, L5 superior,
..., T1 superior; a point's region is the number of planes it lies above). Inside
a body a point moves with that vertebra; in a disc space with the motion
T_below exp(lam log(T_below^-1 T_above)), lam its position between the two
endplate planes (0 at the lower, 1 at the upper): the screw motion between the
two vertebrae, continuous across the planes. Its inverse is found by fixed-point
iteration (POLY_ITERS steps, the last ones half steps so that a guess cannot cycle
between two regions; the region and lam are re-evaluated at each guess). The
field folds where the endplate planes of a disc cross, about h / tan(wedge) from
the disc centre (posteriorly for a lordotic disc): there the inverse is not unique.
Within 25 mm of the spine curve the inverse is found for > 99 % of points at large
deformations (MT 60, TL -45, LL 80, TK 60 deg), all of them at moderate ones.
The field display grid holds the forward displacement of rest points
(``grid``: dx, dy, dz, |d| per point).
viewer/solvers.js is the same computation.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from ..expr import SOLVERS
from ..sdf import FLOW_MARGIN, FLOW_ROW, POLY_ITERS, POLY_RELAX
from . import rotation
from .convert import VERTEBRAE, load_bone

CURVE_STEPS = 4                                    # spine curve samples per span between centres
LORDOSIS = [f"L{i}" for i in range(1, 6)]          # upper vertebrae of the LL discs
KYPHOSIS = [f"T{i}" for i in range(4, 12)]         # upper vertebrae of the TK discs
KNOBS = {"PI": (30.0, 90.0), "PT": (-10.0, 45.0), "LL": (0.0, 90.0), "TK": (0.0, 80.0),
         "MT": (-90.0, 90.0), "TL": (-90.0, 90.0), "AR": (0.0, 1.0), "PO": (-20.0, 20.0), "CB": (-80.0, 80.0)}
BAL_KNEE, BAL_MAX = 20.0, 30.0   # deg: coronal balance bend of the fractional discs, exact up to the knee
BAL_SEARCH = 85.0                 # deg: interval searched for the balance before the limit
CURVES = {"MT": ("T5", "T8", "T12"), "TL": ("T12", "L1", "L4")}  # default (upper end, apex, lower end)
TPS_ARM = 15.0                                     # mm, control point offsets from a body centre
GRID_STEP, GRID_PAD = 15.0, (60.0, 60.0, 20.0)     # mm, field display grid around the spine curve
FLOW_STEPS = 10                                    # flow lines: steps from the subject's values to the current
FLOW_SEED_STEP, FLOW_SEED_REACH = 25.0, 40.0       # mm: seed grid spacing, seeds this close to the spine curve
FLOW_CELL, FLOW_PAD = 15.0, 100.0                  # mm: line lookup cells, their box around the display grid
FLOW_RMAX = 3.0                                    # mm: largest line radius (lookup margin: sdf.FLOW_MARGIN)
FLOW_SKIP = 4                                      # empty cell layers counted around a cell, at most


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
    discs, below, below_name = [], s1, "S1"
    for lv in levels:
        e = landmarks[lv]
        c = 0.5 * (np.asarray(e["inf"]["point"], float) + np.asarray(below["sup"]["point"], float))
        rc = e.get("rotation")
        if rc is None:  # landmarks written before rotation centres were stored
            src = landmarks["L5"] if below_name == "S1" else below
            rc = rotation.centres(e, below, below_name, (src["depth"], src["height"]))
        discs.append({"upper": lv, "center": pt(c), "angle": tilt(_up(below, "sup")) - tilt(_up(e, "inf")),
                      "sagittal": pt(rc["sagittal"]), "coronal": pt(rc["coronal"])})
        below, below_name = e, lv
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
    # Endplate planes (point, upward normal) per chain segment, for the polyrigid field.
    planes = {"sacrum": {"sup": [pt(s1["sup"]["point"]), list(s1["sup"]["normal"])]}}
    for lv in levels:
        e = landmarks[lv]
        planes[lv] = {"inf": [pt(e["inf"]["point"]), (-np.asarray(e["inf"]["normal"], float)).tolist()],
                      "sup": [pt(e["sup"]["point"]), list(e["sup"]["normal"])]}
    curve = _curve([frames["sacrum"]["center"]] + [frames[lv]["center"] for lv in levels])
    C = np.reshape(curve, (-1, 3))
    lo, hi = C.min(0) - GRID_PAD, C.max(0) + GRID_PAD
    n = np.ceil((hi - lo) / GRID_STEP).astype(int) + 1
    # Flow line seeds near the curve, and the cells their lines are looked up in.
    sx, sy, sz = (np.arange(a, b + 1e-9, FLOW_SEED_STEP) for a, b in zip(lo, hi))
    seeds = np.stack(np.meshgrid(sx, sy, sz, indexing="ij"), -1).reshape(-1, 3)
    seeds = seeds[np.min(np.linalg.norm(seeds[:, None] - C[None], axis=2), axis=1) <= FLOW_SEED_REACH]
    flo = lo - FLOW_PAD
    fn = np.ceil((hi + FLOW_PAD - flo) / FLOW_CELL).astype(int)
    data = {"hip": H, "si": SI, "s1_point": pt(s1["sup"]["point"]), "s1_normal": list(s1["sup"]["normal"]),
            "planes": planes, "field": "tps", "grid": {"lo": lo.tolist(), "step": GRID_STEP, "n": n.tolist()},
            "flow": {"seeds": seeds.tolist(), "lo": flo.tolist(), "cell": FLOW_CELL, "n": fn.tolist()},
            "levels": levels, "discs": discs, "frames": frames, "tps_points": tps,
            "curves": {k: list(c) for k, c in CURVES.items() if all(lv in levels for lv in c)},
            "curve": curve}
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
    """x in [lo, hi] with f(x) = target, f monotonic (60 halvings). A target outside
    [f(lo), f(hi)] gives the nearer end of the interval (saturation): the result stays
    continuous in the target (it used to jump to the far end)."""
    f_lo = f(lo) - target
    f_hi = f(hi) - target
    if (f_lo > 0) == (f_hi > 0):
        return lo if abs(f_lo) <= abs(f_hi) else hi
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        f_mid = f(mid) - target
        if (f_mid > 0) == (f_lo > 0):
            lo, f_lo = mid, f_mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def _branch_solve(f, target, lim, n=34):
    """x with f(x) = target on the monotonic branch of f around 0 within [-lim, lim]
    (scanned in n steps each way; a turning point is located by golden-section search;
    a jump of more than pi, an angle wrapping, also ends the branch).
    A target outside the branch's range gives the nearer end of the branch: the
    turning point (the closest f can come) or +-lim. Continuous in the target and in f,
    where a plain bisection over [-lim, lim] would flip between ends when f is not
    monotonic there."""
    h = lim / n
    f0 = f(0.0)
    s = 1.0 if f(h) >= f0 else -1.0  # f increases in direction s
    ends = []
    for sgn in (1.0, -1.0):
        prev, end = f0, sgn * lim
        for i in range(1, n + 1):
            x = sgn * i * h
            fx = f(x)
            if abs(fx - prev) > math.pi:  # an angle wrapped: the branch ends before it
                end = sgn * (i - 1) * h
                break
            if (fx - prev) * sgn * s < 0:  # turned between steps i-2 and i
                a, b = sgn * max(i - 2, 0) * h, x
                g = lambda y: sgn * s * f(y)  # maximised at the turning point
                lo, hi = min(a, b), max(a, b)
                r = (math.sqrt(5.0) - 1.0) / 2.0
                c, d = hi - r * (hi - lo), lo + r * (hi - lo)
                gc, gd = g(c), g(d)
                for _ in range(40):
                    if gc >= gd:
                        hi, d, gd = d, c, gc
                        c = hi - r * (hi - lo)
                        gc = g(c)
                    else:
                        lo, c, gc = c, d, gd
                        d = lo + r * (hi - lo)
                        gd = g(d)
                end = 0.5 * (lo + hi)
                break
            prev = fx
        ends.append(end)
    return _bisect(f, ends[1], ends[0], target)


def _soft_limit(x, knee=None, limit=None):
    """x unchanged up to ``knee`` (rad), then easing to ``limit`` (tanh, same slope at the
    knee): continuous with a continuous derivative."""
    knee = math.radians(BAL_KNEE) if knee is None else knee
    limit = math.radians(BAL_MAX) if limit is None else limit
    a = abs(x)
    if a <= knee:
        return x
    return math.copysign(knee + (limit - knee) * math.tanh((a - knee) / (limit - knee)), x)


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


def solve(data, PI, PT, LL, TK, MT=0.0, TL=0.0, AR=0.0, PO=0.0, CB=0.0, light=False):
    """{segment: [R columns (9), t (3)]}: the rigid motion p -> R p + t of each segment
    ("legs", "pelvis", "sacrum" and the levels), plus the tables "tps" (inverse thin-
    plate spline, see tps_table), "chain" (polyrigid field, see chain_table), "grid"
    (forward displacement of the display grid) and "flow_cells", "flow_segs" (flow
    lines, see flow_tables); ``light``: motions only."""
    knobs = dict(PI=PI, PT=PT, LL=LL, TK=TK, MT=MT, TL=TL, AR=AR, PO=PO, CB=CB)
    out = _motions(data, **knobs)
    res = {k: [R[0][0], R[1][0], R[2][0], R[0][1], R[1][1], R[2][1], R[0][2], R[1][2], R[2][2], *t]
           for k, (R, t) in out.items()}
    if light:
        return res
    pairs = [(q["point"], _apply(out[q["segment"]], q["point"])) for q in data["tps_points"]]
    res["tps"] = tps_table(pairs)
    res["chain"] = chain_table(data, out)
    G = grid_points(data["grid"])
    D = _forward(data, out)(G) - G
    res["grid"] = np.column_stack([D, np.linalg.norm(D, axis=1)]).ravel().tolist()
    res["flow_cells"], res["flow_segs"] = flow_tables(data, knobs)
    return res


def _forward(data, out):
    """The forward soft-tissue map (rest points -> moved) for segment motions ``out``."""
    if data.get("field") == "polyrigid":
        T = chain_table(data, out)
        return lambda X: poly_forward(T, X)
    F = tps_table([(_apply(out[q["segment"]], q["point"]), q["point"]) for q in data["tps_points"]])
    return lambda X: tps_eval(F, X)  # forward: fitted from the rest points


def _motions(data, PI, PT, LL, TK, MT=0.0, TL=0.0, AR=0.0, PO=0.0, CB=0.0):
    """{segment: (R, t)} for the given parameter values (see solve)."""
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
            turn = cor[j] + (bal / len(frac) if j in frac else 0.0)
            T = _compose(T, _compose(_rot(d["sagittal"], X_AXIS, sag[j]), _rot(d["coronal"], Y_AXIS, turn)))
            out[d["upper"]] = T
        return out

    s1 = _apply(sacrum, data["s1_point"])

    def lean(bal):
        """Coronal angle from the S1 -> target line to the S1 -> top line (the target
        at the top's distance from S1, CB to the side, above S1): zero exactly when the
        balance is met, and growing steadily with the bend up to a half turn (the
        lateral offset itself turns back once the column leans past 90 deg)."""
        top = _apply(build(bal)[levels[-1]], data["frames"][levels[-1]]["center"])
        dx, dz = top[0] - s1[0], top[2] - s1[2]
        r = math.hypot(dx, dz)
        cx = max(-r, min(r, CB))  # the target point at the same distance from S1, above it
        cz = math.sqrt(max(r * r - cx * cx, 0.0))
        return math.atan2(dx * cz - dz * cx, dx * cx + dz * cz)  # signed angle target -> current

    moved_cor = any(abs(v) > 0 for v in cor) or abs(PO) > 0 or abs(alpha) > 0 or abs(beta) > 0 or any(sag)
    bal = 0.0
    if frac and (moved_cor or abs(CB - meas["CB"]) > 1e-9):
        bal = _soft_limit(_branch_solve(lean, 0.0, math.radians(BAL_SEARCH)))
    out = build(bal)
    for i, lv in enumerate(levels):  # axial rotation about the vertebra's own axis
        f = data["frames"][lv]
        out[lv] = _compose(out[lv], _rot(f["center"], f["axes"][2], axial[i]))
    out.update(legs=legs, pelvis=pelvis, sacrum=sacrum)
    return out


# ---------------------------------------------------------------- flow lines

def flow_tables(data, knobs):
    """Flow lines: the path of each seed point while the parameters go in FLOW_STEPS
    equal steps from the subject's own values to ``knobs`` (AR held), through the
    soft-tissue field. Segments (displacement so far at both ends) are binned into a
    grid of cells for the shader (see flow_eval):
      cells: per cell (offset, count, empty layers around it (see _empty_layers), 0), x fastest;
      segs: per entry two texels (a, |a - seed|), (b, |b - seed|), FLOW_ROW texels a row.
    """
    f = data["flow"]
    S = np.asarray(f["seeds"], float)
    start = {k: data["base"].get(k, knobs[k]) for k in ("PI", "PT", "LL", "TK", "MT", "TL", "CB")}
    start["PO"] = 0.0
    P = [S]
    for k in range(1, FLOW_STEPS + 1):
        s = k / FLOW_STEPS
        v = {key: start[key] + s * (knobs[key] - start[key]) for key in start}
        P.append(_forward(data, _motions(data, **v, AR=knobs["AR"]))(S))
    P = np.array(P)  # (steps + 1, seeds, 3)
    lo, cs, n = np.asarray(f["lo"], float), f["cell"], np.asarray(f["n"])
    cells = [[] for _ in range(int(np.prod(n)))]
    segs = []
    grow = FLOW_MARGIN + FLOW_RMAX
    for i in range(len(S)):
        if np.linalg.norm(P[-1, i] - S[i]) < 0.5:
            continue  # (nearly) still: no line
        for k in range(FLOW_STEPS):
            a, b = P[k, i], P[k + 1, i]
            c0 = np.clip(np.floor((np.minimum(a, b) - grow - lo) / cs).astype(int), 0, n - 1)
            c1 = np.clip(np.floor((np.maximum(a, b) + grow - lo) / cs).astype(int), 0, n - 1)
            e = len(segs)
            segs.append((a, np.linalg.norm(a - S[i]), b, np.linalg.norm(b - S[i])))
            for z in range(c0[2], c1[2] + 1):
                for y in range(c0[1], c1[1] + 1):
                    for x in range(c0[0], c1[0] + 1):
                        cells[x + n[0] * (y + n[1] * z)].append(e)
    empty = _empty_layers(np.array([len(c) > 0 for c in cells]).reshape(n[2], n[1], n[0]))
    table, entries = [], []
    for lst, k in zip(cells, empty.ravel()):
        table += [float(len(entries) // 8), float(len(lst)), float(k), 0.0]  # offset: entry index
        for e in lst:
            a, va, b, vb = segs[e]
            entries += [*map(float, a), float(va), *map(float, b), float(vb)]
    pad = (-len(entries)) % (4 * FLOW_ROW)
    return table, entries + [0.0] * (pad or 4 * FLOW_ROW)


def _empty_layers(full):
    """Per cell (array z, y, x): how many layers of cells around it hold no line, beyond
    its direct neighbours (Chebyshev distance to the nearest full cell - 1, from 0 to
    FLOW_SKIP); lets a ray cross empty space in long steps."""
    near = _dilate(full)  # full cells and their neighbours: 0
    out = np.zeros(full.shape, int)
    for _ in range(FLOW_SKIP):
        out += ~near
        near = _dilate(near)
    return out


def _dilate(m):
    """Boolean dilation by one cell in all 26 directions."""
    out = m.copy()
    for axis in range(3):
        a = out.copy()
        sl = [slice(None)] * 3
        sl[axis] = slice(1, None)
        sm = [slice(None)] * 3
        sm[axis] = slice(None, -1)
        a[tuple(sl)] |= out[tuple(sm)]
        a[tuple(sm)] |= out[tuple(sl)]
        out = a
    return out


def flow_eval(cells, segs, f, P, r):
    """(distance, displacement so far) of the flow lines at points P: the shader's
    flowLines (only the segments of a point's cell; others are beyond the cell
    boundary + FLOW_MARGIN)."""
    C, E = np.asarray(cells, float).reshape(-1, 4), np.asarray(segs, float).reshape(-1, 2, 4)
    lo, cs, n = np.asarray(f["lo"], float), f["cell"], np.asarray(f["n"])
    hi = lo + cs * n
    out = np.zeros((len(P), 2))
    for j, p in enumerate(P):
        c = np.floor((p - lo) / cs).astype(int)
        if np.any(c < 0) or np.any(c >= n):
            q = np.abs(p - 0.5 * (lo + hi)) - 0.5 * (hi - lo)
            out[j] = (np.linalg.norm(np.maximum(q, 0)) + min(q.max(), 0) + FLOW_MARGIN, 0.0)
            continue
        clo = lo + cs * c
        off, cnt, k = C[c[0] + n[0] * (c[1] + n[1] * c[2]), :3].astype(int)
        best = [min((p - clo).min(), (clo + cs - p).min()) + k * cs + FLOW_MARGIN, 0.0]
        for A, B in E[off:off + cnt]:
            pa, ba = p - A[:3], B[:3] - A[:3]
            h = min(max(pa @ ba / max(ba @ ba, 1e-12), 0.0), 1.0)
            d = np.linalg.norm(pa - ba * h) - r
            if d < best[0]:
                best = [d, A[3] + (B[3] - A[3]) * h]
        out[j] = best
    return out


# ---------------------------------------------------------------- polyrigid field

def _cross(a, b):
    return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]


def _log(T):
    """Screw coordinates (w, u) of a rigid motion: exp of [[skew(w), u], [0, 0]] is T."""
    R, t = T
    c = max(-1.0, min(1.0, (R[0][0] + R[1][1] + R[2][2] - 1.0) / 2.0))
    th = math.acos(c)
    if th < 1e-9:
        return [0.0, 0.0, 0.0], list(t)
    s = math.sin(th)
    w = [th / (2 * s) * v for v in (R[2][1] - R[1][2], R[0][2] - R[2][0], R[1][0] - R[0][1])]
    k = (1.0 - th * s / (2.0 * (1.0 - c))) / (th * th)
    wt = _cross(w, t)
    wwt = _cross(w, wt)
    return w, [t[i] - 0.5 * wt[i] + k * wwt[i] for i in range(3)]


def screw(P, w, u):
    """exp of the screw (w, u) applied to points P (n, 3) (Rodrigues; V u for the translation)."""
    w, u = np.asarray(w, float), np.asarray(u, float)
    phi = float(np.linalg.norm(w))
    if phi < 1e-9:
        return P + u
    k = w / phi
    c, s = math.cos(phi), math.sin(phi)
    R = P * c + np.cross(k, P) * s + np.outer(P @ k, k) * (1 - c)
    wu = np.cross(w, u)
    return R + u + (1 - c) / phi ** 2 * wu + (phi - s) / phi ** 3 * np.cross(w, wu)


def chain_table(data, out):
    """Polyrigid table, one row of 7 RGBA texels per chain segment (sacrum, L5 ... T1):
    0-2 the motion (columns of R, translation in w), 3-4 the screw (w, u) to the next
    segment, 5 the inferior and 6 the superior endplate plane (upward normal, offset;
    a zero normal for an absent plane)."""
    segs = ["sacrum"] + data["levels"]
    rows = []
    for i, s in enumerate(segs):
        R, t = out[s]
        if i + 1 < len(segs):
            inv = ([[R[j][i2] for j in range(3)] for i2 in range(3)], None)
            Rt = inv[0]
            tt = [-v for v in _mv(Rt, t)]
            w, u = _log(_compose((Rt, tt), out[segs[i + 1]]))
        else:
            w, u = [0.0] * 3, [0.0] * 3
        row = [[R[0][0], R[1][0], R[2][0], t[0]], [R[0][1], R[1][1], R[2][1], t[1]],
               [R[0][2], R[1][2], R[2][2], t[2]], [*w, 0.0], [*u, 0.0]]
        for side in ("inf", "sup"):
            pl = data["planes"][s].get(side)
            row.append([0.0, 0.0, 0.0, -1.0] if pl is None else [*pl[1], -float(np.dot(pl[0], pl[1]))])
        rows += row
    return [float(v) for r in rows for v in r]


def _poly_parts(table):
    T = np.asarray(table, float).reshape(-1, 7, 4)
    return T[:, :3, :3].transpose(0, 2, 1), T[:, :3, 3], T[:, 3, :3], T[:, 4, :3], T[:, 5], T[:, 6]


def _poly_region(table, X):
    """Per point: segment index r and blend lam (0 inside a body or outside the chain)."""
    R, t, w, u, inf, sup = _poly_parts(table)
    n = len(R)
    S = lambda pl: X @ pl[:, :3].T + pl[:, 3]
    si, ss = S(inf), S(sup)
    cnt = (si[:, 1:] >= 0).sum(1) + (ss >= 0).sum(1)
    j = cnt - 1
    r = np.where(j < 0, 0, np.where(j % 2 == 1, (j + 1) // 2, j // 2))
    blend = (j >= 0) & (j % 2 == 0) & (r < n - 1)
    rows = np.arange(len(X))
    a = ss[rows, r]
    b = si[rows, np.minimum(r + 1, n - 1)]
    lam = np.where(blend, np.clip(a / np.where(np.abs(a - b) > 1e-12, a - b, 1.0), 0.0, 1.0), 0.0)
    return r, lam


def poly_forward(table, X):
    """The polyrigid motion of rest points X (n, 3)."""
    R, t, w, u, _, _ = _poly_parts(table)
    r, lam = _poly_region(table, X)
    out = np.empty_like(X)
    for i in range(len(X)):
        y = screw(X[i:i + 1], lam[i] * w[r[i]], lam[i] * u[r[i]])[0]
        out[i] = R[r[i]] @ y + t[r[i]]
    return out


def poly_inverse(table, P):
    """Rest points of world points P (n, 3): fixed-point iteration as in the shader."""
    R, t, w, u, _, _ = _poly_parts(table)
    X = np.array(P, float)
    for k in range(POLY_ITERS):
        r, lam = _poly_region(table, X)
        G = np.array([screw((R[r[i]].T @ (P[i] - t[r[i]]))[None], -lam[i] * w[r[i]], -lam[i] * u[r[i]])[0]
                      for i in range(len(X))])
        X = G if k < POLY_RELAX else X + 0.5 * (G - X)
    return X


def grid_points(g):
    """Rest points of the field display grid, x fastest, then y, then z."""
    nx, ny, nz = g["n"]
    k, j, i = np.meshgrid(np.arange(nz), np.arange(ny), np.arange(nx), indexing="ij")
    return np.asarray(g["lo"]) + g["step"] * np.column_stack([i.ravel(), j.ravel(), k.ravel()])


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


solve.light = True  # takes light=True (for bounds)
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
