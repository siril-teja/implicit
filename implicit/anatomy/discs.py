"""Disc space heights: are the gaps between the vertebral bodies physiological?

    python -m implicit.anatomy.discs examples/average_spine.json [--set spp_LL=80] [--sweep]

Each endplate has three points (stored in landmarks.json under "corners" by
implicit.anatomy.landmarks): its anterior and posterior corners and its centre, on the
bone surface in the vertebra's mid-sagittal line (``endplate_corners``). The disc
space between two vertebrae has

  A  anterior height: the two anterior corners' separation along the disc axis
  M  middle height: the same for the two endplate centres
  P  posterior height: the same for the two posterior corners
  L, R  left and right heights: the same for the lateral ends of the endplates
        (mid-coronal line), for lateral bending

with the disc axis the bisector of the two endplate normals (heights perpendicular
to the endplates, as on radiographs and CT; an antero-posterior offset of the two
vertebrae does not count). This is the bony disc space. Endplates whose plane fit is
worse than RMS_MAX get no corners, and their discs are not measured. The corners move rigidly
with their vertebra, so the heights follow every posture; viewer/discs.js is the same
computation.

Posture rules, relative to the subject's own disc at rest, i.e. with the vertebrae
where the source images have them (``posture_flags``):
  - the middle height changes by at most MEAN_CHANGE (10 %) of its rest value: in
    vivo the mean disc height of healthy discs changes by at most 0.5 mm (6 %) from
    maximum extension to 45 deg flexion (Xi et al. 2025, dual fluoroscopy, L1-S1);
  - the anterior, posterior, left and right heights change by at most LOCAL_STRAIN
    (40 %) of their rest values: local axial strains of 20-40 % of the disc thickness were measured
    at the extremes of motion (Xi et al. 2025).
The thresholds are rounded up from the measured values (assumption), so a flag
means "beyond what was measured in healthy discs". Xi et al. measured lumbar discs;
the same limits are used for the thoracic discs (assumption).

Thickness over the whole disc (``thickness``): the A/M/P/L/R heights sample five
points; between them a disc can be thinner (a wedged or curved endplate). The
smallest distance from any surface point of either endplate (``endplate_indices``)
to the other vertebra's bone is its smallest thickness, live like the heights. It
must not fall below the disc's floor (``thickness_floor``): the population's lower
bound of its anterior and posterior heights (mean - 2 SD, below), at least
MIN_THICKNESS (1.5 mm, the thinnest such bound of any level: Kunkel T4-T5 anterior),
at rest (template) and in every posture.

Rest heights against population values (``norm_flags``), mean +- 2 SD:
  lumbar  L1-L2 ... L5-S1: Shin et al. 2024 (CT, 350 women aged 18-35, Table 4,
          female values adjusted for weight, height, BMI and ethnicity);
  thoracic T1-T2 ... T11-T12: Kunkel et al. 2011 (direct measurement of 6 discs per
          level, donors 57 +- 11 years, Table 1); T12-L1, not measured there, uses
          the T11-T12 values (assumption).
The template's own disc spaces are what the posture moves; population values are
reported, not enforced (shape: roadmap phase 3).

References
  Xi X, Jia L, Yu W, et al. Fatigue-induced biomechanical decoupling at L4-S1 discs:
    mechanism of disc degeneration in chronic low back pain. BMC Musculoskelet
    Disord 2025;26:1001. doi:10.1186/s12891-025-09238-y
  Shin D, Vyhmeister E, Im D, et al. Lumbar disc space height in relation to neural
    foraminal dimensions and patient characteristics: a morphometric analysis from
    L1-S1 using computed tomography. Brain Spine 2024;5:104162.
    doi:10.1016/j.bas.2024.104162
  Kunkel ME, Herkommer A, Reinehr M, Bockers TM, Wilke HJ. Morphometric analysis of
    the relationships between intervertebral disc and vertebral body heights: an
    anatomical and radiographic study of the human thoracic spine. J Anat
    2011;219(3):375-87. doi:10.1111/j.1469-7580.2011.01397.x
"""

from __future__ import annotations

import argparse
import itertools
import json

import numpy as np

from .convert import VERTEBRAE

MEAN_CHANGE = 0.10    # largest relative change of the middle height (posture)
LOCAL_STRAIN = 0.40   # largest relative change of the anterior / posterior heights
CORNER_SLOPE = 1.0    # endplate ends where the surface turns steeper than 45 deg
COLUMN_STEP = 0.25    # mm, spacing of the sample columns along the endplate
RMS_MAX = 1.5         # mm: endplates fitted worse than this get no corners (T1 of VIVA+ 50F)
MIN_REST = 0.5        # mm: a rest height below this is not judged by relative change
MIN_THICKNESS = 1.5   # mm: no disc thinner anywhere (the thinnest population bound, Kunkel T4-T5 A)

# (A, M, P) population heights per disc: (mean, SD) in mm (see the module docstring).
NORMS = {
    "T1-T2": ((4.5, 0.77), (5.7, 0.66), (4.3, 0.62)),
    "T2-T3": ((3.4, 0.97), (5.8, 0.79), (3.5, 0.99)),
    "T3-T4": ((3.3, 0.30), (4.4, 0.80), (3.2, 0.49)),
    "T4-T5": ((3.0, 0.74), (5.0, 0.61), (3.3, 0.45)),
    "T5-T6": ((3.5, 0.37), (4.7, 0.91), (3.6, 0.47)),
    "T6-T7": ((4.1, 0.35), (5.5, 0.45), (4.1, 0.65)),
    "T7-T8": ((4.2, 0.97), (5.3, 0.58), (3.6, 0.90)),
    "T8-T9": ((5.6, 1.17), (6.1, 0.89), (5.0, 0.91)),
    "T9-T10": ((5.4, 1.74), (5.5, 0.90), (4.2, 1.10)),
    "T10-T11": ((7.2, 1.21), (6.7, 1.18), (5.8, 1.00)),
    "T11-T12": ((6.0, 1.14), (6.5, 1.15), (4.8, 1.26)),
    "T12-L1": ((6.0, 1.14), (6.5, 1.15), (4.8, 1.26)),
    "L1-L2": ((4.89, 1.53), (6.36, 1.24), (2.98, 1.10)),
    "L2-L3": ((5.83, 1.71), (7.61, 1.48), (3.82, 1.21)),
    "L3-L4": ((7.12, 1.76), (8.53, 1.48), (4.44, 1.38)),
    "L4-L5": ((8.69, 2.02), (8.99, 1.80), (4.73, 1.49)),
    "L5-S1": ((10.38, 2.64), (8.07, 1.86), (3.77, 1.36)),
}
SOURCES = {**{k: "Kunkel 2011" for k in NORMS if k.startswith("T")}, **{k: "Shin 2024 (F)" for k in NORMS if k.startswith("L")}}


# ---------------------------------------------------------------- detection

def endplate_corners(field, entry, ep):
    """{"anterior", "posterior", "left", "right", "centre", "normal"} of endplate ``ep``
    of a landmark entry (points; the endplate plane's outward normal).

    Columns along the vertebra's anterior (then right) axis through the endplate
    centre, each marched from 10 mm outside the endplate plane into the bone; the
    endplate is the run of columns around the centre where the surface is no steeper
    than 45 deg (and within -8 / +5 mm of the plane), its ends are the corners.
    """
    if entry[ep].get("rms", 0.0) > RMS_MAX:
        raise RuntimeError(f"endplate {ep}: plane fit rms {entry[ep]['rms']:.2f} mm > {RMS_MAX} mm, "
                           "no corners (disc heights not measured)")
    p = np.asarray(entry[ep]["point"], float)
    n = np.asarray(entry[ep]["normal"], float)  # outward
    R = np.asarray(entry["R"], float)
    out = {"centre": None, "normal": n}
    for axis, (plus, minus) in ((1, ("anterior", "posterior")), (0, ("right", "left"))):
        u = R[:, axis] - R[:, axis].dot(n) * n
        hi, lo, c = _ends(field, p, n, u / np.linalg.norm(u), ep)
        out[plus], out[minus] = hi, lo
        if axis == 1:
            out["centre"] = c
    return out


def _ends(field, p, n, u, ep):
    """Both ends of the endplate along direction u, and the surface point at its centre."""
    v = np.arange(-40.0, 40.0 + 1e-9, COLUMN_STEP)
    s = np.arange(10.0, -15.0, -0.05)
    P = p + v[:, None, None] * u + s[None, :, None] * n
    inside = field(P.reshape(-1, 3)).reshape(len(v), len(s)) < 0
    hit = inside.any(1)
    h = np.where(hit, s[np.argmax(inside, 1)], -99.0)
    slope = np.abs(np.gradient(h, COLUMN_STEP))
    ok = hit & (h > -8.0) & (h < 5.0) & (slope < CORNER_SLOPE)
    if not ok.any():
        raise RuntimeError(f"endplate {ep}: no surface found")
    c = np.flatnonzero(ok)
    i0 = c[np.argmin(np.abs(v[c]))]
    lo = hi = i0
    while lo > 0 and ok[lo - 1]:
        lo -= 1
    while hi < len(v) - 1 and ok[hi + 1]:
        hi += 1
    at = lambda i: p + v[i] * u + h[i] * n
    return at(hi), at(lo), p + h[np.argmin(np.abs(v))] * n


# ---------------------------------------------------------------- heights

def pairs(names):
    """Neighbouring vertebrae among ``names`` (upper, lower), the lowest with the sacrum."""
    names = set(names)
    levels = [lv for lv in VERTEBRAE if lv in names]
    out = list(zip(levels, levels[1:]))
    if levels and "sacrum" in names:
        out.append((levels[-1], "sacrum"))
    return out


def label(upper, lower):
    return f"{upper}-{'S1' if lower == 'sacrum' else lower}"


def _move(M, p):
    R, t = M
    return np.asarray(R, float) @ np.asarray(p, float) + np.asarray(t, float)


KEYS = ("anterior", "centre", "posterior", "left", "right")  # heights A, M, P, L, R


def heights(bodies):
    """[(upper, lower, A, M, P, L, R)] for the neighbouring vertebrae of ``bodies``:
    {name: (corners, (R, t))}, corners {"sup"/"inf": {"anterior", "posterior",
    "centre"}} in the bone's own frame, (R, t) its current rigid motion."""
    out = []
    for up, lo in pairs([k for k, (c, _) in bodies.items() if c]):
        (cu, Mu), (cl, Ml) = bodies[up], bodies[lo]
        if "inf" not in cu or "sup" not in cl:
            continue
        # Disc axis: bisector of the two endplate normals (outward: lower up, upper down).
        w = np.asarray(Ml[0], float) @ np.asarray(cl["sup"]["normal"], float) \
            - np.asarray(Mu[0], float) @ np.asarray(cu["inf"]["normal"], float)
        w /= np.linalg.norm(w)
        d = [float((_move(Mu, cu["inf"][k]) - _move(Ml, cl["sup"][k])) @ w) for k in KEYS]
        out.append((up, lo, *d))
    return out


IDENTITY = (np.eye(3), np.zeros(3))


def rest_heights(bodies):
    """Heights of the unmoved vertebrae (as in the source images): the reference of
    the posture rules."""
    return heights({k: (c, IDENTITY) for k, (c, _) in bodies.items()})


# ---------------------------------------------------------------- thickness map

def endplate_indices(samples, corners):
    """Indices of a bone's surface samples (implicit.anatomy.clearance.surface_samples,
    bone frame) on one endplate: inside the ellipse through its four corners (in the
    endplate plane) and from 4 mm below to 3 mm above the plane (rim included)."""
    P = np.asarray(samples, float).reshape(-1, 3)
    c = {k: np.asarray(v, float) for k, v in corners.items()}
    a, r = c["anterior"] - c["posterior"], c["right"] - c["left"]
    ha, hr = 0.5 * np.linalg.norm(a), 0.5 * np.linalg.norm(r)
    u = (P - 0.5 * (c["anterior"] + c["posterior"])) @ (a / (2 * ha)) / ha
    v = (P - 0.5 * (c["right"] + c["left"])) @ (r / (2 * hr)) / hr
    w = (P - c["centre"]) @ c["normal"]
    return np.flatnonzero((u * u + v * v <= 1.0) & (w > -4.0) & (w < 3.0))


def thickness(bodies):
    """{(upper, lower): smallest disc thickness (mm)} over the whole disc: the distance
    from every surface point of each endplate to the other vertebra's bone (its field),
    the smaller of both ways, on the surface samples (about 2 mm apart, so the exact
    thinnest spot can be missed by a few tenths of a mm). ``bodies``: {name: (grid,
    samples, corners, (R, t))}.
    viewer/discs.js is the same computation on the same points and grid values."""
    out = {}
    for up, lo in pairs([k for k, b in bodies.items() if b[2]]):
        gu, su, cu, Mu = bodies[up]
        gl, sl, cl, Ml = bodies[lo]
        if "inf" not in cu or "sup" not in cl:
            continue
        best = np.inf
        for (gs, ss, cs, Ms, ep), (go, Mo) in (((gl, sl, cl, Ml, "sup"), (gu, Mu)), ((gu, su, cu, Mu, "inf"), (gl, Ml))):
            P = np.asarray(ss, float).reshape(-1, 3)[endplate_indices(ss, cs[ep])]
            if not len(P):
                continue
            Rs, ts = (np.asarray(x, float) for x in Ms)
            Ro, to = (np.asarray(x, float) for x in Mo)
            W = P @ Rs.T + ts
            best = min(best, float(go.eval((W - to) @ Ro).min()))
        out[(up, lo)] = best
    return out


def thickness_floor(up, lo):
    """Smallest plausible local disc thickness (mm): the population's lower bound of the
    anterior and posterior heights (mean - 2 SD), not below MIN_THICKNESS."""
    ref = NORMS.get(label(up, lo))
    if ref is None:
        return MIN_THICKNESS
    (ma, sa), _, (mp, sp) = ref
    return max(MIN_THICKNESS, min(ma - 2 * sa, mp - 2 * sp))


def thin_flags(thick):
    """{(upper, lower): reason} where the disc is thinner somewhere than its floor."""
    return {k: f"thinnest {x:.1f} mm < {thickness_floor(*k):.1f} mm" for k, x in thick.items()
            if x < thickness_floor(*k)}


def posture_flags(now, rest, thick=None):
    """{(upper, lower): reasons} where the posture deforms a disc beyond the rules
    (``thick``: thickness() now, adds the floor rule)."""
    out = {k: [v] for k, v in thin_flags(thick).items()} if thick else {}
    rest = {(u, l): h for u, l, *h in rest}
    for u, l, *h in now:
        (a, m, p, le, ri), (a0, m0, p0, le0, ri0) = h, rest[(u, l)]
        why = []
        if m0 > MIN_REST and abs(m - m0) > MEAN_CHANGE * m0:
            why.append(f"middle {100 * (m / m0 - 1):+.0f} %")
        for name, x, x0 in (("anterior", a, a0), ("posterior", p, p0), ("left", le, le0), ("right", ri, ri0)):
            if x0 > MIN_REST and abs(x - x0) > LOCAL_STRAIN * x0:
                why.append(f"{name} {100 * (x / x0 - 1):+.0f} %")
        if why:
            out[(u, l)] = out.get((u, l), []) + why
    return out


def norm_flags(hs):
    """{(upper, lower): reasons} where a height is outside the population mean +- 2 SD."""
    out = {}
    for u, l, *h in hs:
        ref = NORMS.get(label(u, l))
        if ref is None:
            continue
        why = [f"{k} {x:.1f} mm (norm {mu:.1f} +- {2 * sd:.1f})"
               for k, x, (mu, sd) in zip("AMP", h[:3], ref) if abs(x - mu) > 2 * sd]
        if why:
            out[(u, l)] = why
    return out


# ---------------------------------------------------------------- scenes

def scene_bodies(scene, grids=False):
    """{name: (corners, (R, t))} of the scene's moved bones (graph scenes); with
    ``grids``: {name: (grid, surface samples, corners, (R, t))} (for thickness())."""
    from .clearance import surface_samples
    out = {}
    for b in scene.bodies:
        rig = getattr(b, "rigid", None)
        if rig is None:
            continue
        g, c, M = rig[0], getattr(rig[0], "corners", None), rig[1]()
        if grids:
            if getattr(g, "_samples", None) is None:
                g._samples = surface_samples(g)
            out[b.name] = (g, g._samples, c, M)
        else:
            out[b.name] = (c, M)
    return out


def report(scene):
    """(heights now, heights at rest) of a graph scene at its current values."""
    b = scene_bodies(scene)
    return heights(b), rest_heights(b)


def report_thickness(scene):
    """(thickness now, thickness at rest) of a graph scene at its current values."""
    b = scene_bodies(scene, grids=True)
    return thickness(b), thickness({k: (g, s, c, IDENTITY) for k, (g, s, c, _) in b.items()})


def main(argv=None):
    from ..graph import load_graph
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("graph")
    ap.add_argument("--set", action="append", default=[], help="name=value, e.g. spp_LL=80")
    ap.add_argument("--sweep", action="store_true", help="posture flags over a grid of slider values")
    a = ap.parse_args(argv)
    scene = load_graph(a.graph)
    defaults = {p.name: p.value() for p in scene.params()}
    scene.set_params({k: float(v) for k, v in (s.split("=") for s in a.set)})
    now, rest = report(scene)
    thick, thick0 = report_thickness(scene)
    norms = norm_flags(rest)
    for k, v in thin_flags(thick0).items():
        norms[k] = norms.get(k, []) + [v]
    flags = posture_flags(now, rest, thick)
    print(f"{'disc':>8} {'A':>6} {'M':>6} {'P':>6} {'L':>6} {'R':>6} {'min':>6}   (rest A, M, P, L, R, min)")
    for (u, l, *h), (_, _, *h0) in zip(now, rest):
        k = (u, l)
        note = "; ".join(flags.get(k, []))
        print(f"{label(u, l):>8} " + " ".join(f"{x:6.1f}" for x in h) + f" {thick[k]:6.1f}   ("
              + ", ".join(f"{x:.1f}" for x in h0) + f", {thick0[k]:.1f})"
              f"{'  POSTURE: ' + note if note else ''}{'  vs norm: ' + '; '.join(norms[k]) if k in norms else ''}")
    if a.sweep:
        names = {p.name for p in scene.params()}
        grid = {k: v for k, v in {"spp_LL": [20, 52, 80], "spp_TK": [10, 30, 60], "spp_PT": [0, 17, 35],
                                  "spp_MT": [-40, 0, 40], "spp_TL": [-40, 0, 40]}.items() if k in names}
        counts, bad = {}, 0
        for combo in itertools.product(*grid.values()):
            scene.set_params({**defaults, **dict(zip(grid, combo))})
            f = posture_flags(report(scene)[0], rest, report_thickness(scene)[0])
            bad += bool(f)
            for k in f:
                counts[k] = counts.get(k, 0) + 1
        n = int(np.prod([len(v) for v in grid.values()]))
        print(f"\nsweep over {', '.join(grid)}: {bad} of {n} settings deform a disc beyond the rules")
        for k, c in sorted(counts.items(), key=lambda kv: -kv[1]):
            print(f"  {label(*k)}: {c}")
        print(json.dumps({"settings": n, "flagged": bad}))


if __name__ == "__main__":
    main()
