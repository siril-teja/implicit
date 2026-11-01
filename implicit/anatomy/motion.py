"""Physiological range of motion of each spinal level and the coupling of its motions.

Limits (``ROM``): total range of motion per level, in degrees, for flexion plus
extension (FE), left plus right lateral bending (LB) and left plus right axial
rotation (AR): medians of
  Wilke HJ, Kienle A, Werner K, Liebsch C. Range of motion and neutral zone of all
    human spinal motion segments: a database of in vitro measurements. JOR Spine
    2025;8(1):e70052. doi:10.1002/jsp2.70052
(supplementary table "Evaluated data"; pure moments of 5 Nm on thoracic and 7.5 Nm on
lumbar segments, T12-L1 tested as lumbar). For T1-T2 ... T11-T12 the values are those
measured with the rib cage intact, which the paper reports to agree with in vivo
measurements (Morita 2014, Fujimori 2012/2013); lumbar values agree with in vivo ones
(Pearcy 1984). A disc may turn from its rest orientation (the template's) by at most
half of each range either way: the split of a range equally into its two directions
is an assumption (standing is not the middle of the range in vivo).

Coupling (``COUPLING``): axial rotation per degree of lateral bending, in the
direction of the bend (ipsilateral), at the thoracic levels T1-T2 ... T12-L1:
  Fujimori T, Iwasaki M, Nagamoto Y, et al. Kinematics of the thoracic spine in
    trunk lateral bending: in vivo three-dimensional analysis. Spine J
    2014;14(9):1991-9. doi:10.1016/j.spinee.2013.11.054
T1-L1 bent 15.6 deg on average with 6.2 deg of coupled axial rotation in the same
direction (supine, 15 men): 6.2 / 15.6 = 0.40, taken for every thoracic level (the
abstract gives the coupled rotation for T1-L1 only; per segment it was "generally"
in the direction of the bend, most variable at T2-T3 ... T5-T6). Lumbar levels: no coupling. In vivo
lumbar coupling of lateral bending and axial rotation is small and changes direction
between levels (Shin 2013, doi:10.1007/s00586-013-2777-6, during axial rotation:
contralateral bending at L2-L3 and L3-L4, ipsilateral at L4-L5 and L5-S1; Fujii 2007,
doi:10.1007/s00586-007-0373-3: contralateral at L1-L2 ... L4-L5), with no
coefficients reported for lateral bending; taken as 0.
"""

from __future__ import annotations

# level: (FE, LB, AR) total range, deg (Wilke et al. 2025, medians; thoracic with rib cage)
ROM = {
    "T1-T2": (6.2, 4.8, 5.0), "T2-T3": (3.7, 4.0, 3.2), "T3-T4": (2.2, 3.3, 1.6),
    "T4-T5": (1.3, 2.0, 1.4), "T5-T6": (1.3, 1.9, 1.9), "T6-T7": (1.3, 1.6, 2.1),
    "T7-T8": (1.5, 2.0, 2.5), "T8-T9": (1.6, 1.7, 3.3), "T9-T10": (1.8, 2.6, 3.9),
    "T10-T11": (2.3, 2.5, 3.7), "T11-T12": (2.4, 3.2, 2.6), "T12-L1": (5.6, 4.8, 2.1),
    "L1-L2": (6.4, 7.3, 2.4), "L2-L3": (8.4, 9.5, 4.0), "L3-L4": (7.8, 9.1, 3.0),
    "L4-L5": (10.3, 9.5, 4.6), "L5-S1": (10.8, 6.4, 4.0),
}
THORACIC_COUPLING = 6.2 / 15.6   # axial rotation per lateral bending, ipsilateral (Fujimori 2014)
COUPLING = {lv: (THORACIC_COUPLING if lv.startswith("T") else 0.0) for lv in ROM}


def fill(total, weights, caps):
    """Spread ``total`` over the levels in proportion to ``weights`` (>= 0), none beyond
    its cap (>= 0) either way: levels that reach their cap keep it and the rest is
    spread over the others in the same proportion (water-filling). Returns (shares,
    shortfall): the part of ``total`` no level could take (0 when it fits).
    Continuous in ``total``; plain Python, as viewer/solvers.js fill."""
    a, sgn = abs(total), (1.0 if total >= 0 else -1.0)
    act = [j for j, w in enumerate(weights) if w > 0]
    out = [0.0] * len(weights)
    if not act:
        return out, total
    room = sum(caps[j] for j in act)
    if a >= room:
        for j in act:
            out[j] = sgn * caps[j]
        return out, sgn * (a - room)
    order = sorted(act, key=lambda j: (caps[j] / weights[j], j))  # first to reach their cap first
    rest, wsum, k = a, sum(weights[j] for j in act), 0
    while k < len(order):
        j = order[k]
        if rest / wsum * weights[j] <= caps[j]:
            break
        rest -= caps[j]
        wsum -= weights[j]
        k += 1
    s = rest / wsum
    for j in act:
        out[j] = sgn * min(s * weights[j], caps[j])
    return out, 0.0
