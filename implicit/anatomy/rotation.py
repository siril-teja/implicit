"""Physiological rotation centres of the intervertebral joints.

Each disc turns about its instantaneous axis of rotation (IAR) instead of the disc
centre; the axes stay the scene's (sagittal turns about the left-right axis, coronal
ones about the antero-posterior axis), only the point they pass through changes. The
point is placed relative to the lower vertebra of the disc: its superior endplate
centre p, upward normal n and anterior axis a (perpendicular to n), its body depth
D and height H (implicit.anatomy.landmarks).

Flexion-extension (sagittal turns)
  lumbar discs L1-L2 ... L5-S1:  p - 1.5 n - D/6 a
      Pearcy & Bogduk 1988 (radiographs of ten men, L1-L2 to L5-S1): the mean IAR of
      flexion-extension lies just below the superior endplate of the lower vertebra,
      just anterior to the posterior third of the disc. "Just below" is taken as
      1.5 mm and "just anterior to the posterior third" as the posterior third's
      boundary, D/6 behind the endplate centre (assumptions; the paper gives the
      positions graphically).
  thoracic discs T1-T2 ... T12-L1:  p - H/4 n
      Qiu et al. 2003, 2004 (finite element model of T10-T11, validated against
      experiments): the IARs of flexion and extension lie on the antero-posterior
      centre of the lower vertebra, between its superior endplate and its geometric
      centre; taken halfway between the two.
Lateral bending (coronal turns)
  lumbar:  the disc centre (midpoint of the two endplate centres)
      Dimnet et al. 1978 (radiographs in vivo, T11-L5): the centre of rotation of
      lateral flexion lies at the centre of the disc.
  thoracic:  halfway between the disc centre and the lower vertebra's body centre
      Qiu et al. 2003: the IARs of lateral bending run from the mid-height of the disc
      towards the geometric centre of the lower vertebra.
For the sacrum (lower vertebra of L5-S1) D and H are taken from L5. Axial rotation
(the AR knob) stays a turn of each vertebra about its own axis.

References
  Pearcy MJ, Bogduk N. Instantaneous axes of rotation of the lumbar intervertebral
    joints. Spine 1988;13(9):1033-41. doi:10.1097/00007632-198809000-00011
  Qiu TX, Teo EC, Lee KK, Ng HW, Yang K. Validation of T10-T11 finite element model
    and determination of instantaneous axes of rotations in three anatomical planes.
    Spine 2003;28(24):2694-9. doi:10.1097/01.BRS.0000101424.24198.AC
  Qiu TX, Teo EC, Lee KK, Ng HW, Yang K. Kinematics of the thoracic T10-T11 motion
    segment: locus of instantaneous axes of rotation in flexion and extension.
    J Spinal Disord Tech 2004;17(2):140-6. doi:10.1097/00024720-200404000-00011
  Dimnet J, Fischer LP, Gonon G, Carret JP. Radiographic studies of lateral flexion
    in the lumbar spine. J Biomech 1978;11(3):143-50.
    doi:10.1016/0021-9290(78)90006-4
The Pearcy & Bogduk abstract does not state the location; the description above is
the one quoted from the paper in reviews. Dimnet et al. has no abstract; its result
is taken from the literature table (Table 1) of
  Allais R, Capart A, Da Silva A, Boiron O. Biomechanical consequences of the
    intervertebral disc centre of rotation kinematics during lateral bending and
    axial rotation. Sci Rep 2023;13:3172. doi:10.1038/s41598-023-29551-7
which also lists other lumbar lateral-bending centres (contralateral, cranial;
Rolander, ex vivo): the reported centres are scattered.
"""

from __future__ import annotations

import numpy as np

LUMBAR = {f"L{i}" for i in range(1, 6)}   # upper vertebrae of the lumbar discs L1-L2 ... L5-S1
BELOW_ENDPLATE = 1.5                       # mm, lumbar sagittal IAR below the lower endplate


def _unit(v):
    v = np.asarray(v, float)
    return v / np.linalg.norm(v)


def centres(upper, lower, lower_name, size):
    """{"sagittal": point, "coronal": point} of the disc between landmark entries
    ``upper`` and ``lower`` (``lower_name``: "S1" or a vertebra); ``size``: (depth,
    height) of the lower body (L5's for S1)."""
    p = np.asarray(lower["sup"]["point"], float)
    n = _unit(lower["sup"]["normal"])
    a = np.asarray(lower["R"], float)[:, 1]
    a = _unit(a - a.dot(n) * n)
    D, H = size
    disc = 0.5 * (p + np.asarray(upper["inf"]["point"], float))
    if upper["level"] in LUMBAR:
        sag, cor = p - BELOW_ENDPLATE * n - D / 6.0 * a, disc
    else:
        body = np.asarray(lower["body_center"], float) if lower_name != "S1" else p - 0.5 * H * n
        sag, cor = p - 0.25 * H * n, 0.5 * (disc + body)
    return {"sagittal": sag, "coronal": cor}


def add_centres(marks):
    """Store each disc's centres in its upper vertebra's entry ("rotation"), for every
    vertebra with both endplates whose lower neighbour (or S1) has a superior one."""
    from .convert import VERTEBRAE
    order = [lv for lv in VERTEBRAE if lv in marks] + (["S1"] if "S1" in marks else [])
    for up, lo in zip(order, order[1:]):
        u, l = marks[up], marks[lo]
        if "inf" not in u or "sup" not in l:
            continue
        src = marks.get("L5") if lo == "S1" else l
        if src is None or "depth" not in src:
            continue
        u["rotation"] = centres(u, l, lo, (src["depth"], src["height"]))
    return marks
