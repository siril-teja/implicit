"""Connected rib cage: junction landmarks and the motion of the sternum.

Ribs and the sternum are bone; the costal cartilages between them are far more
compliant (local modulus 8.7-12.6 MPa, Forman & Kent 2011; whole segments 22 +- 14 MPa
effective, Forman et al. 2010; bone is GPa). So the ribs stay rigid with their
vertebra (rib n with T n), the sternum moves rigidly, and the cartilages bend
between them (implicit.anatomy.alignment: the rib cage field).

Landmarks (``detect``, stored as "ribcage" in landmarks.json):
  costochondral junctions, per rib 1-10 and side: the rib's surface samples within
    CONTACT of its cartilage; JUNCTION_POINTS of them kept (the centroid's nearest
    and the farthest along each in-plane axis of the patch) and their centroid;
  sternocostal junctions, per side: the cartilage's surface samples within CONTACT
    of the sternum, thinned to one per STERNAL_CELL cube.
The template correction (implicit.anatomy.template, 3.) joins every cartilage to its
ribs first, so every rib 1-10 has a junction.

Sternum motion (``sternum_motion``): the rigid motion that best maps the rest
junction centroids of ribs 1-STERNAL_RIBS (both sides; the true ribs, whose
cartilages reach the sternum) onto where their ribs carry them, in the least-squares
sense (Horn 1987, unit quaternions: the largest eigenvector of a 4 x 4 symmetric
matrix, found by power iteration, POWER_STEPS steps from the identity). That the
sternum follows the mean motion of the true rib ends is an assumption: no in vivo
data on sternum motion with spinal posture were found (docs/LITERATURE.md).

References
  Forman JL, Kent RW. Modeling costal cartilage using local material properties with
    consideration for gross heterogeneities. J Biomech 2011;44(5):910-6.
    doi:10.1016/j.jbiomech.2010.11.034
  Forman JL, del Pozo de Dios E, Kent RW. A pseudo-elastic effective material property
    representation of the costal cartilage for use in finite element models of the
    whole human body. Traffic Inj Prev 2010;11(6):613-22.
    doi:10.1080/15389588.2010.517254
  Horn BKP. Closed-form solution of absolute orientation using unit quaternions.
    J Opt Soc Am A 1987;4(4):629-42. doi:10.1364/JOSAA.4.000629
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from .convert import load_bone

CONTACT = 0.5          # mm: a surface sample this close to the other part is in the junction
JUNCTION_POINTS = 5    # control points kept per costochondral junction
STERNAL_CELL = 8.0     # mm: sternocostal samples thinned to one per cube of this size
STERNAL_RIBS = 7       # ribs 1-7 (true ribs) place the sternum
RIBS = range(1, 11)    # ribs with a costal cartilage
POWER_STEPS = 60


def _surface(g):
    return g.origin + g.spacing * np.argwhere(np.abs(g.values) < 0.5 * g.spacing)


def _thin(P, cell):
    k = np.floor((P - P.min(0)) / cell).astype(np.int64)
    _, first = np.unique(k[:, 0] * 1000003 + k[:, 1] * 1009 + k[:, 2], return_index=True)
    return P[np.sort(first)]


def _junction_points(J):
    """Centroid and JUNCTION_POINTS samples spread over a junction patch."""
    c = J.mean(0)
    _, _, vt = np.linalg.svd(J - c, full_matrices=False)
    pick = [int(np.argmin(np.linalg.norm(J - c, axis=1)))]
    for ax in vt[:2]:
        t = (J - c) @ ax
        pick += [int(np.argmax(t)), int(np.argmin(t))]
    pick = list(dict.fromkeys(pick))[:JUNCTION_POINTS]
    return c, J[pick]


def detect(bone_dir, log=print):
    """Rib cage landmarks of ``bone_dir`` (see the module docstring), or None when it
    has no costal cartilage."""
    bone_dir = Path(bone_dir)
    out = {"junctions": {}, "sternal": {}}
    st = load_bone(bone_dir / "sternum.npz")[0] if (bone_dir / "sternum.npz").exists() else None
    for side in ("left", "right"):
        f = bone_dir / f"costal_cartilage_{side}.npz"
        if not f.exists():
            continue
        cc = load_bone(f)[0]
        for n in RIBS:
            rf = bone_dir / f"rib{n}_{side}.npz"
            if not rf.exists():
                continue
            P = _surface(load_bone(rf)[0])
            J = P[np.abs(cc.eval(P)) < CONTACT]
            if len(J) < 3:
                log(f"  rib{n}_{side}: no junction with the cartilage (gap {cc.eval(P).min():.1f} mm)")
                continue
            c, pts = _junction_points(J)
            out["junctions"][f"rib{n}_{side}"] = {"rib": n, "center": c.round(2).tolist(),
                                                  "points": pts.round(2).tolist(), "samples": int(len(J))}
        if st is not None:
            C = _surface(cc)
            S = C[np.abs(st.eval(C)) < CONTACT]
            if len(S):
                out["sternal"][side] = _thin(S, STERNAL_CELL).round(2).tolist()
    return out if out["junctions"] else None


# ---------------------------------------------------------------- sternum motion
# Plain Python (as viewer/solvers.js sternumMotion).

def horn(P, Q):
    """Rigid motion (R rows, t) minimising sum |R p + t - q|^2 (Horn 1987)."""
    n = len(P)
    cp = [sum(p[i] for p in P) / n for i in range(3)]
    cq = [sum(q[i] for q in Q) / n for i in range(3)]
    S = [[sum((p[i] - cp[i]) * (q[j] - cq[j]) for p, q in zip(P, Q)) for j in range(3)] for i in range(3)]
    (xx, xy, xz), (yx, yy, yz), (zx, zy, zz) = S
    N = [[xx + yy + zz, yz - zy, zx - xz, xy - yx],
         [yz - zy, xx - yy - zz, xy + yx, zx + xz],
         [zx - xz, xy + yx, -xx + yy - zz, yz + zy],
         [xy - yx, zx + xz, yz + zy, -xx - yy + zz]]
    shift = sum(math.sqrt(sum((p[i] - cp[i]) ** 2 for i in range(3)) * sum((q[i] - cq[i]) ** 2 for i in range(3)))
                for p, q in zip(P, Q)) + 1e-9  # >= |eigenvalues|: N + shift I is positive definite
    v = [1.0, 0.0, 0.0, 0.0]
    for _ in range(POWER_STEPS):
        w = [sum(N[i][j] * v[j] for j in range(4)) + shift * v[i] for i in range(4)]
        s = math.sqrt(sum(x * x for x in w))
        v = [x / s for x in w]
    a, b, c, d = v
    R = [[a * a + b * b - c * c - d * d, 2 * (b * c - a * d), 2 * (b * d + a * c)],
         [2 * (b * c + a * d), a * a - b * b + c * c - d * d, 2 * (c * d - a * b)],
         [2 * (b * d - a * c), 2 * (c * d + a * b), a * a - b * b - c * c + d * d]]
    Rc = [sum(R[i][j] * cp[j] for j in range(3)) for i in range(3)]
    return R, [cq[i] - Rc[i] for i in range(3)]
