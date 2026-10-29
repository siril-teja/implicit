"""Build an implicit spine + pelvis scene from a preprocessed model.

Bones are Grid fields. Each intervertebral disc is built implicitly between the
two fitted endplate planes:

  footprint  slice of each adjacent body (vertebra cut behind its posterior wall),
             taken ``ep_inset`` mm inside the endplate and extruded along the
             endplate normal; blended (Morph) from the lower to the upper slice
  disc       footprint (+ bulge) inside the two endplate slabs, minus both bones,
             so it fills the real endplate concavities exactly
  nucleus    disc inside the footprint scaled by sqrt(nuc_area) about a centre
             shifted posteriorly by nuc_shift * depth
  annulus    disc minus nucleus

Every parameter is a uniform: dragging nucleus size or bulge updates all discs live.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..expr import Func, Param
from ..scene import Scene
from ..sdf import Grid, HalfSpace, Morph, ProjectToPlane, ScaleAbout, Sphere
from .totalseg import LEVELS

COLORS = {
    "C": (0.95, 0.86, 0.70), "T": (0.93, 0.83, 0.68), "L": (0.91, 0.80, 0.66),
    "sacrum": (0.88, 0.76, 0.62), "hip": (0.86, 0.78, 0.70), "femur": (0.84, 0.80, 0.74),
    "annulus": (0.36, 0.60, 0.86), "nucleus": (0.92, 0.45, 0.42), "marker": (0.2, 0.85, 0.5),
}
GROUP = {"C": "Cervical", "T": "Thoracic", "L": "Lumbar"}
LAT_MARGIN = 1.0  # mm beyond the isolated body's lateral extent


class Model:
    """Processed subject: bone Grids + landmarks (see preprocess.py)."""

    def __init__(self, model_dir):
        model_dir = Path(model_dir)
        meta = json.loads((model_dir / "model.json").read_text())
        data = np.load(model_dir / "bones.npz")
        self.meta = meta
        self.attribution = meta.get("attribution", "")
        self.bones: dict[str, tuple[str, Grid]] = {}
        for b in meta["bones"]:
            g = Grid.from_quantized(data[f"{b['name']}__q"], b["origin"], b["spacing"], b["name"])
            self.bones[b["name"]] = (b["kind"], g)
        self.marks = {k: _arrays(v) for k, v in meta["landmarks"].items()}

    def levels(self):
        return [lv for lv in LEVELS if lv in self.bones or (lv == "S1" and "sacrum" in self.bones)]

    def grid(self, level):
        return self.bones["sacrum" if level == "S1" else level][1]

    def disc_pairs(self):
        """(upper, lower) level pairs that have both endplates (C2-C3 ... L5-S1)."""
        lv = [x for x in LEVELS if x in self.marks]
        out = []
        for u, l in zip(lv, lv[1:]):
            if LEVELS.index(l) == LEVELS.index(u) + 1 and "inf" in self.marks[u] and "sup" in self.marks[l]:
                out.append((u, l))
        return out


def _arrays(d):
    if isinstance(d, dict):
        return {k: _arrays(v) for k, v in d.items()}
    if isinstance(d, list) and d and isinstance(d[0], (int, float, list)):
        return np.array(d, float)
    return d


def _bounds(g: Grid, pad=0.5):
    lo, hi = g.bounds()
    return lo - pad, hi + pad


def _union_bounds(*bs):
    lo = np.min([b[0] for b in bs], axis=0)
    hi = np.max([b[1] for b in bs], axis=0)
    return lo, hi


class DiscParams:
    def __init__(self):
        self.ep_inset = Param("ep_inset", 1.5, 0.5, 4.0, 0.1, "Footprint slice depth below endplate", "mm")
        self.bulge = Param("disc_bulge", 0.5, -1.0, 3.0, 0.1, "Disc radial bulge", "mm")
        self.nuc_area = Param("nuc_area", 0.40, 0.15, 0.65, 0.01, "Nucleus area fraction", "")
        self.nuc_shift = Param("nuc_shift", 0.08, -0.2, 0.3, 0.01, "Nucleus posterior shift (x depth)", "")


def disc_template(P: DiscParams):
    """Builder shared by all discs (see implicit.template); returns [annulus, nucleus]."""

    def build(b):
        def body_slice(pre):
            # Body cut behind its posterior wall, sliced ep_inset mm inside the
            # endplate, extruded along the endplate normal.
            body = (b.grid(pre + "grid")
                    & HalfSpace(b.vec3(pre + "post_pt"), b.vec3(pre + "post_n"))
                    & HalfSpace(b.vec3(pre + "right_pt"), b.vec3(pre + "lat"))
                    & HalfSpace(b.vec3(pre + "left_pt"), -b.vec3(pre + "lat")))
            n = b.vec3(pre + "n")
            return ProjectToPlane(body, b.vec3(pre + "pt") - n * P.ep_inset, n)

        fp = Morph(body_slice("sl_"), body_slice("su_"),
                   b.vec3("m_pt"), b.vec3("m_axis"), b.scalar("m_len")).offset(P.bulge)
        slabs = HalfSpace(b.vec3("slab_l_pt"), b.vec3("slab_l_n")) & HalfSpace(
            b.vec3("slab_u_pt"), b.vec3("slab_u_n"))
        disc = (fp & slabs) - b.grid("grid_u") - b.grid("grid_l")
        center = b.vec3("nuc_c") - b.vec3("nuc_dir") * P.nuc_shift
        nucleus = disc & ScaleAbout(fp, Func("sqrt", P.nuc_area), center)
        return [disc - nucleus, nucleus]

    return build


def disc_instance(m: Model, upper, lower, slab=4.0):
    """Per-disc constants for disc_template (all derived from the landmarks)."""
    eu, el = m.marks[upper], m.marks[lower]
    pu, nu = eu["inf"]["point"], eu["inf"]["normal"]  # nu points caudally
    pl, nl = el["sup"]["point"], el["sup"]["normal"]  # nl points cranially
    axis = nl - nu
    axis /= np.linalg.norm(axis)
    height = float((pu - pl) @ axis)
    inst = {"name": f"{upper}-{lower}", "grid_u": m.grid(upper), "grid_l": m.grid(lower),
            "m_pt": pl, "m_axis": axis, "m_len": max(height, 1.0),
            "slab_l_pt": pl - nl * slab, "slab_l_n": -nl,
            "slab_u_pt": pu - nu * slab, "slab_u_n": -nu,
            "nuc_c": 0.5 * (pu + pl), "nuc_dir": eu["R"][:, 1] * eu["depth"],
            "height": height}
    # Upper footprint: upper body near its inferior endplate; lower footprint:
    # lower body near its superior endplate. S1 (sacral alae) and C2 (dens,
    # lateral masses) have no clean body outline, so the neighbour's is reused.
    su = (lower, "sup") if upper == "C2" else (upper, "inf")
    sl = (upper, "inf") if lower == "S1" else (lower, "sup")
    for pre, (lv, ep) in (("su_", su), ("sl_", sl)):
        e = m.marks[lv]
        inst[pre + "grid"] = m.grid(lv)
        inst[pre + "post_pt"] = e["post_point"]
        inst[pre + "post_n"] = -e["R"][:, 1]  # keep anterior of the posterior wall
        # ... and between the lateral body margins (drops transverse processes,
        # cervical lateral masses and uncinate flaps from the footprint).
        lat = e["R"][:, 0]
        u0, u1 = e.get("u_range", (-e["width"] / 2, e["width"] / 2))
        inst[pre + "lat"] = lat
        inst[pre + "right_pt"] = e["body_center"] + lat * (u1 + LAT_MARGIN)
        inst[pre + "left_pt"] = e["body_center"] + lat * (u0 - LAT_MARGIN)
        inst[pre + "pt"] = e[ep]["point"]
        inst[pre + "n"] = e[ep]["normal"]
    return inst, disc_bounds(eu, el, slab)


def disc_bounds(eu, el, slab, margin=6.0):
    """AABB of both endplate footprints (+ margin for bulge), extended by the slabs."""
    pts = []
    for e, ep in ((eu, "inf"), (el, "sup")):
        R = eu["R"]
        hw, hd = eu["width"] / 2 + margin, eu["depth"] / 2 + margin
        p, n = e[ep]["point"], e[ep]["normal"]
        for su in (-1, 1):
            for sv in (-1, 1):
                for sn in (-slab, slab):
                    pts.append(p + su * hw * R[:, 0] + sv * hd * R[:, 1] + sn * n)
    pts = np.array(pts)
    return pts.min(0), pts.max(0)


def marker_template(radius=12.0, t=0.4):
    """Thin disc of the fitted endplate plane (for checking the detection)."""

    def build(b):
        p, n = b.vec3("p"), b.vec3("n")
        return [HalfSpace(p + n * t, n) & HalfSpace(p - n * t, -n) & Sphere(radius).translate(p)]

    return build


def marker_instances(m: Model, radius=12.0):
    inst, bounds = [], []
    for lv, e in m.marks.items():
        for which in ("sup", "inf"):
            if which in e:
                p = e[which]["point"]
                inst.append({"name": f"{lv} {which}", "p": p, "n": e[which]["normal"]})
                bounds.append((p - radius - 1, p + radius + 1))
    return inst, bounds


def build_scene(model_dir, name=None, include=("spine", "pelvis", "discs", "markers")):
    m = Model(model_dir)
    P = DiscParams()
    allb = [_bounds(g) for _, g in m.bones.values()]
    lo, hi = _union_bounds(*allb)
    scene = Scene(name or f"Spine + pelvis ({m.meta.get('subject', '')})", (lo, hi))
    scene.attribution = m.attribution
    for bname, (kind, g) in m.bones.items():
        if kind == "vertebra":
            if "spine" not in include:
                continue
            scene.add(bname, g, COLORS[bname[0]], group=GROUP[bname[0]], bounds=_bounds(g))
        elif "pelvis" in include:
            group = {"sacrum": "Sacrum", "hip": "Pelvis", "femur": "Femurs"}[kind]
            scene.add(bname, g, COLORS[kind], group=group, bounds=_bounds(g))
    if "discs" in include:
        pairs = m.disc_pairs()
        if pairs:
            inst, bounds = zip(*(disc_instance(m, u, l) for u, l in pairs))
            scene.add_template("disc", disc_template(P), list(inst), [
                ("annulus", COLORS["annulus"], "Annulus", True),
                ("nucleus", COLORS["nucleus"], "Nucleus", True),
            ], list(bounds))
    if "markers" in include:
        inst, bounds = marker_instances(m)
        if inst:
            scene.add_template("endplate", marker_template(), inst, [
                ("plane", COLORS["marker"], "Endplate planes", False)], bounds)
    return scene, m, P
