"""Build an implicit spine + pelvis scene from a preprocessed model.

Every bone is one Grid field (see implicit.sdf.Grid); all further modelling
operates on these implicit bodies. Endplate planes from the landmarks can be
shown as thin markers to check the detection.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..scene import Scene
from ..sdf import Grid, HalfSpace, Sphere
from .totalseg import LEVELS

COLORS = {
    "C": (0.95, 0.86, 0.70), "T": (0.93, 0.83, 0.68), "L": (0.91, 0.80, 0.66),
    "sacrum": (0.88, 0.76, 0.62), "hip": (0.86, 0.78, 0.70), "femur": (0.84, 0.80, 0.74),
    "marker": (0.2, 0.85, 0.5),
}
GROUP = {"C": "Cervical", "T": "Thoracic", "L": "Lumbar"}


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


def build_scene(model_dir, name=None, include=("spine", "pelvis", "markers")):
    m = Model(model_dir)
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
    if "markers" in include:
        inst, bounds = marker_instances(m)
        if inst:
            scene.add_template("endplate", marker_template(), inst, [
                ("plane", COLORS["marker"], "Endplate planes", False)], bounds)
    return scene, m
