"""Build a scene of implicit bones from a converted subject.

Every bone is one Grid field loaded from its own file (see
implicit.anatomy.convert); all further modelling operates on these implicit
bodies. If ``landmarks.json`` exists (implicit.anatomy.landmarks), the fitted
endplate planes can be shown as thin markers.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

from ..expr import Param
from ..scene import Scene
from ..sdf import Grid, HalfSpace, Sphere, smooth
from .convert import VERTEBRAE, load_bones

COLORS = {
    "C": (0.95, 0.86, 0.70), "T": (0.93, 0.83, 0.68), "L": (0.91, 0.80, 0.66),
    "sacrum": (0.88, 0.76, 0.62), "hip": (0.86, 0.78, 0.70), "femur": (0.84, 0.80, 0.74),
    "reference": (0.95, 0.85, 0.35), "marker": (0.2, 0.85, 0.5),
}
GROUP = {"C": "Cervical", "T": "Thoracic", "L": "Lumbar", "sacrum": "Sacrum", "hip": "Pelvis",
         "femur": "Femurs", "reference": "Spinal cord"}


class Model:
    """A converted subject: {name: (kind, Grid)} plus optional landmarks."""

    def __init__(self, bone_dir):
        bone_dir = Path(bone_dir)
        loaded = load_bones(bone_dir)
        if not loaded:
            raise FileNotFoundError(f"no bone files (*.npz) in {bone_dir}")
        self.bones: dict[str, tuple[str, Grid]] = {n: (m["kind"], g) for n, (g, m) in loaded.items()}
        self.meta = {n: m for n, (_, m) in loaded.items()}
        first = next(iter(self.meta.values()))
        self.subject = first.get("subject", "")
        self.attribution = first.get("attribution", "")
        lm = bone_dir / "landmarks.json"
        self.marks = ({k: _arrays(v) for k, v in json.loads(lm.read_text())["landmarks"].items()}
                      if lm.exists() else {})

    def grid(self, level):
        return self.bones["sacrum" if level == "S1" else level][1]

    def shift(self, offset):
        """Move everything by -offset (mm): bones and landmark points."""
        offset = np.asarray(offset, float)
        for _, g in self.bones.values():
            g.origin = g.origin - offset
        for e in self.marks.values():
            if "body_center" in e:
                e["body_center"] = e["body_center"] - offset
            for ep in ("sup", "inf"):
                if ep in e:
                    e[ep]["point"] = e[ep]["point"] - offset


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


def build_scene(bone_dir, name=None, include=("spine", "pelvis", "reference", "markers"), smoothing=True):
    m = Model(bone_dir)
    # Display frame: the scanner frame shifted so the bones sit around the
    # origin. Shader floats lose precision far from the origin (CT coordinates
    # are ~1 m), which shows up as speckle on some GPUs.
    offset = np.round(np.mean(_union_bounds(*[_bounds(g) for k, g in m.bones.values()
                                              if k != "reference"]), axis=0))
    m.shift(offset)
    lo, hi = _union_bounds(*[_bounds(g) for _, g in m.bones.values()])
    scene = Scene(name or f"Implicit bones ({m.subject})", (lo, hi))
    scene.attribution = m.attribution
    scene.frame_offset = offset  # scene coordinates = scanner RAS - offset (mm)
    # Each bone gets a live Gaussian smoothing width (precomputed as a
    # pyramid so the slider is interactive); one switch keeps
    # every bone's volume when smoothing.
    preserve = Param("smooth_preserve", 1, 0, 1, 1, "Smoothing keeps bone volume (1 = on)", "")
    for bname, (kind, g) in m.bones.items():
        key = bname[0] if kind == "vertebra" else kind
        part = {"vertebra": "spine", "reference": "reference"}.get(kind, "pelvis")
        if part not in include:
            continue
        body = g
        if smoothing and kind != "reference":
            sigma = Param(f"smooth_{bname}", 0, 0, 4, 0.1, f"{bname} smoothing σ", "mm")
            body = smooth(g, sigma, preserve)
        scene.add(bname, body, COLORS[key], group=GROUP[key], bounds=_bounds(g),
                  visible=kind != "reference")
    if "markers" in include and m.marks:
        inst, bounds = marker_instances(m)
        if inst:
            scene.add_template("endplate", marker_template(), inst, [
                ("plane", COLORS["marker"], "Endplate planes", False)], bounds)
    return scene, m


def anatomy_graph(bone_dir, relative_to=None, name=None):
    """Block graph of a converted subject.

    Per vertebra: bone -> smooth -> pose -> output, with the vertebra's landmarks
    feeding the pose (so it moves in its own frame) and a hidden marker output.
    Other bones: bone -> smooth -> output (the sacrum also gets landmarks; the
    spinal cord is shown as is). One shared Number keeps bone volume when smoothing.
    File paths are written relative to ``relative_to`` (the graph file's folder).
    """
    bone_dir = Path(bone_dir)
    base = Path(relative_to) if relative_to else bone_dir
    loaded = load_bones(bone_dir)
    rank = {lv: i for i, lv in enumerate(VERTEBRAE)}  # cranial to caudal, then the rest
    loaded = dict(sorted(loaded.items(), key=lambda kv: (rank.get(kv[0], len(rank)), kv[0])))
    subject = next(iter(loaded.values()))[1].get("subject", "")
    hexc = lambda c: "#" + "".join(f"{round(v * 255):02x}" for v in c)
    lm_file = bone_dir / "landmarks.json"
    levels = set(json.loads(lm_file.read_text())["landmarks"]) if lm_file.exists() else set()
    nodes = {"smooth_preserve": {"block": "number", "title": "Keep bone volume", "pos": [0, -110],
                                 "in": {"value": 1, "min": 0, "max": 1, "step": 1,
                                        "label": "Smoothing keeps bone volume (1 = on)"}}}
    ys = {"vertebra": 0, "other": 0}
    for bname, (_, meta) in loaded.items():
        kind = meta["kind"]
        key = bname[0] if kind == "vertebra" else kind
        has_lm = ("S1" if bname == "sacrum" else bname) in levels
        col = "vertebra" if kind == "vertebra" else "other"
        x0, y = (0, ys[col]) if col == "vertebra" else (1000, ys[col])
        ys[col] += 240 if has_lm else 110
        rel = Path(os.path.relpath(bone_dir / f"{bname}.npz", base)).as_posix()
        nodes[bname] = {"block": "bone", "title": bname, "pos": [x0, y], "in": {"file": rel}}
        src = f"@{bname}"
        if kind != "reference":
            nodes[f"{bname}_smooth"] = {"block": "smooth", "title": f"{bname} smoothing", "pos": [x0 + 240, y],
                                        "in": {"bone": src, "sigma": 0, "preserve": "@smooth_preserve"}}
            src = f"@{bname}_smooth"
        if has_lm:
            nodes[f"{bname}_lm"] = {"block": "landmarks", "title": f"{bname} landmarks",
                                    "pos": [x0 + 240, y + 115], "in": {"bone": f"@{bname}"}}
        if has_lm and kind == "vertebra":
            nodes[f"{bname}_pose"] = {"block": "pose", "title": f"{bname} pose", "pos": [x0 + 480, y],
                                      "in": {"body": src, "vertebra": f"@{bname}_lm.vertebra"}}
            src = f"@{bname}_pose"
        right = x0 + (720 if kind == "vertebra" else 480)
        nodes[f"{bname}_out"] = {"block": "output", "title": bname, "pos": [right, y],
                                 "in": {"body": src, "name": bname, "color": hexc(COLORS[key]),
                                        "group": GROUP[key], "visible": kind != "reference"}}
        if has_lm:
            markers = f"@{bname}_lm.markers"
            if kind == "vertebra":  # markers that follow the pose
                nodes[f"{bname}_show"] = {"block": "show_vertebra", "title": f"{bname} markers",
                                          "pos": [x0 + 480, y + 125], "in": {"vertebra": f"@{bname}_pose.vertebra"}}
                markers = f"@{bname}_show"
            nodes[f"{bname}_lm_out"] = {"block": "output", "title": f"{bname} landmarks", "pos": [right, y + 125],
                                        "in": {"body": markers, "name": f"{bname} landmarks",
                                               "color": hexc(COLORS["marker"]), "group": "Landmarks",
                                               "visible": False}}
    return {"format": "implicit.graph", "version": 1,
            "name": name or f"Implicit bones ({subject})", "frame": "auto", "nodes": nodes}
