"""Block graphs: a model as a JSON graph of blocks connected by typed wires.

A graph file is the editable form of a scene. Each node is one block (a bone, a
smoothing step, a boolean, an output body); its inputs are either literal
values or wires from another node's output. ``load_graph`` turns a graph into a
Scene, so everything that works on scenes (viewer, STL export, GPU check) works
on graphs.

    {
      "format": "implicit.graph", "version": 1, "name": "Lumbar spine",
      "frame": "auto",
      "nodes": {
        "L3":        {"block": "bone",   "in": {"file": "../assets/bones/s0751/L3.npz"}, "pos": [0, 0]},
        "L3_smooth": {"block": "smooth", "in": {"bone": "@L3", "sigma": 1.0}, "pos": [220, 0]},
        "L3_out":    {"block": "output", "in": {"body": "@L3_smooth", "name": "L3"}, "pos": [440, 0]}
      }
    }

Wires are strings ``"@node"`` (the node's first output) or ``"@node.port"``.
Node ids are identifiers. Literal ``scalar`` and ``vec3`` inputs become live
parameters named ``<node>_<port>`` (``_x``/``_y``/``_z`` for vectors), so
changing a value never rebuilds the shader; only rewiring does.

Port types: ``body`` (a field), ``bone`` (a sampled field from a bone file,
also accepted wherever a body is), ``scalar``, ``vec3``, ``plane`` (point and
unit normal), ``frame`` (origin and right/anterior/cranial axes), ``vertebra``
(a frame with the superior and inferior endplate planes, moved together), and static
settings ``number``, ``text``, ``color``, ``bool``, ``choice``, ``path``.

    python -m implicit.graph blocks             # block manifest (JSON) for the editor
    python -m implicit.graph check GRAPH.json   # validate and summarise
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .expr import BinOp, Computed, ComputedTexture, Const, Expr, Func, Param, Solver, Vec3, cos, radians, sin
from .scene import Scene, pack_grids
from .sdf import (GLSL_LIB, SDF, Box, Capsule, Ctx, Cylinder, Difference, FramePose, Grid, GridSmooth, HalfSpace, RigidMove, TpsWarp, Tube,
                  _rot_np,
                  Intersect, Offset, Rotate, Shell, Sphere, Translate, Union, _pyramid, smooth)

FORMAT, VERSION = "implicit.graph", 1

LIVE = {"scalar", "vec3"}                 # literal values become parameters
WIRED = {"body", "bone", "scalar", "vec3", "plane", "frame", "vertebra", "alignment"}  # types a wire can carry
WIRE_ONLY = {"body", "bone", "plane", "frame", "vertebra", "alignment"}  # inputs that need a wire (no literal)
STATIC = {"number", "text", "color", "bool", "choice", "path"}
ACCEPTS = {"body": {"body", "bone"}, "bone": {"bone"}, "scalar": {"scalar"}, "vec3": {"vec3"},
           "plane": {"plane"}, "frame": {"frame"}, "vertebra": {"vertebra"},
           "alignment": {"alignment"}}


class GraphError(ValueError):
    pass


@dataclass
class Port:
    name: str
    type: str
    default: object = None
    lo: float | None = None
    hi: float | None = None
    step: float | None = None
    unit: str = ""
    label: str = ""
    choices: tuple = ()
    optional: bool = False
    span: float | None = None  # slider range = value +- span (instead of lo..hi)

    def to_json(self):
        d = {"name": self.name, "type": self.type, "label": self.label or self.name}
        for k in ("default", "lo", "hi", "step", "span"):
            if getattr(self, k) is not None:
                d[k] = getattr(self, k)
        if self.unit:
            d["unit"] = self.unit
        if self.choices:
            d["choices"] = list(self.choices)
        if self.optional:
            d["optional"] = True
        return d


@dataclass
class Block:
    name: str
    title: str
    category: str
    doc: str
    inputs: list
    outputs: list
    build: object = field(repr=False, default=None)
    optional: dict = field(default_factory=dict)  # {input: (part, ...)}: parts that may be absent

    def input(self, name):
        return next(p for p in self.inputs if p.name == name)

    def to_json(self):
        d = {"name": self.name, "title": self.title, "category": self.category, "doc": self.doc,
             "inputs": [p.to_json() for p in self.inputs],
             "outputs": [p.to_json() for p in self.outputs]}
        if self.optional:
            d["optional"] = {k: list(v) for k, v in self.optional.items()}
        return d


BLOCKS: dict[str, Block] = {}


def block(name, title, category, inputs, outputs):
    """Register a block; the decorated function maps input values to output values."""

    def reg(fn):
        BLOCKS[name] = Block(name, title, category, (fn.__doc__ or "").strip(), inputs, outputs, fn)
        return fn

    return reg


def manifest():
    """Every block's ports, for the graph editor."""
    return {"format": FORMAT, "version": VERSION, "blocks": [b.to_json() for b in BLOCKS.values()]}


# ---------------------------------------------------------------- values on wires

@dataclass
class Field:
    """A body on a wire: its SDF and how to bound it.

    ``bfun(now)`` returns an axis-aligned box (lo, hi) containing the body, or None:
    with ``now`` False for every slider value in range (baked into shaders, so the
    renderer can skip the body safely), with ``now`` True at the current values
    (tight; for the camera, export and probes).
    """
    sdf: object
    bfun: object = None

    @property
    def bounds(self):
        return self.bfun(False) if self.bfun else None

    def now(self):
        return self.bfun(True) if self.bfun else None


@dataclass
class PlaneV:
    """A plane on a wire: a point on it and its unit normal (vectors of expressions)."""
    point: Vec3
    normal: Vec3


@dataclass
class FrameV:
    """A frame on a wire: origin and the right (x), anterior (y), cranial (z) axes."""
    origin: Vec3
    x: Vec3
    y: Vec3
    z: Vec3


@dataclass
class VertebraV:
    """A vertebra's landmarks on one wire: frame and endplate planes (None if absent)."""
    frame: FrameV
    sup: PlaneV | None
    inf: PlaneV | None


def _const_vec(v):
    return Vec3(*(Const(float(c)) for c in v))


def _interval(e: Expr, now=False):
    """(lo, hi) of an expression over its parameters' slider ranges (``now``: at their
    current values), or None if unknown."""
    if hasattr(e, "expanded"):  # component of a whole vector expression
        return _interval(e.expanded(), now)
    if isinstance(e, Param):
        return (e.v, e.v) if now else (min(e.lo, e.v), max(e.hi, e.v))
    if isinstance(e, Const) or e.is_const():
        v = e.value()
        return v, v
    if isinstance(e, BinOp):
        a, b = _interval(e.a, now), _interval(e.b, now)
        if a is None or b is None:
            return None
        if e.op == "+":
            return a[0] + b[0], a[1] + b[1]
        if e.op == "-":
            return a[0] - b[1], a[1] - b[0]
        if e.op == "*":
            c = [x * y for x in a for y in b]
            return min(c), max(c)
    if isinstance(e, Func) and e.fn in ("sin", "cos"):
        return -1.0, 1.0
    return None


def _upper(e: Expr, now=False):
    iv = _interval(e, now)
    return None if iv is None else iv[1]


def _vec_interval(v: Vec3, now=False):
    ivs = [_interval(c, now) for c in v.c]
    if any(iv is None for iv in ivs):
        return None
    return np.array([iv[0] for iv in ivs]), np.array([iv[1] for iv in ivs])


def _grow(b, r):
    return None if b is None or r is None else (b[0] - max(r, 0.0), b[1] + max(r, 0.0))


def _union_b(a, b):
    if a is None or b is None:
        return None
    return np.minimum(a[0], b[0]), np.maximum(a[1], b[1])


def _intersect_b(a, b):
    if a is None:
        return b
    if b is None:
        return a
    lo, hi = np.maximum(a[0], b[0]), np.minimum(a[1], b[1])
    return lo, np.maximum(hi, lo)


# ---------------------------------------------------------------- blocks

S = lambda name, default, lo, hi, step, unit="", label="": Port(name, "scalar", default, lo, hi, step, unit, label)
V = lambda name, default, lo, hi, step, unit="", label="": Port(name, "vec3", list(default), lo, hi, step, unit, label)
# Positions (centres, offsets, plane points) slide +-SPAN mm around their value, so a
# body's bounds (which must hold over the whole slider range) stay tight; typing a
# value outside the range re-centres it.
SPAN = 50.0
P = lambda name, label="": Port(name, "vec3", [0.0, 0.0, 0.0], None, None, 0.5, "mm", label, span=SPAN)
BODY = lambda name="body", label="": Port(name, "body", label=label)


@block("number", "Number", "Inputs",
       [Port("value", "number", 0.0, label="Value"), Port("min", "number", 0.0, label="Min"),
        Port("max", "number", 1.0, label="Max"), Port("step", "number", 0.01, label="Step"),
        Port("unit", "text", "", label="Unit"), Port("label", "text", "", label="Label")],
       [Port("value", "scalar")])
def _number(x, node):
    """A named slider that can drive any number of scalar inputs."""
    lo, hi = min(x["min"], x["value"]), max(x["max"], x["value"])
    return {"value": Param(node.id, x["value"], lo, hi, x["step"] or None,
                           x["label"] or node.title, x["unit"])}


@block("vector", "Vector", "Inputs",
       [Port("x", "scalar", 0.0, step=0.5, unit="mm", span=SPAN), Port("y", "scalar", 0.0, step=0.5, unit="mm", span=SPAN), Port("z", "scalar", 0.0, step=0.5, unit="mm", span=SPAN)],
       [Port("vector", "vec3")])
def _vector(x, node):
    """Point or direction from three scalars."""
    return {"vector": Vec3(x["x"], x["y"], x["z"])}


@block("bone", "Bone", "Inputs", [Port("file", "path", label="Bone file (.npz)")], [Port("bone", "bone")])
def _bone(x, node):
    """One implicit bone converted from a segmentation (implicit.anatomy.convert)."""
    g = node.ctx.grid(x["file"])
    f = Field(g, lambda now: _grow(g.bounds(), 0.5))
    f.sdf_file = x["file"]
    return {"bone": f}


@block("sphere", "Sphere", "Shapes",
       [S("radius", 10, 0.1, 100, 0.1, "mm"), P("center")],
       [Port("body", "body")])
def _sphere(x, node):
    """Sphere of a given radius."""
    c, r = x["center"], x["radius"]

    def bounds(now):
        civ, riv = _vec_interval(c, now), _upper(r, now)
        return None if civ is None or riv is None else (civ[0] - riv, civ[1] + riv)
    return {"body": Field(Translate(Sphere(r), c), bounds)}


@block("box", "Box", "Shapes",
       [V("size", (20, 20, 20), 0.1, 200, 0.5, "mm", "Size (full extents)"),
        P("center"), S("round", 0, 0, 20, 0.1, "mm", "Edge radius")],
       [Port("body", "body")])
def _box(x, node):
    """Axis-aligned box with optional rounded edges."""
    half = x["size"] * 0.5

    def bounds(now):
        civ, hiv = _vec_interval(x["center"], now), _vec_interval(half, now)
        return None if civ is None or hiv is None else (civ[0] - hiv[1], civ[1] + hiv[1])
    return {"body": Field(Translate(Box(half, x["round"]), x["center"]), bounds)}


@block("cylinder", "Cylinder", "Shapes",
       [S("radius", 10, 0.1, 100, 0.1, "mm"), S("height", 20, 0.1, 200, 0.5, "mm"),
        P("center")],
       [Port("body", "body")])
def _cylinder(x, node):
    """Cylinder along z (cranial)."""
    c = x["center"]

    def bounds(now):
        civ, r, h = _vec_interval(c, now), _upper(x["radius"], now), _upper(x["height"], now)
        return None if civ is None or r is None or h is None else (
            civ[0] - [r, r, h / 2], civ[1] + [r, r, h / 2])
    return {"body": Field(Translate(Cylinder(x["radius"], x["height"] * 0.5), c), bounds)}


def _boolean(cls, combine):
    def build(x, node):
        a, b, k = x["a"], x["b"], x["blend"]
        smooth_k = not (k.is_const() and k.value() == 0)

        def bounds(now):
            bb = combine(a.bfun(now) if a.bfun else None, b.bfun(now) if b.bfun else None)
            return _grow(bb, _upper(k, now)) if smooth_k else bb
        return {"body": Field(cls(a.sdf, b.sdf, k if smooth_k else 0.0), bounds)}
    return build


_BOOL_IN = lambda: [BODY("a", "A"), BODY("b", "B"), S("blend", 0, 0, 20, 0.1, "mm", "Blend radius")]
block("union", "Union", "Booleans", _BOOL_IN(), [Port("body", "body")])(
    _boolean(Union, _union_b))
block("intersect", "Intersect", "Booleans", _BOOL_IN(), [Port("body", "body")])(
    _boolean(Intersect, _intersect_b))
block("subtract", "Subtract", "Booleans", _BOOL_IN(), [Port("body", "body")])(
    _boolean(Difference, lambda a, b: a))
BLOCKS["union"].doc = "A or B; a blend radius > 0 fillets the junction."
BLOCKS["intersect"].doc = "Common part of A and B."
BLOCKS["subtract"].doc = "A with B removed."


def _child(f: Field, now):
    return f.bfun(now) if f.bfun else None


@block("translate", "Move", "Transforms",
       [BODY(), P("offset")], [Port("body", "body")])
def _translate(x, node):
    """Move a body by an offset vector."""
    b, t = x["body"], x["offset"]

    def bounds(now):
        bb, tiv = _child(b, now), _vec_interval(t, now)
        return None if bb is None or tiv is None else (bb[0] + tiv[0], bb[1] + tiv[1])
    return {"body": Field(Translate(b.sdf, t), bounds)}


@block("rotate", "Rotate", "Transforms",
       [BODY(), Port("axis", "choice", "x", choices=("x", "y", "z"), label="Axis"),
        S("angle", 0, -45, 45, 0.5, "deg"), P("center")],
       [Port("body", "body")])
def _rotate(x, node):
    """Rotate a body about an axis through a centre point (right-hand rule)."""
    b, c = x["body"], x["center"]
    sdf = Translate(Rotate(Translate(b.sdf, -c), x["axis"], x["angle"]), c)

    def bounds(now):
        bb, civ = _child(b, now), _vec_interval(c, now)
        if bb is None or civ is None:
            return None
        corners = np.array([[bb[i][0], bb[j][1], bb[k][2]] for i in (0, 1) for j in (0, 1) for k in (0, 1)])
        if now:  # exact: the box corners rotated by the current angle
            i, j = {"x": (1, 2), "y": (2, 0), "z": (0, 1)}[x["axis"]]
            a = np.radians(x["angle"].value())
            q = corners - civ[0]
            r = q.copy()
            r[:, i] = np.cos(a) * q[:, i] - np.sin(a) * q[:, j]
            r[:, j] = np.sin(a) * q[:, i] + np.cos(a) * q[:, j]
            r += civ[0]
            return r.min(0), r.max(0)
        # Any rotation stays inside the sphere about the centre through the farthest corner.
        far = np.maximum(np.abs(corners - civ[0]), np.abs(corners - civ[1]))  # any centre in range
        r = np.linalg.norm(far, axis=1).max()
        return civ[0] - r, civ[1] + r
    return {"body": Field(sdf, bounds)}


@block("offset", "Offset", "Modify", [BODY(), S("distance", 0, -10, 10, 0.1, "mm")], [Port("body", "body")])
def _offset(x, node):
    """Grow (> 0) or shrink (< 0) a body by a distance."""
    b, d = x["body"], x["distance"]
    return {"body": Field(Offset(b.sdf, d), lambda now: _grow(_child(b, now), _upper(d, now)))}


@block("shell", "Shell", "Modify", [BODY(), S("thickness", 1, 0.1, 10, 0.1, "mm")], [Port("body", "body")])
def _shell(x, node):
    """Hollow a body, keeping an inner wall of the given thickness (e.g. a cortex)."""
    b = x["body"]
    return {"body": Field(Shell(b.sdf, x["thickness"]), lambda now: _child(b, now))}


@block("plane_cut", "Plane cut", "Modify",
       [BODY(), P("point"),
        V("normal", (0, 0, 1), -1, 1, 0.01, "", "Normal (removed side)")],
       [Port("body", "body")])
def _plane_cut(x, node):
    """Remove the part of a body on the normal side of a plane."""
    b = x["body"]
    return {"body": Field(Intersect(b.sdf, HalfSpace(x["point"], x["normal"])), lambda now: _child(b, now))}


@block("smooth", "Smooth", "Modify",
       [Port("bone", "bone", label="Bone"), S("sigma", 0, 0, 4, 0.1, "mm", "Smoothing σ"),
        S("preserve", 1, 0, 1, 1, "", "Keep volume (1 = on)")],
       [Port("body", "body")])
def _smooth(x, node):
    """Gaussian smoothing of a bone surface, width σ; optionally keeps the bone volume."""
    b = x["bone"]
    return {"body": Field(smooth(b.sdf, x["sigma"], x["preserve"]), lambda now: _child(b, now))}


# ---------------------------------------------------------------- landmarks, planes, frames

MARK_R, MARK_T, AXIS_L, AXIS_R = 14.0, 0.35, 12.0, 0.6  # landmark marker sizes (mm)


def _disc(p: Vec3, n: Vec3, radius, half_t):
    return Intersect(Intersect(HalfSpace(p + n * half_t, n), HalfSpace(p - n * half_t, -n)),
                     Translate(Sphere(radius), p))


def _axes(f: FrameV, length, radius):
    o = f.origin
    return Union(Union(Capsule(o, o + f.x * length, radius), Capsule(o, o + f.y * length, radius)),
                 Capsule(o, o + f.z * length, radius))


def landmark_markers(lm):
    """Marker body of a bone's landmarks (endplate discs and frame axes) and its bounds."""
    frame = FrameV(_const_vec(lm["origin"]), *(_const_vec(a) for a in lm["axes"]))
    sdf = _axes(frame, AXIS_L, AXIS_R)
    pts = [lm["origin"] - AXIS_L - AXIS_R, lm["origin"] + AXIS_L + AXIS_R]
    for k in ("sup", "inf"):
        if k in lm:
            p, n = lm[k]
            sdf = Union(sdf, _disc(_const_vec(p), _const_vec(n), MARK_R, MARK_T))
            pts += [p - MARK_R - MARK_T, p + MARK_R + MARK_T]
    return sdf, (np.min(pts, axis=0), np.max(pts, axis=0))


@block("landmarks", "Landmarks", "Anatomy",
       [Port("bone", "bone", label="Vertebra (or sacrum)")],
       [Port("vertebra", "vertebra", label="Vertebra"), Port("frame", "frame", label="Frame"),
        Port("superior", "plane", label="Superior endplate"), Port("inferior", "plane", label="Inferior endplate"),
        Port("markers", "body", label="Markers")])
def _landmarks(x, node):
    """Vertebral body frame (origin between the endplates; right, anterior, cranial axes) and the
    endplate planes (normals point away from the body), detected from the implicit bone."""
    lm = node.ctx.landmarks(x["bone"].sdf_file)
    if lm is None:
        raise GraphError("no landmarks for this bone (vertebrae and sacrum only; "
                         "run python -m implicit.anatomy.landmarks on its folder)")
    out = {"frame": FrameV(_const_vec(lm["origin"]), *(_const_vec(a) for a in lm["axes"]))}
    for k, port in (("sup", "superior"), ("inf", "inferior")):
        out[port] = PlaneV(_const_vec(lm[k][0]), _const_vec(lm[k][1])) if k in lm else None
    sdf, b = landmark_markers(lm)
    out["markers"] = Field(sdf, lambda now: b)
    out["vertebra"] = VertebraV(out["frame"], out["superior"], out["inferior"])
    return out


@block("plane", "Plane", "Planes & frames",
       [P("point"), V("normal", (0, 0, 1), -1, 1, 0.01, "", "Normal")], [Port("plane", "plane")])
def _plane(x, node):
    """Plane through a point with a normal (normalised)."""
    return {"plane": PlaneV(x["point"], x["normal"])}


@block("plane_parts", "Plane parts", "Planes & frames", [Port("plane", "plane", label="Plane")],
       [Port("point", "vec3", label="Point"), Port("normal", "vec3", label="Normal")])
def _plane_parts(x, node):
    """Point and normal of a plane."""
    return {"point": x["plane"].point, "normal": x["plane"].normal}


@block("frame_parts", "Frame parts", "Planes & frames", [Port("frame", "frame", label="Frame")],
       [Port("origin", "vec3", label="Origin"), Port("x", "vec3", label="Right (x)"),
        Port("y", "vec3", label="Anterior (y)"), Port("z", "vec3", label="Cranial (z)")])
def _frame_parts(x, node):
    """Origin and axes of a frame."""
    f = x["frame"]
    return {"origin": f.origin, "x": f.x, "y": f.y, "z": f.z}


@block("show_plane", "Show plane", "Planes & frames",
       [Port("plane", "plane", label="Plane"), S("radius", 14, 1, 60, 0.5, "mm"),
        S("thickness", 0.7, 0.1, 5, 0.1, "mm")], [Port("body", "body")])
def _show_plane(x, node):
    """A thin disc on a plane, to see it in 3D."""
    pl, r, t = x["plane"], x["radius"], x["thickness"]
    n = Vec3(*(c for c in pl.normal.c))

    def bounds(now):
        piv, ru, tu = _vec_interval(pl.point, now), _upper(r, now), _upper(t, now)
        return None if piv is None or ru is None or tu is None else (piv[0] - ru - tu, piv[1] + ru + tu)
    # The normal is used as given: planes from Landmarks and Plane parts are unit length.
    return {"body": Field(_disc(pl.point, n, r, t * 0.5), bounds)}


@block("show_frame", "Show frame", "Planes & frames",
       [Port("frame", "frame", label="Frame"), S("length", 12, 1, 60, 0.5, "mm"),
        S("radius", 0.6, 0.1, 5, 0.1, "mm")], [Port("body", "body")])
def _show_frame(x, node):
    """The three axes of a frame as thin rods (right, anterior, cranial)."""
    f, length, r = x["frame"], x["length"], x["radius"]

    def bounds(now):
        oiv, lu, ru = _vec_interval(f.origin, now), _upper(length, now), _upper(r, now)
        return None if oiv is None or lu is None or ru is None else (oiv[0] - lu - ru, oiv[1] + lu + ru)
    return {"body": Field(_axes(f, length, r), bounds)}


# ---------------------------------------------------------------- pose

def _pose_np(fn, args):
    """NumPy value of poseDir/posePoint/rigDir/rigPoint (see GLSL_LIB) for float vectors."""
    if fn in ("rigDir", "rigPoint"):
        v, c0, c1, c2, t = args
        return c0 * v[0] + c1 * v[1] + c2 * v[2] + (t if fn == "rigPoint" else 0.0)
    if fn == "poseDir":
        v, ex, ey, ez, ang = args
        E = np.stack([ex, ey, ez])
        loc = (E @ v)[None]
        loc = _rot_np(_rot_np(_rot_np(loc, 0, -np.radians(ang[0])), 1, np.radians(ang[1])), 2, np.radians(ang[2]))
        return loc[0] @ E
    p, o, ex, ey, ez, ang, sh = args
    E = np.stack([ex, ey, ez])
    return o + _pose_np("poseDir", [p - o, ex, ey, ez, ang]) + sh @ E


class _PoseCall:
    """poseDir(...) or posePoint(...) (see GLSL_LIB) on vectors of expressions."""

    def __init__(self, fn, args):
        self.fn, self.args = fn, args

    def glsl(self):
        return f"{self.fn}({', '.join(a.glsl() for a in self.args)})"

    def value(self):
        return _pose_np(self.fn, [np.array(a.value(), float) for a in self.args])


class _PoseComp(Expr):
    """One component of a pose call. A Vec3 of the three components emits the call once,
    so moved planes and frames stay small in the shader."""

    def __init__(self, whole: _PoseCall, comp):
        self.whole, self.comp = whole, comp

    def glsl(self):
        return f"{self.whole.glsl()}.{'xyz'[self.comp]}"

    def value(self):
        return float(self.whole.value()[self.comp])

    def params(self):
        return [q for a in self.whole.args for q in a.params()]

    def is_const(self):
        return all(a.is_const() for a in self.whole.args)


class _Pose:
    """The rigid motion of the Pose block, applied to points and directions (as expressions)."""

    def __init__(self, f: FrameV, flexion, lateral, axial, shift: Vec3):
        self.f, self.shift = f, shift
        self.ang = Vec3(flexion, lateral, axial)

    def _call(self, fn, *args):
        whole = _PoseCall(fn, list(args))
        return Vec3(*(_PoseComp(whole, k) for k in range(3)))

    def direction(self, n: Vec3):
        f = self.f
        return self._call("poseDir", n, f.x, f.y, f.z, self.ang)

    def point(self, p: Vec3):
        f = self.f
        return self._call("posePoint", p, f.origin, f.x, f.y, f.z, self.ang, self.shift)

    def plane(self, pl):
        return None if pl is None else PlaneV(self.point(pl.point), self.direction(pl.normal))

    def frame(self):
        f = self.f
        return FrameV(self.point(f.origin), self.direction(f.x), self.direction(f.y), self.direction(f.z))


@block("pose", "Pose", "Anatomy",
       [BODY(label="Body"), Port("vertebra", "vertebra", label="Vertebra (its frame)"),
        S("flexion", 0, -30, 30, 0.5, "deg", "Flexion (+) / extension (−)"),
        S("lateral", 0, -30, 30, 0.5, "deg", "Lateral bending (+ right)"),
        S("axial", 0, -30, 30, 0.5, "deg", "Axial rotation (+ front turns left)"),
        Port("shift", "vec3", [0.0, 0.0, 0.0], step=0.5, unit="mm", span=20.0,
             label="Shift (right, anterior, cranial)")],
       [Port("body", "body", label="Body"), Port("vertebra", "vertebra", label="Vertebra")])
def _pose(x, node):
    """Move a vertebra in its own frame: rotations about its body centre (flexion,
    lateral bending, axial rotation) and a shift along its axes. Its frame and
    endplates move with it, so blocks downstream (discs, cages) follow."""
    b, v = x["body"], x["vertebra"]
    f = v.frame
    pose = _Pose(f, x["flexion"], x["lateral"], x["axial"], x["shift"])
    sdf = FramePose(b.sdf, f.origin, f.x, f.y, f.z, x["flexion"], x["lateral"], x["axial"], x["shift"])

    def bounds(now):
        bb, oiv = _child(b, now), _vec_interval(f.origin, now)
        if bb is None or oiv is None:
            return None
        corners = np.array([[bb[i][0], bb[j][1], bb[k][2]] for i in (0, 1) for j in (0, 1) for k in (0, 1)])
        if now:  # exact: the corners moved by the current pose
            c = np.array([[e.value() for e in pose.point(Vec3(*map(float, q))).c] for q in corners])
            return c.min(0), c.max(0)
        # Any rotation about the origin stays in the sphere through the farthest corner; the
        # shift adds at most its length.
        far = np.maximum(np.abs(corners - oiv[0]), np.abs(corners - oiv[1]))
        siv = _vec_interval(x["shift"], now)
        if siv is None:
            return None
        r = np.linalg.norm(far, axis=1).max() + np.linalg.norm(np.maximum(np.abs(siv[0]), np.abs(siv[1])))
        return oiv[0] - r, oiv[1] + r
    return {"body": Field(sdf, bounds), "vertebra": VertebraV(pose.frame(), pose.plane(v.sup), pose.plane(v.inf))}


@block("show_vertebra", "Show vertebra", "Anatomy",
       [Port("vertebra", "vertebra", label="Vertebra"), S("radius", 14, 1, 60, 0.5, "mm", "Endplate disc radius"),
        S("length", 12, 1, 60, 0.5, "mm", "Axis length")], [Port("body", "body")])
def _show_vertebra(x, node):
    """A vertebra's frame axes and endplate discs, to check them in 3D (follows Pose)."""
    v, r, length = x["vertebra"], x["radius"], x["length"]
    sdf = _axes(v.frame, length, AXIS_R)
    for pl in (v.sup, v.inf):
        if pl is not None:
            sdf = Union(sdf, _disc(pl.point, pl.normal, r, MARK_T))

    def bounds(now):
        parts = [_vec_interval(v.frame.origin, now)] + [_vec_interval(pl.point, now) for pl in (v.sup, v.inf) if pl]
        ru, lu = _upper(r, now), _upper(length, now)
        if any(q is None for q in parts) or ru is None or lu is None:
            return None
        e = max(ru, lu) + AXIS_R + MARK_T
        return np.min([q[0] for q in parts], axis=0) - e, np.max([q[1] for q in parts], axis=0) + e
    return {"body": Field(sdf, bounds)}


BLOCKS["show_vertebra"].optional = {"vertebra": ("sup", "inf")}  # e.g. the sacrum has no inferior endplate


@block("vertebra_parts", "Vertebra parts", "Anatomy", [Port("vertebra", "vertebra", label="Vertebra")],
       [Port("frame", "frame", label="Frame"), Port("superior", "plane", label="Superior endplate"),
        Port("inferior", "plane", label="Inferior endplate")])
def _vertebra_parts(x, node):
    """Frame and endplate planes of a vertebra (e.g. after Pose)."""
    v = x["vertebra"]
    return {"frame": v.frame, "superior": v.sup, "inferior": v.inf}


# ---------------------------------------------------------------- spine alignment

@dataclass
class AlignmentV:
    """The chain model of one subject: a Solver, its motion uniforms and its TPS table."""
    solver: Solver
    motions: dict = field(default_factory=dict)
    table: object = None

    def motion(self, seg):
        """The rigid motion of a segment as 12 Computed uniforms (see RigidMove)."""
        if seg not in self.motions:
            self.motions[seg] = [Computed(f"{self.solver.id}_{seg}_{i}", self.solver, seg, i) for i in range(12)]
        return self.motions[seg]

    def tps(self):
        """The inverse warp table (see alignment.tps_table) as a computed texture."""
        if self.table is None:
            self.table = ComputedTexture(f"{self.solver.id}_tps", self.solver, "tps", 3)
        return self.table

    def displacement(self, now):
        """Largest control point displacement (mm), now or over the slider ranges."""
        from .anatomy.alignment import _apply
        pts = self.solver.data["tps_points"]
        out = 0.0
        for res in [self.solver.result()] if now else self.solver.samples():
            for q in pts:
                m = res[q["segment"]]
                R, t = [[m[0], m[3], m[6]], [m[1], m[4], m[7]], [m[2], m[5], m[8]]], m[9:]
                out = max(out, float(np.linalg.norm(np.subtract(_apply((R, t), q["point"]), q["point"]))))
        return out


def _rig_point(m, v: Vec3, move=True):
    """A point (or direction) moved by the rigid motion m, as one rigPoint/rigDir call."""
    whole = _PoseCall("rigPoint" if move else "rigDir", [v, *(Vec3(*m[i:i + 3]) for i in range(0, 12, 3))])
    return Vec3(*(_PoseComp(whole, k) for k in range(3)))


def _rig_np(m, P):
    m = np.asarray(m, float)
    return np.asarray(P, float) @ m[:9].reshape(3, 3) + m[9:]


_LEVELS = tuple([f"T{i}" for i in range(1, 13)] + [f"L{i}" for i in range(1, 6)])
_CURVE_PORTS = [(k, part) for k in ("MT", "TL") for part in ("upper", "apex", "lower")]


def _curve_defaults():
    from .anatomy.alignment import CURVES
    return {f"{k}_{part}": CURVES[k][i] for k in ("MT", "TL") for i, part in enumerate(("upper", "apex", "lower"))}


@block("spinopelvic", "Spine alignment", "Anatomy",
       [Port("bone", "bone", label="Sacrum (finds the subject's landmarks)"),
        S("PI", 58.7, 30, 90, 0.5, "deg", "Pelvic incidence"), S("PT", 17.2, -10, 45, 0.5, "deg", "Pelvic tilt"),
        S("LL", 51.9, 0, 90, 0.5, "deg", "Lumbar lordosis (L1-S1)"),
        S("TK", 30.3, 0, 80, 0.5, "deg", "Thoracic kyphosis (T4-T12)"),
        S("MT", 0, -90, 90, 0.5, "deg", "Main thoracic Cobb (+ convex right)"),
        S("TL", 0, -90, 90, 0.5, "deg", "Thoracolumbar/lumbar Cobb (+ convex right)"),
        S("AR", 0.3, 0, 1, 0.05, "", "Apical rotation per degree of Cobb"),
        S("PO", 0, -20, 20, 0.5, "deg", "Pelvic obliquity (+ right side up)"),
        S("CB", 0, -80, 80, 1, "mm", "Coronal balance (T1 right of S1)")]
       + [Port(f"{k}_{part}", "choice", d, label=f"{k} {part}{' end' if part != 'apex' else ''}", choices=_LEVELS)
          for (k, part), d in zip(_CURVE_PORTS, _curve_defaults().values())],
       [Port("alignment", "alignment", label="Alignment")])
def _spinopelvic(x, node):
    """Sagittal and coronal alignment of the spine and pelvis. At the subject's own
    values nothing moves. Sagittal: the pelvis turns about the hip axis (PT), the
    sacrum about the sacroiliac joints (PI), the discs L1-S1 (LL) and T4-T12 (TK).
    Coronal: two Cobb curves between chosen end vertebrae and apex (MT, TL) with the
    apex vertebrae turned toward the convexity (AR), pelvic obliquity (PO) and coronal
    balance (CB). Feed it to an Align block per bone and to TPS warp blocks for soft
    structures (see implicit/anatomy/alignment.py)."""
    from .anatomy.alignment import _curve_plan
    data = node.ctx.spine(x["bone"].sdf_file)
    if data is None:
        raise GraphError("no spine and pelvis landmarks for this subject (needs vertebrae, sacrum, hip bones "
                         "and femurs; run python -m implicit.anatomy.landmarks on its folder)")
    curves = {k: [x[f"{k}_{part}"] for part in ("upper", "apex", "lower")] for k in ("MT", "TL")}
    curves = {k: c for k, c in curves.items() if all(lv in data["levels"] for lv in c)}  # e.g. a lumbar-only subject
    try:
        _curve_plan(data["levels"], curves)
        solver = Solver(node.id, "spinopelvic", {**data, "curves": curves},
                        {k: x[k] for k in ("PI", "PT", "LL", "TK", "MT", "TL", "AR", "PO", "CB")})
    except ValueError as e:
        raise GraphError(str(e)) from e
    return {"alignment": AlignmentV(solver)}


@block("align", "Align", "Anatomy",
       [BODY(label="Body"), Port("vertebra", "vertebra", label="Vertebra (optional)", optional=True),
        Port("alignment", "alignment", label="Alignment"),
        Port("segment", "text", "L5", label="Segment (legs, pelvis, sacrum, L5 ... T1)")],
       [Port("body", "body", label="Body"), Port("vertebra", "vertebra", label="Vertebra")])
def _align(x, node):
    """Move a bone (and its vertebra frame) rigidly as one segment of a spine alignment."""
    a, seg, b, v = x["alignment"], x["segment"], x["body"], x["vertebra"]
    if seg not in a.solver.result() or seg == "tps":
        raise GraphError(f"segment {seg!r} is not in this alignment "
                         f"({', '.join(k for k in a.solver.result() if k != 'tps')})")
    m = a.motion(seg)

    def bounds(now):
        bb = _child(b, now)
        if bb is None:
            return None
        corners = [[bb[i][0], bb[j][1], bb[k][2]] for i in (0, 1) for j in (0, 1) for k in (0, 1)]
        ms = [[e.value() for e in m]] if now else [r[seg] for r in a.solver.samples()]
        pts = np.concatenate([_rig_np(q, corners) for q in ms])
        return pts.min(0), pts.max(0)
    out = {"body": Field(RigidMove(b.sdf, m, _child(b, False)), bounds), "vertebra": None}  # local box: culling
    if v is not None:
        f = v.frame
        plane = lambda pl: None if pl is None else PlaneV(_rig_point(m, pl.point), _rig_point(m, pl.normal, False))
        out["vertebra"] = VertebraV(FrameV(_rig_point(m, f.origin), *(_rig_point(m, d, False) for d in (f.x, f.y, f.z))),
                                    plane(v.sup), plane(v.inf))
    return out


@block("spine_curve", "Spine curve", "Anatomy",
       [Port("alignment", "alignment", label="Alignment"), S("radius", 1.5, 0.2, 6, 0.1, "mm", "Tube radius")],
       [Port("body", "body", label="Body")])
def _spine_curve(x, node):
    """A tube along the subject's spine curve: a smooth curve through the S1 endplate
    centre and the vertebral body centres, as it is before any alignment change (wire
    it through a TPS warp to follow the alignment)."""
    P = np.reshape(x["alignment"].solver.data["curve"], (-1, 3))
    r = x["radius"]

    def bounds(now):
        ru = _upper(r, now)
        return None if ru is None else (P.min(0) - ru, P.max(0) + ru)
    return {"body": Field(Tube([_const_vec(q) for q in P], r), bounds)}


@block("tps_warp", "TPS warp", "Anatomy",
       [BODY(label="Body"), Port("alignment", "alignment", label="Alignment")],
       [Port("body", "body", label="Body")])
def _tps_warp(x, node):
    """Carry a soft structure (spine curve, discs, ligaments) with the alignment: a
    thin-plate spline through control points on every bone, which move rigidly."""
    a, b = x["alignment"], x["body"]

    def bounds(now):
        bb = _child(b, now)
        return None if bb is None else _grow(bb, a.displacement(now) + 2.0)
    return {"body": Field(TpsWarp(b.sdf, a.tps(), len(a.solver.data["tps_points"])), bounds)}


@block("output", "Output body", "Output",
       [BODY(), Port("name", "text", "", label="Name"), Port("color", "color", "#d9c4a8", label="Colour"),
        Port("group", "text", "", label="Group"), Port("visible", "bool", True, label="Visible")],
       [])
def _output(x, node):
    """Show a body in the viewer and make it exportable."""
    return {}


# ---------------------------------------------------------------- graph -> scene

@dataclass
class Node:
    id: str
    block: Block
    raw: dict
    title: str
    ctx: object = None


class _Context:
    """Bone grids of a graph (and, with ``whole_dirs``, every bone file next to them)."""

    def __init__(self, base: Path, files, whole_dirs=False):
        self.base = Path(base)
        self._grids, self.kinds, self.names = {}, {}, {}
        self.attribution = ""
        from .anatomy.convert import load_bone
        self.graph_paths = [self.resolve(f) for f in files]
        for f, path in zip(files, self.graph_paths):
            if not path.exists():
                raise GraphError(f"bone file not found: {f} ({path})")
        paths = list(dict.fromkeys(self.graph_paths))
        if whole_dirs:
            for d in dict.fromkeys(q.parent for q in self.graph_paths):
                paths += [q.resolve() for q in sorted(d.glob("*.npz")) if q.resolve() not in paths]
        for path in paths:
            g, meta = load_bone(path)
            self._grids[path] = g
            self.kinds[path] = meta.get("kind", "")
            self.names[path] = meta.get("name", path.stem)
            self.attribution = self.attribution or meta.get("attribution", "")

    def resolve(self, f):
        p = Path(f)
        return (p if p.is_absolute() else self.base / p).resolve()

    def grid(self, f) -> Grid:
        return self._grids[self.resolve(f)]

    def auto_offset(self):
        """Display frame: bones (not references) centred at the origin, whole mm."""
        own = {p: self._grids[p] for p in self.graph_paths}
        bs = [g.bounds() for p, g in own.items() if self.kinds[p] != "reference"]
        bs = bs or [g.bounds() for g in own.values()]
        if not bs:
            return np.zeros(3)
        lo = np.min([b[0] - 0.5 for b in bs], axis=0)
        hi = np.max([b[1] + 0.5 for b in bs], axis=0)
        return np.round((lo + hi) / 2)

    def shift(self, offset):
        for g in self._grids.values():
            g.origin = g.origin - offset
        self.offset = np.asarray(offset, float)

    def landmarks(self, f):
        """Landmarks of a bone (landmarks.json next to it), in the scene frame, or None."""
        path = self.resolve(f)
        cache = self.__dict__.setdefault("_lm", {})
        if path.parent not in cache:
            lj = path.parent / "landmarks.json"
            cache[path.parent] = json.loads(lj.read_text())["landmarks"] if lj.exists() else {}
        name = self.names.get(path, path.stem)
        e = cache[path.parent].get("S1" if name == "sacrum" else name)
        if not e:
            return None
        off = getattr(self, "offset", np.zeros(3))
        # Stored to 4 decimals: restore exact orthonormal axes (nearest rotation), so a
        # pose with zero angles is exactly the identity.
        u, _, vt = np.linalg.svd(np.array(e["R"], float))
        R = u @ vt
        out = {"origin": np.array(e["body_center"], float) - off, "axes": R.T.copy()}
        for k in ("sup", "inf"):
            if k in e:
                out[k] = (np.array(e[k]["point"], float) - off, np.array(e[k]["normal"], float))
        return out


def _spine(ctx, f):
    """Chain-model data of the subject of bone file ``f`` (scene frame), or None."""
    from .anatomy import alignment
    path = ctx.resolve(f)
    cache = ctx.__dict__.setdefault("_spine", {})
    if path.parent not in cache:
        lj = path.parent / "landmarks.json"
        d = json.loads(lj.read_text()) if lj.exists() else {}
        cache[path.parent] = alignment.chain(d.get("landmarks", {}), d.get("pelvis"),
                                             getattr(ctx, "offset", np.zeros(3)))
    return cache[path.parent]


_Context.spine = _spine


def _parse_wire(v):
    if isinstance(v, str) and v.startswith("@"):
        ref = v[1:]
        node, _, port = ref.partition(".")
        return node, port or None
    return None


def _order(nodes: dict[str, Node]):
    """Topological order of the nodes (inputs before users); rejects cycles."""
    deps = {nid: {w[0] for w in map(_parse_wire, n.raw.get("in", {}).values()) if w} for nid, n in nodes.items()}
    order, state = [], {}

    def visit(nid, path):
        if state.get(nid) == 2:
            return
        if state.get(nid) == 1:
            raise GraphError("cycle: " + " -> ".join(path + [nid]))
        state[nid] = 1
        for d in sorted(deps[nid]):
            if d not in nodes:
                raise GraphError(f"{nid}: wire from unknown node {d!r}")
            visit(d, path + [nid])
        state[nid] = 2
        order.append(nid)

    for nid in nodes:
        visit(nid, [])
    return order


def _color(v):
    if isinstance(v, str):
        s = v.lstrip("#")
        if len(s) != 6:
            raise GraphError(f"colour must be #rrggbb, got {v!r}")
        return tuple(int(s[i:i + 2], 16) / 255 for i in (0, 2, 4))
    return tuple(float(c) for c in v)


def _literal(node: Node, port: Port, v):
    """A literal input value; scalar and vec3 literals become live parameters."""
    label = f"{node.title}: {port.label or port.name}"
    def rng(v):
        if port.span is not None:
            return v - port.span, v + port.span
        return (min(port.lo if port.lo is not None else v, v), max(port.hi if port.hi is not None else v, v))

    if port.type == "scalar":
        v = float(v)
        lo, hi = rng(v)
        return Param(f"{node.id}_{port.name}", v, lo, hi, port.step, label, port.unit)
    if port.type == "vec3":
        if not (isinstance(v, (list, tuple)) and len(v) == 3):
            raise GraphError(f"{node.id}.{port.name}: expected [x, y, z], got {v!r}")
        comps = []
        for axis, c in zip("xyz", v):
            c = float(c)
            lo, hi = rng(c)
            comps.append(Param(f"{node.id}_{port.name}_{axis}", c, lo, hi, port.step, f"{label} {axis}", port.unit))
        return Vec3(*comps)
    if port.type == "number":
        return float(v)
    if port.type in ("text", "path"):
        return str(v)
    if port.type == "bool":
        return bool(v)
    if port.type == "color":
        return _color(v)
    if port.type == "choice":
        if v not in port.choices:
            raise GraphError(f"{node.id}.{port.name}: {v!r} is not one of {list(port.choices)}")
        return v
    raise GraphError(f"{node.id}.{port.name}: port type {port.type!r} cannot take a literal")


def read_graph(path):
    """Parse and check the header of a graph file; returns the dict."""
    g = json.loads(Path(path).read_text())
    if g.get("format") != FORMAT:
        raise GraphError(f"{path}: not an {FORMAT} file")
    if g.get("version", 0) > VERSION:
        raise GraphError(f"{path}: graph version {g['version']} is newer than this program ({VERSION})")
    return g


def _bone_files(nodes):
    files = [n["in"].get("file") if isinstance(n.get("in"), dict) else None
             for n in nodes.values() if n.get("block") == "bone"]
    if any(f is None for f in files):
        raise GraphError("every bone block needs a file")
    return files


def _frame_offset(g, ctx):
    frame = g.get("frame", "auto")
    return ctx.auto_offset() if frame == "auto" else np.asarray(frame.get("offset", (0, 0, 0)), float)


def build_graph(g: dict, base=".", library=None) -> Scene:
    """Build a Scene from a graph dict (``base``: directory that relative file paths start from).

    With a ``Library`` the bones come from its shared, already packed atlas (as in the viewer).
    """
    raw_nodes = g.get("nodes") or {}
    nodes: dict[str, Node] = {}
    for nid, raw in raw_nodes.items():
        if not nid.isidentifier():
            raise GraphError(f"node id must be an identifier: {nid!r}")
        if raw.get("block") not in BLOCKS:
            raise GraphError(f"{nid}: unknown block {raw.get('block')!r}")
        blk = BLOCKS[raw["block"]]
        unknown = set(raw.get("in", {})) - {p.name for p in blk.inputs}
        if unknown:
            raise GraphError(f"{nid}: block {blk.name!r} has no input(s) {sorted(unknown)}")
        nodes[nid] = Node(nid, blk, raw, raw.get("title") or nid)

    files = _bone_files(raw_nodes)
    if library is not None:
        ctx, offset = library.ctx, library.offset
        missing = [f for f in files if ctx.resolve(f) not in ctx._grids]
        if missing:
            raise GraphError(f"bone(s) not in the library: {missing}")
    else:
        ctx = _Context(Path(base), files)
        offset = _frame_offset(g, ctx)
        ctx.shift(offset)

    values: dict[str, dict] = {}
    outputs = []
    for nid in _order(nodes):
        node = nodes[nid]
        node.ctx = ctx
        given = node.raw.get("in", {})
        x = {}
        for port in node.block.inputs:
            v = given.get(port.name, port.default)
            wire = _parse_wire(v)
            if wire:
                src, sport = wire
                outs = nodes[src].block.outputs
                sp = next((p for p in outs if p.name == sport), None) if sport else (outs[0] if outs else None)
                if sp is None:
                    raise GraphError(f"{nid}.{port.name}: node {src!r} has no output {sport or ''!r}")
                if port.type not in ACCEPTS or sp.type not in ACCEPTS[port.type]:
                    raise GraphError(f"{nid}.{port.name}: cannot connect {sp.type} ({src}.{sp.name}) "
                                     f"to a {port.type} input")
                x[port.name] = values[src][sp.name]
                if x[port.name] is None:
                    raise GraphError(f"{nid}.{port.name}: {src} has no {sp.label or sp.name} "
                                     f"(e.g. the sacrum has no inferior endplate)")
            elif port.type in WIRE_ONLY and port.optional:
                x[port.name] = None
            elif port.type in WIRE_ONLY:
                raise GraphError(f"{nid}.{port.name}: needs a {port.type} wire")
            elif v is None:
                raise GraphError(f"{nid}.{port.name}: missing value")
            else:
                x[port.name] = _literal(node, port, v)
        try:
            values[nid] = node.block.build(x, node)
        except GraphError:
            raise
        except Exception as e:
            raise GraphError(f"{nid} ({node.block.name}): {e}") from e
        if node.block.name == "output":
            outputs.append((node, x))

    if not outputs:
        raise GraphError("graph has no output block")
    known = [f.bounds for _, x in outputs if (f := x["body"]).bounds is not None]
    if g.get("bounds"):
        lo, hi = (np.asarray(b, float) for b in g["bounds"])
    elif known:
        lo, hi = np.min([b[0] for b in known], axis=0), np.max([b[1] for b in known], axis=0)
    else:
        raise GraphError("no output has finite bounds; give the graph \"bounds\": [[x,y,z],[x,y,z]]")
    scene = Scene(g.get("name") or "graph", (lo, hi))
    scene.frame_offset = offset
    scene.attribution = ctx.attribution
    scene.extra = {"graph": True}
    if library is not None:
        scene.packed_atlas = library.atlas
    for node, x in outputs:
        f = x["body"]
        scene.add(x["name"] or node.id, f.sdf, x["color"], group=x["group"] or None,
                  bounds=f.bounds, visible=x["visible"])
        scene.bodies[-1].bounds_now = f.now
    now = [b for _, x in outputs if (b := x["body"].now()) is not None]
    if now:  # where the bodies are at the current values: the viewer frames this
        scene.extra["view_bounds"] = [np.min([b[0] for b in now], axis=0).tolist(),
                                      np.max([b[1] for b in now], axis=0).tolist()]
    return scene


# ---------------------------------------------------------------- browser code generation
#
# The viewer builds the shader of an edited graph itself. Each block's GLSL is
# recorded once here by running its Python emitter on placeholders ("tracing"),
# so the browser assembles the same code without a second implementation:
#   {"ops": [["var", "float", "length($p) - ${radius}"], ["sub", "body", "$0"], ...], "out": "$1"}
# "$p" is the point the block is evaluated at, "$k" the result of op k,
# "${name}" a scalar input ("${center.x}" a vector component), and a "sub" op
# evaluates the body wired to an input at a point.

class _Hole(SDF):
    def __init__(self, name):
        self.name = name


class _ScalarHole(Expr):
    def __init__(self, name):
        self.name = name

    def glsl(self):
        return "${" + self.name + "}"


class _TraceCtx(Ctx):
    def __init__(self, holes):
        super().__init__()
        self.ops, self.holes = [], holes

    def var(self, kind, expr):
        self.ops.append(["var", kind, expr])
        return f"${len(self.ops) - 1}"

    def sub(self, node, p):
        key = (id(node), p)
        if key not in self.memo:
            if id(node) in self.holes:
                self.ops.append(["sub", self.holes[id(node)], p])
                r = f"${len(self.ops) - 1}"
            else:
                r = node.emit(self, p)
            self.memo[key] = (node, r)
        return self.memo[key][1]


def trace(sdf, holes=None):
    """GLSL template of ``sdf`` evaluated at "$p"; ``holes`` maps id(node) -> input name."""
    ctx = _TraceCtx(holes or {})
    return {"ops": ctx.ops, "out": ctx.sub(sdf, "$p")}


def _block_templates(blk: Block):
    """Templates of a block's outputs, per combination of its choice inputs ("" if none).

    {combo: {port: template}}: a body output is a traced shader program; a value
    output (vector, plane, frame, vertebra) holds the GLSL expressions of its
    components, written with the same "${input}" placeholders.
    """
    choices = [p for p in blk.inputs if p.type == "choice"]
    combos = [{}]
    for p in choices:
        combos = [{**c, p.name: v} for c in combos for v in p.choices]
    for inp, parts in blk.optional.items():  # one variant per present/absent part
        for part in parts:
            combos = [{**c, f"{inp}.{part}": v} for c in combos for v in (1, 0)]
    out = {}
    for combo in combos:
        holes, x = {}, {}
        for p in blk.inputs:
            if p.type in ("body", "bone"):
                h = _Hole(p.name)
                holes[id(h)] = p.name
                x[p.name] = Field(h, None)
            elif p.type in ("plane", "frame", "vertebra", "vec3"):
                x[p.name] = _value_hole(p.type, p.name)
            elif p.type == "scalar":
                x[p.name] = _ScalarHole(p.name)
            elif p.type == "choice":
                x[p.name] = combo[p.name]
            else:
                x[p.name] = p.default
        for key, present in combo.items():
            inp, _, part = key.partition(".")
            if part and not present:
                setattr(x[inp], part, None)
        node = Node("trace", blk, {}, blk.title)
        res = blk.build(x, node)
        ports = {}
        for op in blk.outputs:
            v = res[op.name]
            ports[op.name] = trace(v.sdf, holes) if op.type == "body" else _value_glsl(v)
        out[",".join(f"{k}={v}" for k, v in combo.items())] = ports
    return out


def _value_hole(kind, name):
    if kind == "vec3":
        return _vec_hole(name)
    if kind == "plane":
        return PlaneV(*(_vec_hole(f"{name}.{k}") for k in ("point", "normal")))
    if kind == "frame":
        return FrameV(*(_vec_hole(f"{name}.{k}") for k in ("origin", "x", "y", "z")))
    return VertebraV(_value_hole("frame", f"{name}.frame"), _value_hole("plane", f"{name}.sup"),
                     _value_hole("plane", f"{name}.inf"))


def _value_glsl(v):
    """GLSL expressions of a value output's scalar components (nested like the value)."""
    if v is None:
        return None
    if isinstance(v, Vec3):
        return [c.glsl() for c in v.c]
    if isinstance(v, Expr):
        return v.glsl()
    return {k: _value_glsl(getattr(v, k)) for k in v.__dataclass_fields__}


def _vec_hole(name):
    return Vec3(*(_ScalarHole(f"{name}.{a}") for a in "xyz"))


# Handled in viewer/graph.js: value plumbing and per-bone data from the library.
BROWSER_BUILT = {"number", "vector", "bone", "smooth", "output", "landmarks", "plane", "plane_parts", "frame_parts",
                 "vertebra_parts", "spinopelvic", "align", "spine_curve", "tps_warp"}


def browser_manifest():
    """Block manifest plus GLSL templates and the shared GLSL library, for viewer/graph.js."""
    m = manifest()
    for b in m["blocks"]:
        if b["name"] not in BROWSER_BUILT:
            b["glsl"] = _block_templates(BLOCKS[b["name"]])
    m["glsl_lib"] = GLSL_LIB
    return m


class Library:
    """Every bone file next to a graph's bones, in the graph's frame, packed in one atlas.

    The viewer can add any of these bones to the graph. Smoothing pyramids are
    built for all of them, so a Smooth block works on any bone without Python.
    """

    def __init__(self, g: dict, base):
        self.base = Path(base)
        files = _bone_files(g.get("nodes") or {})
        self.ctx = _Context(self.base, files, whole_dirs=True)
        self.offset = _frame_offset(g, self.ctx)
        self.ctx.shift(self.offset)
        grids = list(self.ctx._grids.values())
        levels = [lv for gr in grids for lv in _pyramid(gr, GridSmooth.LEVELS)[0]]
        self.atlas = pack_grids(grids + levels)  # None without bones
        self.keys = {}
        for f, path in zip(files, self.ctx.graph_paths):
            self.keys.setdefault(path, f)
        for path in self.ctx._grids:
            self.keys.setdefault(path, Path(os.path.relpath(path, self.base.resolve())).as_posix())

    def to_json(self, atlas_url=None):
        bones = {}
        for path, g in self.ctx._grids.items():
            lo, hi = g.bounds()
            sm = GridSmooth(g, _ScalarHole("sigma"), _ScalarHole("preserve"))
            bones[self.keys[path]] = {
                "name": self.ctx.names[path], "kind": self.ctx.kinds[path],
                "bounds": [(lo - 0.5).tolist(), (hi + 0.5).tolist()],
                "bone": trace(g), "smooth": trace(sm, {id(g): "bone"}),
            }
            lm = self.ctx.landmarks(path)
            if lm:
                sdf, (blo, bhi) = landmark_markers(lm)
                bones[self.keys[path]]["landmarks"] = {
                    "origin": lm["origin"].tolist(), "axes": lm["axes"].tolist(),
                    **{k: {"point": lm[k][0].tolist(), "normal": lm[k][1].tolist()} for k in ("sup", "inf") if k in lm},
                    "markers": trace(sdf), "markers_bounds": [blo.tolist(), bhi.tolist()]}
            if self.ctx.names[path] == "sacrum" and (spine := self.ctx.spine(path)) is not None:
                bones[self.keys[path]]["spine"] = spine
        a = self.atlas
        atlas = None if a is None else {"size": [a.shape[2], a.shape[1], a.shape[0]], "url": atlas_url}
        return {"frame_offset": [float(x) for x in self.offset], "attribution": self.ctx.attribution,
                "atlas": atlas, "bones": bones}

    def atlas_gz(self):
        if self.atlas is None:
            return None
        return gzip.compress(self.atlas.tobytes(), compresslevel=6, mtime=0)


def load_graph(path) -> Scene:
    path = Path(path)
    return build_graph(read_graph(path), base=path.parent)


def save_graph(g: dict, path):
    """Write a graph with one node per line (readable diffs)."""
    head = {k: v for k, v in g.items() if k != "nodes"}
    lines = [json.dumps(head)[:-1] + ', "nodes": {']
    items = list(g.get("nodes", {}).items())
    for i, (nid, raw) in enumerate(items):
        lines.append(f"  {json.dumps(nid)}: {json.dumps(raw)}" + ("," if i < len(items) - 1 else ""))
    lines.append("}}")
    Path(path).write_text("\n".join(lines) + "\n")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("blocks", help="print the block manifest (JSON)")
    c = sub.add_parser("check", help="validate a graph and list its bodies and parameters")
    c.add_argument("graph")
    a = ap.parse_args(argv)
    if a.cmd == "blocks":
        print(json.dumps(manifest(), indent=1))
        return
    scene = load_graph(a.graph)
    print(f"{scene.name}: {len(scene.bodies)} bodies, {len(scene.params())} parameters")
    for b in scene.bodies:
        print(f"  {b.name:<16} group {b.group}")


if __name__ == "__main__":
    main()
