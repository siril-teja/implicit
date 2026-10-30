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
also accepted wherever a body is), ``scalar``, ``vec3``, and static settings
``number``, ``text``, ``color``, ``bool``, ``choice``, ``path``.

    python -m implicit.graph blocks             # block manifest (JSON) for the editor
    python -m implicit.graph check GRAPH.json   # validate and summarise
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .expr import BinOp, Const, Expr, Func, Param, Vec3
from .scene import Scene
from .sdf import (Box, Cylinder, Difference, Grid, HalfSpace, Intersect, Offset, Rotate,
                  Shell, Sphere, Translate, Union, smooth)

FORMAT, VERSION = "implicit.graph", 1

LIVE = {"scalar", "vec3"}                 # literal values become parameters
WIRED = {"body", "bone", "scalar", "vec3"}  # types a wire can carry
STATIC = {"number", "text", "color", "bool", "choice", "path"}
ACCEPTS = {"body": {"body", "bone"}, "bone": {"bone"}, "scalar": {"scalar"}, "vec3": {"vec3"}}


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

    def to_json(self):
        d = {"name": self.name, "type": self.type, "label": self.label or self.name}
        for k in ("default", "lo", "hi", "step"):
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

    def input(self, name):
        return next(p for p in self.inputs if p.name == name)

    def to_json(self):
        return {"name": self.name, "title": self.title, "category": self.category, "doc": self.doc,
                "inputs": [p.to_json() for p in self.inputs],
                "outputs": [p.to_json() for p in self.outputs]}


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
    """A body on a wire: its SDF and an axis-aligned box that contains it (or None)."""
    sdf: object
    bounds: tuple | None


def _interval(e: Expr):
    """(lo, hi) of an expression over its parameters' slider ranges, or None if unknown."""
    if isinstance(e, Param):
        return min(e.lo, e.v), max(e.hi, e.v)
    if isinstance(e, Const) or e.is_const():
        v = e.value()
        return v, v
    if isinstance(e, BinOp):
        a, b = _interval(e.a), _interval(e.b)
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


def _upper(e: Expr):
    iv = _interval(e)
    return None if iv is None else iv[1]


def _vec_interval(v: Vec3):
    ivs = [_interval(c) for c in v.c]
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
       [S("x", 0, -100, 100, 0.5, "mm"), S("y", 0, -100, 100, 0.5, "mm"), S("z", 0, -100, 100, 0.5, "mm")],
       [Port("vector", "vec3")])
def _vector(x, node):
    """Point or direction from three scalars."""
    return {"vector": Vec3(x["x"], x["y"], x["z"])}


@block("bone", "Bone", "Inputs", [Port("file", "path", label="Bone file (.npz)")], [Port("bone", "bone")])
def _bone(x, node):
    """One implicit bone converted from a segmentation (implicit.anatomy.convert)."""
    g = node.ctx.grid(x["file"])
    return {"bone": Field(g, _grow(g.bounds(), 0.5))}


@block("sphere", "Sphere", "Shapes",
       [S("radius", 10, 0.1, 100, 0.1, "mm"), V("center", (0, 0, 0), -200, 200, 0.5, "mm")],
       [Port("body", "body")])
def _sphere(x, node):
    """Sphere of a given radius."""
    c, r = x["center"], x["radius"]
    civ, riv = _vec_interval(c), _upper(r)
    b = None if civ is None or riv is None else (civ[0] - riv, civ[1] + riv)
    return {"body": Field(Translate(Sphere(r), c), b)}


@block("box", "Box", "Shapes",
       [V("size", (20, 20, 20), 0.1, 200, 0.5, "mm", "Size (full extents)"),
        V("center", (0, 0, 0), -200, 200, 0.5, "mm"), S("round", 0, 0, 20, 0.1, "mm", "Edge radius")],
       [Port("body", "body")])
def _box(x, node):
    """Axis-aligned box with optional rounded edges."""
    half = x["size"] * 0.5
    civ, hiv = _vec_interval(x["center"]), _vec_interval(half)
    b = None if civ is None or hiv is None else (civ[0] - hiv[1], civ[1] + hiv[1])
    return {"body": Field(Translate(Box(half, x["round"]), x["center"]), b)}


@block("cylinder", "Cylinder", "Shapes",
       [S("radius", 10, 0.1, 100, 0.1, "mm"), S("height", 20, 0.1, 200, 0.5, "mm"),
        V("center", (0, 0, 0), -200, 200, 0.5, "mm")],
       [Port("body", "body")])
def _cylinder(x, node):
    """Cylinder along z (cranial)."""
    c = x["center"]
    civ, r, h = _vec_interval(c), _upper(x["radius"]), _upper(x["height"])
    b = None if civ is None or r is None or h is None else (
        civ[0] - [r, r, h / 2], civ[1] + [r, r, h / 2])
    return {"body": Field(Translate(Cylinder(x["radius"], x["height"] * 0.5), c), b)}


def _boolean(cls, bounds):
    def build(x, node):
        a, b, k = x["a"], x["b"], x["blend"]
        bb = bounds(a.bounds, b.bounds)
        smooth_k = not (k.is_const() and k.value() == 0)
        return {"body": Field(cls(a.sdf, b.sdf, k if smooth_k else 0.0), _grow(bb, _upper(k)) if smooth_k else bb)}
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


@block("translate", "Move", "Transforms",
       [BODY(), V("offset", (0, 0, 0), -100, 100, 0.5, "mm")], [Port("body", "body")])
def _translate(x, node):
    """Move a body by an offset vector."""
    b, t = x["body"], x["offset"]
    tiv = _vec_interval(t)
    bb = None if b.bounds is None or tiv is None else (b.bounds[0] + tiv[0], b.bounds[1] + tiv[1])
    return {"body": Field(Translate(b.sdf, t), bb)}


@block("rotate", "Rotate", "Transforms",
       [BODY(), Port("axis", "choice", "x", choices=("x", "y", "z"), label="Axis"),
        S("angle", 0, -45, 45, 0.5, "deg"), V("center", (0, 0, 0), -200, 200, 0.5, "mm")],
       [Port("body", "body")])
def _rotate(x, node):
    """Rotate a body about an axis through a centre point (right-hand rule)."""
    b, c = x["body"], x["center"]
    sdf = Translate(Rotate(Translate(b.sdf, -c), x["axis"], x["angle"]), c)
    civ = _vec_interval(c)
    bb = None
    if b.bounds is not None and civ is not None:
        # Any rotation stays inside the sphere about the centre through the farthest corner.
        corners = np.array([[b.bounds[i][0], b.bounds[j][1], b.bounds[k][2]]
                            for i in (0, 1) for j in (0, 1) for k in (0, 1)])
        far = np.maximum(np.abs(corners - civ[0]), np.abs(corners - civ[1]))  # any centre in range
        r = np.linalg.norm(far, axis=1).max()
        bb = (civ[0] - r, civ[1] + r)
    return {"body": Field(sdf, bb)}


@block("offset", "Offset", "Modify", [BODY(), S("distance", 0, -10, 10, 0.1, "mm")], [Port("body", "body")])
def _offset(x, node):
    """Grow (> 0) or shrink (< 0) a body by a distance."""
    b, d = x["body"], x["distance"]
    return {"body": Field(Offset(b.sdf, d), _grow(b.bounds, _upper(d)))}


@block("shell", "Shell", "Modify", [BODY(), S("thickness", 1, 0.1, 10, 0.1, "mm")], [Port("body", "body")])
def _shell(x, node):
    """Hollow a body, keeping an inner wall of the given thickness (e.g. a cortex)."""
    b = x["body"]
    return {"body": Field(Shell(b.sdf, x["thickness"]), b.bounds)}


@block("plane_cut", "Plane cut", "Modify",
       [BODY(), V("point", (0, 0, 0), -200, 200, 0.5, "mm"),
        V("normal", (0, 0, 1), -1, 1, 0.01, "", "Normal (removed side)")],
       [Port("body", "body")])
def _plane_cut(x, node):
    """Remove the part of a body on the normal side of a plane."""
    b = x["body"]
    return {"body": Field(Intersect(b.sdf, HalfSpace(x["point"], x["normal"])), b.bounds)}


@block("smooth", "Smooth", "Modify",
       [Port("bone", "bone", label="Bone"), S("sigma", 0, 0, 4, 0.1, "mm", "Smoothing σ"),
        S("preserve", 1, 0, 1, 1, "", "Keep volume (1 = on)")],
       [Port("body", "body")])
def _smooth(x, node):
    """Gaussian smoothing of a bone surface, width σ; optionally keeps the bone volume."""
    b = x["bone"]
    return {"body": Field(smooth(b.sdf, x["sigma"], x["preserve"]), b.bounds)}


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
    def __init__(self, base: Path, files):
        self.base = base
        self._grids = {}
        self.kinds = {}
        from .anatomy.convert import load_bone
        for f in files:
            path = self.resolve(f)
            if path not in self._grids:
                if not path.exists():
                    raise GraphError(f"bone file not found: {f} ({path})")
                g, meta = load_bone(path)
                self._grids[path] = g
                self.kinds[path] = meta.get("kind", "")
                self.attribution = getattr(self, "attribution", "") or meta.get("attribution", "")

    def resolve(self, f):
        p = Path(f)
        return (p if p.is_absolute() else self.base / p).resolve()

    def grid(self, f) -> Grid:
        return self._grids[self.resolve(f)]

    def auto_offset(self):
        """Display frame: bones (not references) centred at the origin, whole mm."""
        bs = [g.bounds() for p, g in self._grids.items() if self.kinds[p] != "reference"]
        bs = bs or [g.bounds() for g in self._grids.values()]
        if not bs:
            return np.zeros(3)
        lo = np.min([b[0] - 0.5 for b in bs], axis=0)
        hi = np.max([b[1] + 0.5 for b in bs], axis=0)
        return np.round((lo + hi) / 2)

    def shift(self, offset):
        for g in self._grids.values():
            g.origin = g.origin - offset


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
    if port.type == "scalar":
        v = float(v)
        lo = min(port.lo if port.lo is not None else v, v)
        hi = max(port.hi if port.hi is not None else v, v)
        return Param(f"{node.id}_{port.name}", v, lo, hi, port.step, label, port.unit)
    if port.type == "vec3":
        if not (isinstance(v, (list, tuple)) and len(v) == 3):
            raise GraphError(f"{node.id}.{port.name}: expected [x, y, z], got {v!r}")
        comps = []
        for axis, c in zip("xyz", v):
            c = float(c)
            lo = min(port.lo if port.lo is not None else c, c)
            hi = max(port.hi if port.hi is not None else c, c)
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


def build_graph(g: dict, base=".") -> Scene:
    """Build a Scene from a graph dict (``base``: directory that relative file paths start from)."""
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

    files = [n.raw.get("in", {}).get("file") for n in nodes.values() if n.block.name == "bone"]
    if any(f is None for f in files):
        raise GraphError("every bone block needs a file")
    ctx = _Context(Path(base), files)
    frame = g.get("frame", "auto")
    offset = ctx.auto_offset() if frame == "auto" else np.asarray(frame.get("offset", (0, 0, 0)), float)
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
            elif port.type in ("body", "bone"):
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
    scene.attribution = getattr(ctx, "attribution", "")
    for node, x in outputs:
        f = x["body"]
        scene.add(x["name"] or node.id, f.sdf, x["color"], group=x["group"] or None,
                  bounds=f.bounds, visible=x["visible"])
    return scene


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
