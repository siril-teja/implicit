"""Signed distance field (SDF) node graph.

Convention: f(p) < 0 inside, f(p) = 0 on the surface, f(p) > 0 outside.
Units are whatever the scene uses (mm for the spine examples).
Axes for the spine examples: x = lateral (right), y = anterior, z = cranial.

Every node has two backends:
  * ``emit`` writes GLSL for real-time ray marching in the browser;
  * ``eval`` evaluates on an (N, 3) NumPy array for meshing / export.
Both must implement the same math; ``tests/test_sdf.py`` checks they agree.

Primitives marked "exact" return true Euclidean distance. Booleans, plane cuts
and elliptic cross-sections return a conservative bound (|f| <= true distance
near the surface), which is sufficient for ray marching and iso-surfacing.
"""

from __future__ import annotations

import numpy as np

from .expr import Expr, Vec3, glsl_float, vec3, wrap


class Ctx:
    """GLSL emission context: straight-line SSA code, one variable per node."""

    def __init__(self):
        self.lines: list[str] = []
        self.n = 0

    def var(self, kind, expr):
        self.n += 1
        name = f"{'p' if kind == 'vec3' else 'd'}{self.n}"
        self.lines.append(f"    {kind} {name} = {expr};")
        return name


class SDF:
    def emit(self, ctx: Ctx, p: str) -> str:
        raise NotImplementedError

    def eval(self, p: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def children(self):
        return []

    def exprs(self):
        return []

    def params(self):
        out = [q for e in self.exprs() for q in e.params()]
        for c in self.children():
            out += c.params()
        return out

    # booleans
    def __or__(self, o): return Union(self, o)
    def __and__(self, o): return Intersect(self, o)
    def __sub__(self, o): return Difference(self, o)

    def smooth_union(self, o, k): return Union(self, o, k=k)
    def smooth_intersect(self, o, k): return Intersect(self, o, k=k)
    def smooth_difference(self, o, k): return Difference(self, o, k=k)

    # transforms
    def translate(self, t): return Translate(self, t)

    def rotate(self, axis, angle_deg, center=(0, 0, 0)):
        c = vec3(center)
        return Translate(Rotate(Translate(self, -c), axis, angle_deg), c)

    def rotate_x(self, a, center=(0, 0, 0)): return self.rotate("x", a, center)
    def rotate_y(self, a, center=(0, 0, 0)): return self.rotate("y", a, center)
    def rotate_z(self, a, center=(0, 0, 0)): return self.rotate("z", a, center)
    def scale(self, s): return Scale(self, s)
    def mirror_x(self): return MirrorX(self)

    # offsets
    def offset(self, d): return Offset(self, d)
    def shell(self, t): return Shell(self, t)


def _v(e: Expr) -> float:
    return e.value()


def _vv(v: Vec3) -> np.ndarray:
    return np.array(v.value(), dtype=float)


def _len(a, axis=-1):
    return np.sqrt(np.sum(a * a, axis=axis))


# ---------------------------------------------------------------- primitives

class Sphere(SDF):
    """Exact."""

    def __init__(self, r):
        self.r = wrap(r)

    def exprs(self): return [self.r]

    def emit(self, ctx, p):
        return ctx.var("float", f"length({p}) - {self.r.glsl()}")

    def eval(self, p):
        return _len(p) - _v(self.r)


class Box(SDF):
    """Exact. ``half`` are half extents; ``round`` rounds edges."""

    def __init__(self, half, round=0.0):
        self.half, self.round = vec3(half), wrap(round)

    def exprs(self): return [self.half, self.round]

    def emit(self, ctx, p):
        q = ctx.var("vec3", f"abs({p}) - {self.half.glsl()} + {self.round.glsl()}")
        return ctx.var(
            "float",
            f"length(max({q}, 0.0)) + min(max({q}.x, max({q}.y, {q}.z)), 0.0)"
            f" - {self.round.glsl()}",
        )

    def eval(self, p):
        r = _v(self.round)
        q = np.abs(p) - _vv(self.half) + r
        return _len(np.maximum(q, 0)) + np.minimum(q.max(axis=-1), 0) - r


def _extrude_glsl(ctx, d2, pz, h):
    w = ctx.var("vec3", f"vec3({d2}, abs({pz}) - {h}, 0.0)")
    return ctx.var(
        "float",
        f"min(max({w}.x, {w}.y), 0.0) + length(max({w}.xy, 0.0))",
    )


def _extrude_np(d2, pz, h):
    w = np.stack([d2, np.abs(pz) - h], axis=-1)
    return np.minimum(w.max(axis=-1), 0) + _len(np.maximum(w, 0))


class Cylinder(SDF):
    """Exact. Axis along z, centred at origin, half height ``h``."""

    def __init__(self, r, h):
        self.r, self.h = wrap(r), wrap(h)

    def exprs(self): return [self.r, self.h]

    def emit(self, ctx, p):
        d2 = ctx.var("float", f"length({p}.xy) - {self.r.glsl()}")
        return _extrude_glsl(ctx, d2, f"{p}.z", self.h.glsl())

    def eval(self, p):
        return _extrude_np(_len(p[:, :2]) - _v(self.r), p[:, 2], _v(self.h))


class EllipticCylinder(SDF):
    """Bound (exact if a == b). Semi-axes a (x), b (y); axis z, half height h."""

    def __init__(self, a, b, h):
        self.a, self.b, self.h = wrap(a), wrap(b), wrap(h)

    def exprs(self): return [self.a, self.b, self.h]

    def emit(self, ctx, p):
        a, b = self.a.glsl(), self.b.glsl()
        d2 = ctx.var(
            "float",
            f"(length({p}.xy / vec2({a}, {b})) - 1.0) * min({a}, {b})",
        )
        return _extrude_glsl(ctx, d2, f"{p}.z", self.h.glsl())

    def eval(self, p):
        a, b = _v(self.a), _v(self.b)
        d2 = (_len(p[:, :2] / np.array([a, b])) - 1.0) * min(a, b)
        return _extrude_np(d2, p[:, 2], _v(self.h))


class Capsule(SDF):
    """Exact. Segment a-b swept by radius r."""

    def __init__(self, a, b, r):
        self.a, self.b, self.r = vec3(a), vec3(b), wrap(r)

    def exprs(self): return [self.a, self.b, self.r]

    def emit(self, ctx, p):
        a, b = self.a.glsl(), self.b.glsl()
        pa = ctx.var("vec3", f"{p} - {a}")
        ba = ctx.var("vec3", f"{b} - {a}")
        h = ctx.var("float", f"clamp(dot({pa}, {ba}) / dot({ba}, {ba}), 0.0, 1.0)")
        return ctx.var("float", f"length({pa} - {ba} * {h}) - {self.r.glsl()}")

    def eval(self, p):
        a, b = _vv(self.a), _vv(self.b)
        pa, ba = p - a, b - a
        h = np.clip(pa @ ba / (ba @ ba), 0, 1)
        return _len(pa - h[:, None] * ba) - _v(self.r)


class HalfSpace(SDF):
    """Exact. Solid on the side opposite to ``normal``; boundary through ``point``."""

    def __init__(self, point, normal):
        self.point, self.normal = vec3(point), vec3(normal)

    def exprs(self): return [self.point, self.normal]

    def emit(self, ctx, p):
        return ctx.var(
            "float",
            f"dot({p} - {self.point.glsl()}, normalize({self.normal.glsl()}))",
        )

    def eval(self, p):
        n = _vv(self.normal)
        return (p - _vv(self.point)) @ (n / np.linalg.norm(n))


# ---------------------------------------------------------------- booleans

def _smin_glsl(ctx, a, b, k):
    h = ctx.var("float", f"clamp(0.5 + 0.5 * ({b} - {a}) / max({k}, 1e-6), 0.0, 1.0)")
    return ctx.var("float", f"mix({b}, {a}, {h}) - {k} * {h} * (1.0 - {h})")


def _smin_np(a, b, k):
    h = np.clip(0.5 + 0.5 * (b - a) / max(k, 1e-6), 0, 1)
    return b + (a - b) * h - k * h * (1 - h)


class _Bool(SDF):
    def __init__(self, a, b, k=0.0):
        self.a, self.b, self.k = a, b, wrap(k)
        self.smooth = not (isinstance(k, (int, float)) and k == 0)

    def children(self): return [self.a, self.b]
    def exprs(self): return [self.k]


class Union(_Bool):
    def emit(self, ctx, p):
        a, b = self.a.emit(ctx, p), self.b.emit(ctx, p)
        if self.smooth:
            return _smin_glsl(ctx, a, b, self.k.glsl())
        return ctx.var("float", f"min({a}, {b})")

    def eval(self, p):
        a, b = self.a.eval(p), self.b.eval(p)
        return _smin_np(a, b, _v(self.k)) if self.smooth else np.minimum(a, b)


class Intersect(_Bool):
    def emit(self, ctx, p):
        a, b = self.a.emit(ctx, p), self.b.emit(ctx, p)
        if self.smooth:
            m = _smin_glsl(ctx, f"(-{a})", f"(-{b})", self.k.glsl())
            return ctx.var("float", f"-{m}")
        return ctx.var("float", f"max({a}, {b})")

    def eval(self, p):
        a, b = self.a.eval(p), self.b.eval(p)
        return -_smin_np(-a, -b, _v(self.k)) if self.smooth else np.maximum(a, b)


class Difference(_Bool):
    def emit(self, ctx, p):
        a, b = self.a.emit(ctx, p), self.b.emit(ctx, p)
        if self.smooth:
            m = _smin_glsl(ctx, f"(-{a})", b, self.k.glsl())
            return ctx.var("float", f"-{m}")
        return ctx.var("float", f"max({a}, -{b})")

    def eval(self, p):
        a, b = self.a.eval(p), self.b.eval(p)
        return -_smin_np(-a, b, _v(self.k)) if self.smooth else np.maximum(a, -b)


def union(*xs, k=0.0):
    out = xs[0]
    for x in xs[1:]:
        out = Union(out, x, k=k)
    return out


# ---------------------------------------------------------------- transforms

class Translate(SDF):
    def __init__(self, child, t):
        self.child, self.t = child, vec3(t)

    def children(self): return [self.child]
    def exprs(self): return [self.t]

    def emit(self, ctx, p):
        return self.child.emit(ctx, ctx.var("vec3", f"{p} - {self.t.glsl()}"))

    def eval(self, p):
        return self.child.eval(p - _vv(self.t))


_ROT_AXES = {"x": (1, 2), "y": (2, 0), "z": (0, 1)}


class Rotate(SDF):
    """Rotate the child by +angle (deg, right-hand rule) about an axis through the origin."""

    def __init__(self, child, axis, angle_deg):
        if axis not in _ROT_AXES:
            raise ValueError("axis must be 'x', 'y' or 'z'")
        self.child, self.axis, self.angle = child, axis, wrap(angle_deg)

    def children(self): return [self.child]
    def exprs(self): return [self.angle]

    def emit(self, ctx, p):
        # Sample the child at R(-angle) p.
        return self.child.emit(
            ctx, ctx.var("vec3", f"rot{self.axis.upper()}({p}, -radians({self.angle.glsl()}))")
        )

    def eval(self, p):
        i, j = _ROT_AXES[self.axis]
        a = -np.radians(_v(self.angle))
        c, s = np.cos(a), np.sin(a)
        q = p.copy()
        q[:, i] = c * p[:, i] - s * p[:, j]
        q[:, j] = s * p[:, i] + c * p[:, j]
        return self.child.eval(q)


class Scale(SDF):
    """Uniform scale (keeps the field a distance)."""

    def __init__(self, child, s):
        self.child, self.s = child, wrap(s)

    def children(self): return [self.child]
    def exprs(self): return [self.s]

    def emit(self, ctx, p):
        s = self.s.glsl()
        d = self.child.emit(ctx, ctx.var("vec3", f"{p} / {s}"))
        return ctx.var("float", f"{d} * {s}")

    def eval(self, p):
        s = _v(self.s)
        return self.child.eval(p / s) * s


class MirrorX(SDF):
    """Bilateral symmetry about the sagittal plane x = 0 (model the +x side only)."""

    def __init__(self, child):
        self.child = child

    def children(self): return [self.child]

    def emit(self, ctx, p):
        return self.child.emit(ctx, ctx.var("vec3", f"vec3(abs({p}.x), {p}.yz)"))

    def eval(self, p):
        q = p.copy()
        q[:, 0] = np.abs(q[:, 0])
        return self.child.eval(q)


# ---------------------------------------------------------------- offsets

class Offset(SDF):
    """Grow (d > 0) or shrink (d < 0) the solid by d."""

    def __init__(self, child, d):
        self.child, self.d = child, wrap(d)

    def children(self): return [self.child]
    def exprs(self): return [self.d]

    def emit(self, ctx, p):
        return ctx.var("float", f"{self.child.emit(ctx, p)} - {self.d.glsl()}")

    def eval(self, p):
        return self.child.eval(p) - _v(self.d)


class Shell(SDF):
    """Inward shell of thickness t, e.g. a cortical wall: max(f, -(f + t))."""

    def __init__(self, child, t):
        self.child, self.t = child, wrap(t)

    def children(self): return [self.child]
    def exprs(self): return [self.t]

    def emit(self, ctx, p):
        d = self.child.emit(ctx, p)
        return ctx.var("float", f"max({d}, -({d} + {self.t.glsl()}))")

    def eval(self, p):
        d = self.child.eval(p)
        return np.maximum(d, -(d + _v(self.t)))


GLSL_LIB = """
vec3 rotX(vec3 p, float a) { float c = cos(a), s = sin(a); return vec3(p.x, c*p.y - s*p.z, s*p.y + c*p.z); }
vec3 rotY(vec3 p, float a) { float c = cos(a), s = sin(a); return vec3(c*p.x + s*p.z, p.y, -s*p.x + c*p.z); }
vec3 rotZ(vec3 p, float a) { float c = cos(a), s = sin(a); return vec3(c*p.x - s*p.y, s*p.x + c*p.y, p.z); }
"""

__all__ = [
    "SDF", "Sphere", "Box", "Cylinder", "EllipticCylinder", "Capsule", "HalfSpace",
    "Union", "Intersect", "Difference", "union", "Translate", "Rotate", "Scale",
    "MirrorX", "Offset", "Shell", "Ctx", "GLSL_LIB", "glsl_float",
]
