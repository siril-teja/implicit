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
        self.memo: dict = {}

    def sub(self, node, p):
        """Emit ``node`` at point variable ``p`` once; reuse the result for shared subgraphs."""
        key = (id(node), p)
        if key not in self.memo:
            self.memo[key] = (node, node.emit(self, p))
        return self.memo[key][1]

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


def _unit_glsl(v: Vec3):
    """GLSL for normalize(v); folded to a literal when v has no parameters."""
    if not v.is_const():
        return f"normalize({v.glsl()})"
    n = np.array(v.value(), dtype=float)
    n /= np.linalg.norm(n)
    return "vec3(" + ", ".join(glsl_float(x) for x in n) + ")"


class HalfSpace(SDF):
    """Exact. Solid on the side opposite to ``normal``; boundary through ``point``."""

    def __init__(self, point, normal):
        self.point, self.normal = vec3(point), vec3(normal)

    def exprs(self): return [self.point, self.normal]

    def emit(self, ctx, p):
        return ctx.var(
            "float",
            f"dot({p} - {self.point.glsl()}, {_unit_glsl(self.normal)})",
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
        a, b = ctx.sub(self.a, p), ctx.sub(self.b, p)
        if self.smooth:
            return _smin_glsl(ctx, a, b, self.k.glsl())
        return ctx.var("float", f"min({a}, {b})")

    def eval(self, p):
        a, b = self.a.eval(p), self.b.eval(p)
        return _smin_np(a, b, _v(self.k)) if self.smooth else np.minimum(a, b)


class Intersect(_Bool):
    def emit(self, ctx, p):
        a, b = ctx.sub(self.a, p), ctx.sub(self.b, p)
        if self.smooth:
            m = _smin_glsl(ctx, f"(-{a})", f"(-{b})", self.k.glsl())
            return ctx.var("float", f"-{m}")
        return ctx.var("float", f"max({a}, {b})")

    def eval(self, p):
        a, b = self.a.eval(p), self.b.eval(p)
        return -_smin_np(-a, -b, _v(self.k)) if self.smooth else np.maximum(a, b)


class Difference(_Bool):
    def emit(self, ctx, p):
        a, b = ctx.sub(self.a, p), ctx.sub(self.b, p)
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
        return ctx.sub(self.child, ctx.var("vec3", f"{p} - {self.t.glsl()}"))

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
        return ctx.sub(
            self.child, ctx.var("vec3", f"rot{self.axis.upper()}({p}, -radians({self.angle.glsl()}))")
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
        d = ctx.sub(self.child, ctx.var("vec3", f"{p} / {s}"))
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
        return ctx.sub(self.child, ctx.var("vec3", f"vec3(abs({p}.x), {p}.yz)"))

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
        return ctx.var("float", f"{ctx.sub(self.child, p)} - {self.d.glsl()}")

    def eval(self, p):
        return self.child.eval(p) - _v(self.d)


class Shell(SDF):
    """Inward shell of thickness t, e.g. a cortical wall: max(f, -(f + t))."""

    def __init__(self, child, t):
        self.child, self.t = child, wrap(t)

    def children(self): return [self.child]
    def exprs(self): return [self.t]

    def emit(self, ctx, p):
        d = ctx.sub(self.child, p)
        return ctx.var("float", f"max({d}, -({d} + {self.t.glsl()}))")

    def eval(self, p):
        d = self.child.eval(p)
        return np.maximum(d, -(d + _v(self.t)))


# ---------------------------------------------------------------- sampled fields

QSTEP = 0.1          # mm per uint8 level
QZERO = 128          # uint8 level of the zero iso-surface
QRANGE = QSTEP * 127  # |distances| beyond this are clamped


class Grid(SDF):
    """Sampled SDF on an axis-aligned grid (e.g. a bone from a CT segmentation).

    Values are quantised to uint8 (QSTEP mm per level, clamped to +-QRANGE) so the
    browser (an R8 3D texture, hardware trilinear) and NumPy see identical data.
    Sample i sits at origin + i * spacing. Outside the grid the field is the larger
    of two lower bounds: the distance to the grid box plus the smallest boundary
    value (the object lies inside the box), and the value at the nearest grid point
    minus the distance to it. Both tend to that value at the box faces, so the field
    is continuous there.

    ``atlas_offset`` is assigned by Scene when packing grids into one texture.
    """

    def __init__(self, values, origin, spacing, name="grid"):
        v = np.asarray(values, dtype=float)
        self._init(np.clip(np.round(v / QSTEP) + QZERO, 0, 255).astype(np.uint8), origin, spacing, name)

    @classmethod
    def from_quantized(cls, q, origin, spacing, name="grid"):
        g = cls.__new__(cls)
        g._init(np.asarray(q, dtype=np.uint8), origin, spacing, name)
        return g

    def _init(self, q, origin, spacing, name):
        self.q = q
        self.origin = np.asarray(origin, dtype=float)
        self.spacing = float(spacing)
        self.name = name
        self.atlas_offset = None
        self.atlas_size = None
        border = np.concatenate([
            self.q[[0, -1]].ravel(), self.q[:, [0, -1]].ravel(), self.q[:, :, [0, -1]].ravel()
        ])
        self.margin = max(float(border.min()) - QZERO, 0.0) * QSTEP
        if self.margin <= 0:
            raise ValueError(f"grid {name!r}: object touches the grid boundary; pad it")

    @property
    def shape(self):
        return self.q.shape

    @property
    def values(self):
        return (self.q.astype(np.float32) - QZERO) * QSTEP

    def bounds(self):
        return self.origin, self.origin + self.spacing * (np.array(self.shape) - 1)

    def call(self, p):
        """GLSL expression for the field at point ``p``."""
        if self.atlas_offset is None:
            raise RuntimeError("Grid not packed; add it to a Scene before emitting GLSL")
        f = lambda xs: "vec3(" + ", ".join(glsl_float(x) for x in xs) + ")"
        n = np.array(self.shape) - 1
        return (f"gridSDF({p}, {f(self.origin)}, {glsl_float(1.0 / self.spacing)}, {f(n)}, "
                f"{f(np.array(self.atlas_offset) + 0.5)}, {glsl_float(self.spacing)}, {glsl_float(self.margin)})")

    def emit(self, ctx, p):
        return ctx.var("float", self.call(p))

    def eval(self, p):
        q = (p - self.origin) / self.spacing
        n = np.array(self.shape) - 1
        qc = np.clip(q, 0, n)
        i0 = np.minimum(np.floor(qc).astype(int), np.maximum(n - 1, 0))
        f = qc - i0
        v = self.values
        out = np.zeros(len(p))
        for dx in (0, 1):
            for dy in (0, 1):
                for dz in (0, 1):
                    w = ((f[:, 0] if dx else 1 - f[:, 0]) * (f[:, 1] if dy else 1 - f[:, 1])
                         * (f[:, 2] if dz else 1 - f[:, 2]))
                    out += w * v[np.minimum(i0[:, 0] + dx, n[0]), np.minimum(i0[:, 1] + dy, n[1]),
                                 np.minimum(i0[:, 2] + dz, n[2])]
        ext = _len(q - qc) * self.spacing
        return np.where(ext > 0, np.maximum(ext + self.margin, out - ext), out)  # see gridSDF


def _gauss_blur(v, sigma):
    """Separable Gaussian blur (sigma in voxels), edge values replicated."""
    r = int(np.ceil(3 * sigma))
    x = np.arange(-r, r + 1)
    k = np.exp(-0.5 * (x / sigma) ** 2)
    k /= k.sum()
    for ax in range(3):
        pad = [(0, 0)] * 3
        pad[ax] = (r, r)
        vp = np.pad(v, pad, mode="edge")
        out = np.zeros_like(v)
        for i, w in enumerate(k):
            sl = [slice(None)] * 3
            sl[ax] = slice(i, i + v.shape[ax])
            out += w * vp[tuple(sl)]
        v = out
    return v


class GridSmooth(SDF):
    """Gaussian-smoothed sampled body with a live smoothing width ``sigma`` (mm).

    A Gaussian blur of a distance field stays 1-Lipschitz, so the smoothed body
    ray-marches and meshes like any other. To stay interactive the blur is
    precomputed as a pyramid: level k (k = 1..K) is the grid blurred with
    sigma_k = 2^k * h and resampled at spacing 2^k * h (about +14 % memory).
    The field blends the two levels around ``sigma`` (linear in sigma up to
    sigma_1, then in log2 sigma), branch-free in GLSL and identical in NumPy.

    Blurring shrinks convex features. ``preserve`` (0..1) subtracts, per level,
    the offset that restores the unsmoothed volume (measured on level-0 samples).
    """

    LEVELS = 3

    def __init__(self, grid: Grid, sigma, preserve=1.0):
        self.grid, self.sigma, self.preserve = grid, wrap(sigma), wrap(preserve)
        self.levels, self.offsets = _pyramid(grid, self.LEVELS)
        self.sigmas = [0.0] + [grid.spacing * 2 ** k for k in range(1, self.LEVELS + 1)]

    def children(self): return [self.grid] + self.levels
    def exprs(self): return [self.sigma, self.preserve]

    def bounds(self):
        return self.grid.bounds()

    def _t_glsl(self):
        s1 = glsl_float(self.sigmas[1])
        s = f"max({self.sigma.glsl()}, 0.0)"
        return (f"({s} <= {s1} ? {s} / {s1} : min(1.0 + log2({s} / {s1}), "
                f"{glsl_float(float(self.LEVELS))}))")

    def _t(self):
        s, s1 = max(self.sigma.value(), 0.0), self.sigmas[1]
        return s / s1 if s <= s1 else min(1.0 + np.log2(s / s1), float(self.LEVELS))

    def emit(self, ctx, p):
        # At most two levels have non-zero weight; the others are not sampled. The
        # condition depends only on uniforms (same for every pixel), and the grid
        # lookup uses textureLod, so skipping it is well defined on every GPU.
        t = ctx.var("float", self._t_glsl())
        pres = self.preserve.glsl()
        terms = []
        for k, g in enumerate([self.grid] + self.levels):
            w = ctx.var("float", f"max(1.0 - abs({t} - {glsl_float(float(k))}), 0.0)")
            off = f" - {pres} * {glsl_float(self.offsets[k])}" if self.offsets[k] else ""
            if k == 0:  # the bone itself: may be shared with other uses of it
                d = ctx.sub(g, p)
                terms.append(f"{w} * ({d}{off})")
            else:
                terms.append(ctx.var("float", f"{w} > 0.0 ? {w} * ({g.call(p)}{off}) : 0.0"))
        return ctx.var("float", " + ".join(terms))

    def eval(self, p):
        t = self._t()
        pres = self.preserve.value()
        out = np.zeros(len(p))
        for k, g in enumerate([self.grid] + self.levels):
            w = max(1.0 - abs(t - k), 0.0)
            if w > 0:
                out += w * (g.eval(p) - pres * self.offsets[k])
        return out


_PYRAMIDS: dict = {}


def _pyramid(grid: Grid, levels):
    """Blurred, downsampled levels of a grid and their volume-restoring offsets (cached)."""
    key = (id(grid), levels)
    if key in _PYRAMIDS:
        return _PYRAMIDS[key]
    h = grid.spacing
    v = grid.values.astype(np.float32)
    out, offsets = [], [0.0]
    # Reference volume on every 2nd level-0 sample (accurate and cheap).
    sub = np.argwhere(np.ones(tuple((np.array(v.shape) + 1) // 2), bool)) * 2
    pts = grid.origin + h * sub
    n_inside = int((v[tuple(sub.T)] < 0).sum())
    sigma_prev = 0.0  # mm, blur already in v
    spacing = h
    for k in range(1, levels + 1):
        sigma_k = h * 2 ** k
        inc = np.sqrt(sigma_k ** 2 - sigma_prev ** 2) / spacing  # in voxels of v
        v = _gauss_blur(v, inc)[::2, ::2, ::2]
        spacing *= 2
        sigma_prev = sigma_k
        g = Grid(v, grid.origin, spacing, f"{grid.name}~{k}")
        out.append(g)
        fk = np.sort(g.eval(pts))
        offsets.append(float(fk[min(n_inside, len(fk) - 1)]) if n_inside else 0.0)
    _PYRAMIDS[key] = (out, offsets)
    return out, offsets


def smooth(grid: Grid, sigma, preserve=1.0) -> GridSmooth:
    """Gaussian smoothing of a sampled body with width ``sigma`` (mm, may be a Param)."""
    return GridSmooth(grid, sigma, preserve)


class GridSlot(SDF):
    """A Grid whose placement comes from template instance slots (GLSL only).

    Slots: s0 = (origin, spacing), s1 = (1/spacing, n - 1), s2 = (atlas offset + 0.5, margin).
    """

    def __init__(self, s0, s1, s2):
        self.s = (s0, s1, s2)

    def emit(self, ctx, p):
        s0, s1, s2 = self.s
        return ctx.var("float", f"gridSDF({p}, {s0}.xyz, {s1}.x, {s1}.yzw, {s2}.xyz, {s0}.w, {s2}.w)")

    def eval(self, p):
        raise RuntimeError("GridSlot is GLSL-only; evaluate the concrete instance instead")

    @staticmethod
    def slot_values(g: "Grid"):
        n = np.array(g.shape) - 1
        return [(*g.origin, g.spacing), (1.0 / g.spacing, *n), (*(np.array(g.atlas_offset) + 0.5), g.margin)]


class ProjectToPlane(SDF):
    """Extrude a planar slice of the child along the plane normal.

    f(x) = child(x - ((x - point) . n) n). With an exact child this is a lower
    bound of the 2-D distance within the slice. Used for disc footprints.
    """

    def __init__(self, child, point, normal):
        self.child, self.point, self.normal = child, vec3(point), vec3(normal)

    def children(self): return [self.child]
    def exprs(self): return [self.point, self.normal]

    def emit(self, ctx, p):
        n = _unit_glsl(self.normal)
        q = ctx.var("vec3", f"{p} - dot({p} - {self.point.glsl()}, {n}) * {n}")
        return ctx.sub(self.child, q)

    def eval(self, p):
        n = _vv(self.normal)
        n = n / np.linalg.norm(n)
        return self.child.eval(p - np.outer((p - _vv(self.point)) @ n, n))


class Morph(SDF):
    """Blend a -> b along a direction: s = clamp(((x - point) . n) / length, 0, 1)."""

    def __init__(self, a, b, point, normal, length):
        self.a, self.b = a, b
        self.point, self.normal, self.length = vec3(point), vec3(normal), wrap(length)

    def children(self): return [self.a, self.b]
    def exprs(self): return [self.point, self.normal, self.length]

    def emit(self, ctx, p):
        da, db = ctx.sub(self.a, p), ctx.sub(self.b, p)
        s = ctx.var(
            "float",
            f"clamp(dot({p} - {self.point.glsl()}, {_unit_glsl(self.normal)})"
            f" / {self.length.glsl()}, 0.0, 1.0)",
        )
        return ctx.var("float", f"mix({da}, {db}, {s})")

    def eval(self, p):
        n = _vv(self.normal)
        s = np.clip((p - _vv(self.point)) @ (n / np.linalg.norm(n)) / _v(self.length), 0, 1)
        return self.a.eval(p) * (1 - s) + self.b.eval(p) * s


def _rot_np(q, axis, a):
    """Rotate rows of q by angle a (radians, right-hand rule) about axis 0/1/2."""
    i, j = {0: (1, 2), 1: (2, 0), 2: (0, 1)}[axis]
    c, s = np.cos(a), np.sin(a)
    r = q.copy()
    r[:, i] = c * q[:, i] - s * q[:, j]
    r[:, j] = s * q[:, i] + c * q[:, j]
    return r


class FramePose(SDF):
    """Rigid motion of a body in a local frame (e.g. a vertebra's own axes).

    The frame has origin o and orthonormal axes ex (right), ey (anterior), ez
    (cranial). The body is rotated about o by R = Rz(axial) Ry(lateral) Rx(-flexion)
    in frame coordinates (degrees: flexion + bends forward, lateral bending + to the
    right, axial rotation + turns the front to the left) and then moved by
    ``shift`` (frame coordinates, mm). Exact (a rigid motion keeps distances).
    """

    def __init__(self, child, origin, ex, ey, ez, flexion, lateral, axial, shift):
        self.child = child
        self.o, self.ex, self.ey, self.ez = vec3(origin), vec3(ex), vec3(ey), vec3(ez)
        self.flexion, self.lateral, self.axial = wrap(flexion), wrap(lateral), wrap(axial)
        self.shift = vec3(shift)

    def children(self): return [self.child]
    def exprs(self): return [self.o, self.ex, self.ey, self.ez, self.flexion, self.lateral, self.axial, self.shift]

    def emit(self, ctx, p):
        o, ex, ey, ez = self.o.glsl(), self.ex.glsl(), self.ey.glsl(), self.ez.glsl()
        t = self.shift.c
        q = ctx.var("vec3", f"{p} - {o} - ({ex} * {t[0].glsl()} + {ey} * {t[1].glsl()} + {ez} * {t[2].glsl()})")
        l0 = ctx.var("vec3", f"vec3(dot({q}, {ex}), dot({q}, {ey}), dot({q}, {ez}))")
        l1 = ctx.var("vec3", f"rotZ({l0}, -radians({self.axial.glsl()}))")
        l2 = ctx.var("vec3", f"rotY({l1}, -radians({self.lateral.glsl()}))")
        l3 = ctx.var("vec3", f"rotX({l2}, radians({self.flexion.glsl()}))")
        return ctx.sub(self.child, ctx.var("vec3", f"{o} + {ex} * {l3}.x + {ey} * {l3}.y + {ez} * {l3}.z"))

    def eval(self, p):
        o, E = _vv(self.o), np.stack([_vv(self.ex), _vv(self.ey), _vv(self.ez)])  # rows: axes
        q = p - o - _vv(self.shift) @ E
        loc = q @ E.T
        loc = _rot_np(loc, 2, -np.radians(_v(self.axial)))
        loc = _rot_np(loc, 1, -np.radians(_v(self.lateral)))
        loc = _rot_np(loc, 0, np.radians(_v(self.flexion)))
        return self.child.eval(o + loc @ E)


class Tube(SDF):
    """Exact. Polyline through points (Vec3 expressions) swept by radius r."""

    def __init__(self, points, r):
        self.points, self.r = [vec3(q) for q in points], wrap(r)

    def exprs(self): return self.points + [self.r]

    CHUNK = 8  # segments per culling box

    def emit(self, ctx, p):
        # Chunks of segments behind a box around their points: a chunk whose box is
        # farther than the nearest segment so far cannot change the minimum (exact).
        r, pts = self.r.glsl(), [q.glsl() for q in self.points]
        d = "1e10"
        for i in range(0, len(pts) - 1, self.CHUNK):
            c = pts[i:i + self.CHUNK + 1]
            lo = ctx.var("vec3", "min(" * (len(c) - 1) + c[0] + "".join(f", {q})" for q in c[1:]))
            hi = ctx.var("vec3", "max(" * (len(c) - 1) + c[0] + "".join(f", {q})" for q in c[1:]))
            caps = d
            for a, b in zip(c, c[1:]):
                caps = f"min({caps}, sdCap({p}, {a}, {b}, {r}))"
            d = ctx.var("float", f"sdAABB({p}, {lo} - {r}, {hi} + {r}) < {d} ? {caps} : {d}")
        return d

    def eval(self, p):
        P = [_vv(q) for q in self.points]
        out = np.full(len(p), np.inf)
        for a, b in zip(P, P[1:]):
            pa, ba = p - a, b - a
            h = np.clip(pa @ ba / max(ba @ ba, 1e-12), 0, 1)
            out = np.minimum(out, _len(pa - h[:, None] * ba))
        return out - _v(self.r)


class RigidMove(SDF):
    """Rigid motion p -> R p + t; ``m`` = 12 scalars: R's columns, then t. Exact.

    ``local_box`` (lo, hi), when set, contains the child in its own (unmoved) frame:
    the scene then culls the body with that box moved along (Scene.glsl), which stays
    tight however far the motion goes.
    """

    def __init__(self, child, m, local_box=None):
        self.child = child
        self.m = [wrap(e) for e in m]
        self.local_box = local_box

    def children(self): return [self.child]
    def exprs(self): return self.m

    def m_glsl(self):
        """The motion as rigInv/rigPoint arguments: four vec3 (columns of R, then t)."""
        g = [e.glsl() for e in self.m]
        return ", ".join(f"vec3({', '.join(g[i:i + 3])})" for i in range(0, 12, 3))

    def emit(self, ctx, p):
        return ctx.sub(self.child, ctx.var("vec3", f"rigInv({p}, {self.m_glsl()})"))

    def eval(self, p):
        m = np.array([_v(e) for e in self.m])
        R, t = m[:9].reshape(3, 3).T, m[9:]
        return self.child.eval((p - t) @ R)


class Warp(SDF):
    """A body carried by a soft-tissue field: child(f(p)) * LIP, f the inverse map of a
    computed table (implicit.anatomy.alignment): kind "tps" (tps_table, ``n`` control
    points) or "polyrigid" (chain_table, ``n`` segments).

    Not an exact distance: the field stretches space, so the value is scaled by LIP
    (a lower bound while the field stretches distances by at most 1/LIP).
    """
    LIP = 0.8

    def __init__(self, child, kind, table, n):
        self.child, self.kind, self.table, self.n = child, kind, table, int(n)

    def children(self): return [self.child]
    def exprs(self): return [self.table]

    def emit(self, ctx, p):
        q = ctx.var("vec3", inverse_glsl(self.kind, p, self.table.glsl(), self.n))
        return ctx.var("float", f"{glsl_float(self.LIP)} * {ctx.sub(self.child, q)}")

    def eval(self, p):
        return self.LIP * self.child.eval(inverse_np(self.kind, self.table.value(), p))


def inverse_glsl(kind, p, table, n):
    return f"{'tpsInv' if kind == 'tps' else 'polyInv'}({p}, {table}, {n})"


def inverse_np(kind, table, p):
    from .anatomy.alignment import poly_inverse, tps_eval
    return tps_eval(table, p) if kind == "tps" else poly_inverse(table, p)


class Lattice(SDF):
    """Exact. Lines along x, y and z through the points lo + k * step, radius r, inside
    the box lo-hi (constants)."""

    def __init__(self, lo, hi, step, r):
        self.lo, self.hi, self.step, self.r = np.asarray(lo, float), np.asarray(hi, float), wrap(step), wrap(r)

    def exprs(self): return [self.step, self.r]

    def emit(self, ctx, p):
        v = lambda a: "vec3(" + ", ".join(glsl_float(x) for x in a) + ")"
        return ctx.var("float", f"sdLattice({p}, {v(self.lo)}, {v(self.hi)}, {self.step.glsl()}, {self.r.glsl()})")

    def eval(self, p):
        a, s = p - self.lo, _v(self.step)
        m = np.abs(a - s * np.floor(a / s + 0.5))
        d = np.minimum(np.minimum(np.hypot(m[:, 0], m[:, 1]), np.hypot(m[:, 1], m[:, 2])),
                       np.hypot(m[:, 0], m[:, 2])) - _v(self.r)
        return np.maximum(d, _sd_aabb(p, self.lo, self.hi))


def _sd_aabb(p, lo, hi):
    q = np.abs(p - 0.5 * (lo + hi)) - 0.5 * (hi - lo)
    return _len(np.maximum(q, 0.0)) + np.minimum(q.max(axis=1), 0.0)


class Arrows(SDF):
    """Arrows on a grid (lo, step, n points per axis): from each grid point along its
    displacement in the computed table ``table`` (dx, dy, dz, |d| per point; x fastest),
    drawn at most 0.8 step long (colour shows the length), shaft radius r; none below
    0.5 mm. Exact near the arrows: a point only meets arrows of the 27 grid points
    around it; farther away a lower bound."""

    def __init__(self, table, lo, step, n, r):
        self.table, self.lo, self.step, self.n, self.r = table, np.asarray(lo, float), float(step), list(n), wrap(r)

    def exprs(self): return [self.table, self.r]

    def call(self, p):
        lo = "vec3(" + ", ".join(glsl_float(x) for x in self.lo) + ")"
        n = "ivec3(" + ", ".join(str(int(k)) for k in self.n) + ")"
        return f"arrows({p}, {self.table.glsl()}, {lo}, {glsl_float(self.step)}, {n}, {self.r.glsl()})"

    def emit(self, ctx, p):
        return ctx.var("float", f"{self.call(p)}.x")

    def _np(self, p):
        D = np.asarray(self.table.value(), float).reshape(-1, 4)
        nx, ny, nz = self.n
        r, s = _v(self.r), self.step
        c = np.floor((p - self.lo) / s + 0.5).astype(int)
        best, mag = np.full(len(p), 1e10), np.zeros(len(p))
        for off in np.array(np.meshgrid([-1, 0, 1], [-1, 0, 1], [-1, 0, 1], indexing="ij")).reshape(3, -1).T:
            g = c + off
            ok = np.all((g >= 0) & (g < [nx, ny, nz]), axis=1)
            gi = np.where(ok, g[:, 0] + nx * (g[:, 1] + ny * g[:, 2]), 0)
            d4 = D[gi]
            ok &= d4[:, 3] >= 0.5
            a = self.lo + s * g
            L = np.minimum(d4[:, 3], 0.8 * s)
            u = d4[:, :3] / np.maximum(d4[:, 3:4], 1e-9)
            b = a + u * L[:, None]
            hl = np.minimum(4.0 * r, 0.4 * L)
            m = b - u * hl[:, None]
            dd = np.minimum(_cap_np(p, a, m, r), _round_cone_np(p, m, b, 2.2 * r, 0.2 * r))
            dd = np.where(ok, dd, 1e10)
            take = dd < best
            best, mag = np.where(take, dd, best), np.where(take, d4[:, 3], mag)
        hi = self.lo + s * (np.asarray(self.n) - 1)  # bounds of the arrows not tested (see arrows())
        far = np.maximum(0.7 * s - 2.2 * r, _sd_aabb(p, self.lo, hi) - 0.8 * s - 2.2 * r)
        return np.minimum(best, far), mag

    def eval(self, p):
        return self._np(p)[0]


def _cap_np(p, a, b, r):
    pa, ba = p - a, b - a
    h = np.clip(np.sum(pa * ba, 1) / np.maximum(np.sum(ba * ba, 1), 1e-12), 0, 1)
    return _len(pa - ba * h[:, None]) - r


def _round_cone_np(p, a, b, r1, r2):
    ba = b - a
    l2 = np.maximum(np.sum(ba * ba, 1), 1e-12)
    rr = r1 - r2
    a2 = l2 - rr * rr
    pa = p - a
    y = np.sum(pa * ba, 1)
    z = y - l2
    xv = pa * l2[:, None] - ba * y[:, None]
    x2 = np.sum(xv * xv, 1)
    y2, z2 = y * y * l2, z * z * l2
    k = np.sign(rr) * rr * rr * x2
    return np.where(np.sign(z) * a2 * z2 > k, np.sqrt(x2 + z2) / l2 - r2,
                    np.where(np.sign(y) * a2 * y2 < k, np.sqrt(x2 + y2) / l2 - r1,
                             (np.sqrt(np.maximum(x2 * a2 / l2, 0)) + y * rr) / l2 - r1))


class FlowLines(SDF):
    """Flow lines (implicit.anatomy.alignment.flow_tables): tubes of radius r along the
    segments of two computed tables, looked up in a grid of cells (lo, cell size, n).
    Exact near the lines; elsewhere a lower bound (the cell boundary + a margin)."""

    def __init__(self, cells, segs, flow, r):
        self.cells, self.segs, self.flow, self.r = cells, segs, flow, wrap(r)

    def exprs(self): return [self.cells, self.segs, self.r]

    def call(self, p):
        f = self.flow
        lo = "vec3(" + ", ".join(glsl_float(x) for x in f["lo"]) + ")"
        n = "ivec3(" + ", ".join(str(int(k)) for k in f["n"]) + ")"
        return (f"flowLines({p}, {self.cells.glsl()}, {self.segs.glsl()}, {lo}, {glsl_float(f['cell'])}, {n}, "
                f"{self.r.glsl()})")

    def emit(self, ctx, p):
        return ctx.var("float", f"{self.call(p)}.x")

    def _np(self, p):
        from .anatomy.alignment import flow_eval
        return flow_eval(self.cells.value(), self.segs.value(), self.flow, p, _v(self.r))

    def eval(self, p):
        return self._np(p)[:, 0]


class Slab(SDF):
    """Exact. A slab of half-thickness h across the box lo-hi, perpendicular to an axis
    (0, 1, 2) at ``offset``."""

    def __init__(self, axis, offset, h, lo, hi):
        self.axis, self.offset, self.h = int(axis), wrap(offset), float(h)
        self.lo, self.hi = np.asarray(lo, float), np.asarray(hi, float)

    def exprs(self): return [self.offset]

    def emit(self, ctx, p):
        v = lambda a: "vec3(" + ", ".join(glsl_float(x) for x in a) + ")"
        c = "xyz"[self.axis]
        return ctx.var("float", f"max(abs({p}.{c} - {self.offset.glsl()}) - {glsl_float(self.h)}, "
                                f"sdAABB({p}, {v(self.lo)}, {v(self.hi)}))")

    def eval(self, p):
        return np.maximum(np.abs(p[:, self.axis] - _v(self.offset)) - self.h, _sd_aabb(p, self.lo, self.hi))


class ColorValue:
    """A scalar field (GLSL expression at a point) that colours a body (Scene.add colormap)."""

    def __init__(self, glsl_fn, np_fn, exprs):
        self.glsl_fn, self.np_fn, self._exprs = glsl_fn, np_fn, exprs

    def params(self):
        return [q for e in self._exprs for q in e.params()]


class ScaleAbout(SDF):
    """Uniform scale by k about a centre point."""

    def __init__(self, child, k, center):
        self.child, self.k, self.center = child, wrap(k), vec3(center)

    def children(self): return [self.child]
    def exprs(self): return [self.k, self.center]

    def emit(self, ctx, p):
        c, k = self.center.glsl(), self.k.glsl()
        d = ctx.sub(self.child, ctx.var("vec3", f"{c} + ({p} - {c}) / {k}"))
        return ctx.var("float", f"{d} * {k}")

    def eval(self, p):
        c, k = _vv(self.center), _v(self.k)
        return self.child.eval(c + (p - c) / k) * k


POLY_ITERS, POLY_RELAX = 12, 4  # polyrigid inverse: steps, full steps before half steps (alignment.py)
# Flow line lookup margin (mm; above the renderer's hit tolerance, 0.002 mm per mm of ray, at
# viewing distances up to 3 m) and segment table width (alignment.py).
FLOW_MARGIN, FLOW_ROW = 6.0, 512

GLSL_LIB = f"""
vec3 rotX(vec3 p, float a) {{ float c = cos(a), s = sin(a); return vec3(p.x, c*p.y - s*p.z, s*p.y + c*p.z); }}
vec3 rotY(vec3 p, float a) {{ float c = cos(a), s = sin(a); return vec3(c*p.x + s*p.z, p.y, -s*p.x + c*p.z); }}
vec3 rotZ(vec3 p, float a) {{ float c = cos(a), s = sin(a); return vec3(c*p.x - s*p.y, s*p.x + c*p.y, p.z); }}
uniform highp sampler3D u_atlas;
uniform highp sampler2D u_data;
uniform vec3 u_atlasSize;
float gridSDF(vec3 p, vec3 o, float invh, vec3 nmax, vec3 off, float h, float margin) {{
    // Clamp in world space and measure the distance in world space: inside the
    // grid p - cw is exactly zero. (Computing it as length(q - clamp(q)) * h in
    // scaled grid coordinates left a rounding residue on Intel/Direct3D, so the
    // "outside" branch was taken everywhere and most bones vanished.)
    // Explicit LOD and no fetch inside a branch: implicit-derivative sampling in
    // divergent control flow is undefined in GLSL ES and returns garbage on many GPUs.
    vec3 cw = clamp(p, o, o + nmax * h);
    float ext = length(p - cw);
    vec3 qc = (cw - o) * invh;
    float d = (textureLod(u_atlas, (qc.zyx + off) / u_atlasSize, 0.0).r * 255.0 - {QZERO}.0) * {QSTEP};
    // Outside the grid both terms are lower bounds of the distance (the object lies
    // ``margin`` inside the box; d is the value at the nearest grid point, at least
    // ``margin`` on the border), and their max tends to d at the box faces: no jump
    // there for GPU and NumPy rounding to fall on different sides of.
    return ext > 0.0 ? max(ext + margin, d - ext) : d;
}}
// Pose (graph Pose block): R = Rz(axial) Ry(lateral) Rx(-flexion) in the frame (ex, ey, ez);
// ang = (flexion, lateral, axial) in degrees, sh = shift along the frame axes.
vec3 poseRot(vec3 l, vec3 ang) {{
    return rotZ(rotY(rotX(l, -radians(ang.x)), radians(ang.y)), radians(ang.z));
}}
vec3 poseDir(vec3 v, vec3 ex, vec3 ey, vec3 ez, vec3 ang) {{
    vec3 l = poseRot(vec3(dot(v, ex), dot(v, ey), dot(v, ez)), ang);
    return ex * l.x + ey * l.y + ez * l.z;
}}
vec3 posePoint(vec3 p, vec3 o, vec3 ex, vec3 ey, vec3 ez, vec3 ang, vec3 sh) {{
    return o + poseDir(p - o, ex, ey, ez, ang) + ex * sh.x + ey * sh.y + ez * sh.z;
}}
// Distance to the segment a-b minus r (see Tube).
float sdCap(vec3 p, vec3 a, vec3 b, float r) {{
    vec3 pa = p - a, ba = b - a;
    return length(pa - ba * clamp(dot(pa, ba) / max(dot(ba, ba), 1e-12), 0.0, 1.0)) - r;
}}
// Rigid motion p -> R p + t (R's columns c0, c1, c2) of a point / a direction, and its inverse.
vec3 rigDir(vec3 v, vec3 c0, vec3 c1, vec3 c2, vec3 t) {{
    return c0 * v.x + c1 * v.y + c2 * v.z;
}}
vec3 rigPoint(vec3 v, vec3 c0, vec3 c1, vec3 c2, vec3 t) {{
    return c0 * v.x + c1 * v.y + c2 * v.z + t;
}}
vec3 rigInv(vec3 p, vec3 c0, vec3 c1, vec3 c2, vec3 t) {{
    vec3 q = p - t;
    return vec3(dot(q, c0), dot(q, c1), dot(q, c2));
}}
float sdAABB(vec3 p, vec3 lo, vec3 hi) {{
    vec3 q = abs(p - 0.5 * (lo + hi)) - 0.5 * (hi - lo);
    return length(max(q, 0.0)) + min(max(q.x, max(q.y, q.z)), 0.0);
}}
// Soft-tissue fields (implicit.anatomy.alignment): inverse maps from computed tables.
vec3 tpsInv(vec3 p, highp sampler2D T, int n) {{
    vec4 a0 = texelFetch(T, ivec2(0, 0), 0), a1 = texelFetch(T, ivec2(1, 0), 0), a2 = texelFetch(T, ivec2(2, 0), 0);
    vec3 q = vec3(dot(a0.xyz, p) + a0.w, dot(a1.xyz, p) + a1.w, dot(a2.xyz, p) + a2.w);
    for (int i = 1; i <= n; i++) q += texelFetch(T, ivec2(1, i), 0).xyz * length(p - texelFetch(T, ivec2(0, i), 0).xyz);
    return q;
}}
vec3 screwApply(vec3 x, vec3 w, vec3 u) {{
    float phi = length(w);
    if (phi < 1e-9) return x + u;
    vec3 k = w / phi, wu = cross(w, u);
    float c = cos(phi), s = sin(phi);
    return x * c + cross(k, x) * s + k * dot(k, x) * (1.0 - c) + u + (1.0 - c) / (phi * phi) * wu
        + (phi - s) / (phi * phi * phi) * cross(w, wu);
}}
vec3 polyInv(vec3 p, highp sampler2D T, int n) {{
    vec3 x = p;
    for (int it = 0; it < {POLY_ITERS}; it++) {{
        int cnt = 0;
        for (int r = 0; r < n; r++) {{
            vec4 pi = texelFetch(T, ivec2(5, r), 0), ps = texelFetch(T, ivec2(6, r), 0);
            if (r > 0 && dot(pi.xyz, x) + pi.w >= 0.0) cnt++;
            if (dot(ps.xyz, x) + ps.w >= 0.0) cnt++;
        }}
        int j = cnt - 1;
        int r = j < 0 ? 0 : ((j & 1) == 1 ? (j + 1) / 2 : j / 2);
        float lam = 0.0;
        if (j >= 0 && (j & 1) == 0 && r < n - 1) {{
            vec4 ps = texelFetch(T, ivec2(6, r), 0), pi = texelFetch(T, ivec2(5, r + 1), 0);
            float a = dot(ps.xyz, x) + ps.w, b = dot(pi.xyz, x) + pi.w;
            lam = clamp(a / (abs(a - b) > 1e-12 ? a - b : 1.0), 0.0, 1.0);
        }}
        vec4 c0 = texelFetch(T, ivec2(0, r), 0), c1 = texelFetch(T, ivec2(1, r), 0), c2 = texelFetch(T, ivec2(2, r), 0);
        vec3 q = p - vec3(c0.w, c1.w, c2.w);
        q = vec3(dot(q, c0.xyz), dot(q, c1.xyz), dot(q, c2.xyz));
        vec3 g = screwApply(q, -lam * texelFetch(T, ivec2(3, r), 0).xyz, -lam * texelFetch(T, ivec2(4, r), 0).xyz);
        x = it < {POLY_RELAX} ? g : x + 0.5 * (g - x);
    }}
    return x;
}}
float sdLattice(vec3 q, vec3 lo, vec3 hi, float s, float r) {{
    vec3 a = q - lo, m = abs(a - s * floor(a / s + 0.5));
    return max(min(min(length(m.xy), length(m.yz)), length(m.xz)) - r, sdAABB(q, lo, hi));
}}
float sdRoundCone(vec3 p, vec3 a, vec3 b, float r1, float r2) {{
    vec3 ba = b - a;
    float l2 = max(dot(ba, ba), 1e-12), rr = r1 - r2, a2 = l2 - rr * rr;
    vec3 pa = p - a;
    float y = dot(pa, ba), z = y - l2;
    vec3 xv = pa * l2 - ba * y;
    float x2 = dot(xv, xv), y2 = y * y * l2, z2 = z * z * l2, k = sign(rr) * rr * rr * x2;
    if (sign(z) * a2 * z2 > k) return sqrt(x2 + z2) / l2 - r2;
    if (sign(y) * a2 * y2 < k) return sqrt(x2 + y2) / l2 - r1;
    return (sqrt(max(x2 * a2 / l2, 0.0)) + y * rr) / l2 - r1;
}}
// Arrows of a displacement grid (see Arrows): (distance, length of the nearest arrow's displacement).
vec2 arrows(vec3 p, highp sampler2D T, vec3 lo, float s, ivec3 n, float r) {{
    ivec3 c = ivec3(floor((p - lo) / s + 0.5));
    vec2 best = vec2(1e10, 0.0);
    for (int i = -1; i <= 1; i++) for (int j = -1; j <= 1; j++) for (int k = -1; k <= 1; k++) {{
        ivec3 g = c + ivec3(i, j, k);
        if (any(lessThan(g, ivec3(0))) || any(greaterThanEqual(g, n))) continue;
        vec4 D = texelFetch(T, ivec2(g.x, g.y + n.y * g.z), 0);
        if (D.w < 0.5) continue;
        vec3 a = lo + s * vec3(g), u = D.xyz / D.w;
        float L = min(D.w, 0.8 * s), hl = min(4.0 * r, 0.4 * L);
        vec3 b = a + u * L, m = b - u * hl;
        float d = min(sdCap(p, a, m, r), sdRoundCone(p, m, b, 2.2 * r, 0.2 * r));
        if (d < best.x) best = vec2(d, D.w);
    }}
    // Lower bounds of the arrows not tested: beyond the 27 cells (>= 0.7 s - 2.2 r away)
    // and outside the grid box grown by an arrow's reach.
    vec3 hi = lo + s * vec3(n - 1);
    best.x = min(best.x, max(0.7 * s - 2.2 * r, sdAABB(p, lo, hi) - 0.8 * s - 2.2 * r));
    return best;
}}
// Flow lines (see FlowLines): (distance, displacement so far) from the segments of p's cell.
vec2 flowLines(vec3 p, highp sampler2D C, highp sampler2D S, vec3 lo, float cs, ivec3 n, float r) {{
    vec3 hi = lo + cs * vec3(n);
    ivec3 c = ivec3(floor((p - lo) / cs));
    if (any(lessThan(c, ivec3(0))) || any(greaterThanEqual(c, n))) return vec2(sdAABB(p, lo, hi) + {FLOW_MARGIN}, 0.0);
    vec3 clo = lo + cs * vec3(c), e = min(p - clo, clo + cs - p);
    vec4 cell = texelFetch(C, ivec2(c.x, c.y + n.y * c.z), 0);
    // Lines outside the cell (and its empty layers) are beyond its boundary + the margin.
    vec2 best = vec2(min(e.x, min(e.y, e.z)) + cell.z * cs + {FLOW_MARGIN}, 0.0);
    int off = int(cell.x + 0.5), cnt = int(cell.y + 0.5);
    for (int i = 0; i < cnt; i++) {{
        int k = 2 * (off + i);
        vec4 A = texelFetch(S, ivec2(k % {FLOW_ROW}, k / {FLOW_ROW}), 0), B = texelFetch(S, ivec2((k + 1) % {FLOW_ROW}, (k + 1) / {FLOW_ROW}), 0);
        vec3 pa = p - A.xyz, ba = B.xyz - A.xyz;
        float h = clamp(dot(pa, ba) / max(dot(ba, ba), 1e-12), 0.0, 1.0), d = length(pa - ba * h) - r;
        if (d < best.x) best = vec2(d, A.w + (B.w - A.w) * h);
    }}
    return best;
}}
// Colour map for field magnitudes (0..1; a viridis-like polynomial fit).
vec3 fieldColor(float t) {{
    t = clamp(t, 0.0, 1.0);
    vec3 c0 = vec3(0.2777, 0.0054, 0.3341), c1 = vec3(0.1050, 1.4046, 1.3846), c2 = vec3(-0.3309, 0.2148, 0.0951),
         c3 = vec3(-4.6342, -5.7991, -19.3324), c4 = vec3(6.2283, 14.1799, 56.6906), c5 = vec3(4.7764, -13.7451, -65.3530),
         c6 = vec3(-5.4355, 4.6459, 26.3124);
    return c0 + t * (c1 + t * (c2 + t * (c3 + t * (c4 + t * (c5 + t * c6)))));
}}
"""

__all__ = [
    "SDF", "Sphere", "Box", "Cylinder", "EllipticCylinder", "Capsule", "HalfSpace",
    "Union", "Intersect", "Difference", "union", "Translate", "Rotate", "Scale",
    "MirrorX", "Offset", "Shell", "Grid", "GridSmooth", "smooth", "GridSlot", "ProjectToPlane", "Morph",
    "ScaleAbout", "FramePose",
    "Ctx", "GLSL_LIB", "glsl_float",
]
