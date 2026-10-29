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
    Sample i sits at origin + i * spacing. Outside the grid the field is the
    distance to the grid box plus the smallest boundary value, a lower bound
    because the object lies inside the box.

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

    def emit(self, ctx, p):
        if self.atlas_offset is None:
            raise RuntimeError("Grid not packed; add it to a Scene before emitting GLSL")
        f = lambda xs: "vec3(" + ", ".join(glsl_float(x) for x in xs) + ")"
        n = np.array(self.shape) - 1
        return ctx.var(
            "float",
            f"gridSDF({p}, {f(self.origin)}, {glsl_float(1.0 / self.spacing)}, {f(n)}, "
            f"{f(np.array(self.atlas_offset) + 0.5)}, {glsl_float(self.spacing)}, {glsl_float(self.margin)})",
        )

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
        return np.where(ext > 1e-4, ext + self.margin, out)


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
        t = ctx.var("float", self._t_glsl())
        pres = self.preserve.glsl()
        terms = []
        for k, g in enumerate([self.grid] + self.levels):
            d = ctx.sub(g, p)
            w = f"max(1.0 - abs({t} - {glsl_float(float(k))}), 0.0)"
            off = f" - {pres} * {glsl_float(self.offsets[k])}" if self.offsets[k] else ""
            terms.append(f"{w} * ({d}{off})")
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
    return ext > 1e-4 ? ext + margin : d;
}}
float sdAABB(vec3 p, vec3 lo, vec3 hi) {{
    vec3 q = abs(p - 0.5 * (lo + hi)) - 0.5 * (hi - lo);
    return length(max(q, 0.0)) + min(max(q.x, max(q.y, q.z)), 0.0);
}}
"""

__all__ = [
    "SDF", "Sphere", "Box", "Cylinder", "EllipticCylinder", "Capsule", "HalfSpace",
    "Union", "Intersect", "Difference", "union", "Translate", "Rotate", "Scale",
    "MirrorX", "Offset", "Shell", "Grid", "GridSmooth", "smooth", "GridSlot", "ProjectToPlane", "Morph",
    "ScaleAbout",
    "Ctx", "GLSL_LIB", "glsl_float",
]
