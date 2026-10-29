"""Scalar expressions and named parameters.

A ``Param`` becomes a GLSL uniform in the browser, so changing it there only
updates a uniform value; the shader is not recompiled and nothing is rebuilt.
On the Python side the same expression is evaluated to a float for meshing.
"""

from __future__ import annotations

import math


def glsl_float(v: float) -> str:
    s = repr(float(v))
    if "inf" in s or "nan" in s:
        raise ValueError(f"cannot emit {v} as a GLSL literal")
    return s


class Expr:
    def glsl(self) -> str:
        raise NotImplementedError

    def value(self) -> float:
        raise NotImplementedError

    def params(self):
        return []

    def __add__(self, o):
        return BinOp("+", self, wrap(o))

    def __radd__(self, o):
        return BinOp("+", wrap(o), self)

    def __sub__(self, o):
        return BinOp("-", self, wrap(o))

    def __rsub__(self, o):
        return BinOp("-", wrap(o), self)

    def __mul__(self, o):
        return BinOp("*", self, wrap(o))

    def __rmul__(self, o):
        return BinOp("*", wrap(o), self)

    def __truediv__(self, o):
        return BinOp("/", self, wrap(o))

    def __rtruediv__(self, o):
        return BinOp("/", wrap(o), self)

    def __neg__(self):
        return BinOp("-", Const(0.0), self)


class Const(Expr):
    def __init__(self, v: float):
        self.v = float(v)

    def glsl(self):
        return glsl_float(self.v)

    def value(self):
        return self.v


class Param(Expr):
    """A named, bounded design variable exposed as a slider in the viewer."""

    def __init__(self, name, value, lo, hi, step=None, label=None, unit=""):
        if not name.isidentifier():
            raise ValueError(f"param name must be an identifier: {name!r}")
        self.name = name
        self.default = float(value)
        self.v = float(value)
        self.lo, self.hi = float(lo), float(hi)
        self.step = float(step) if step else (self.hi - self.lo) / 200.0
        self.label = label or name
        self.unit = unit

    @property
    def uniform(self):
        return f"u_{self.name}"

    def glsl(self):
        return self.uniform

    def value(self):
        return self.v

    def set(self, v):
        self.v = float(v)

    def params(self):
        return [self]

    def to_json(self):
        return {
            "name": self.name, "uniform": self.uniform, "value": self.v,
            "default": self.default, "min": self.lo, "max": self.hi,
            "step": self.step, "label": self.label, "unit": self.unit,
        }


_PY_FUNCS = {
    "sin": math.sin, "cos": math.cos, "tan": math.tan,
    "radians": math.radians, "abs": abs, "sqrt": math.sqrt,
    "min": min, "max": max,
}
_PY_OPS = {
    "+": lambda a, b: a + b, "-": lambda a, b: a - b,
    "*": lambda a, b: a * b, "/": lambda a, b: a / b,
}


class BinOp(Expr):
    def __init__(self, op, a, b):
        self.op, self.a, self.b = op, a, b

    def glsl(self):
        return f"({self.a.glsl()} {self.op} {self.b.glsl()})"

    def value(self):
        return _PY_OPS[self.op](self.a.value(), self.b.value())

    def params(self):
        return self.a.params() + self.b.params()


class Func(Expr):
    def __init__(self, fn, *args):
        if fn not in _PY_FUNCS:
            raise ValueError(f"unsupported function {fn}")
        self.fn, self.args = fn, [wrap(a) for a in args]

    def glsl(self):
        return f"{self.fn}({', '.join(a.glsl() for a in self.args)})"

    def value(self):
        return _PY_FUNCS[self.fn](*(a.value() for a in self.args))

    def params(self):
        return [p for a in self.args for p in a.params()]


def wrap(x) -> Expr:
    if isinstance(x, Expr):
        return x
    if isinstance(x, (int, float)):
        return Const(x)
    raise TypeError(f"expected number or Expr, got {type(x).__name__}")


def sin(x): return Func("sin", x)
def cos(x): return Func("cos", x)
def tan(x): return Func("tan", x)
def radians(x): return Func("radians", x)


class Vec3:
    def __init__(self, x, y, z):
        self.c = (wrap(x), wrap(y), wrap(z))

    def glsl(self):
        return "vec3(" + ", ".join(c.glsl() for c in self.c) + ")"

    def value(self):
        return tuple(c.value() for c in self.c)

    def params(self):
        return [p for c in self.c for p in c.params()]

    def __neg__(self):
        return Vec3(*(-c for c in self.c))


def vec3(v) -> Vec3:
    if isinstance(v, Vec3):
        return v
    if len(v) != 3:
        raise ValueError("expected 3 components")
    return Vec3(*v)
