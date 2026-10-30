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

    def is_const(self):
        """True when the value is known at code-generation time (no params or slots)."""
        return False

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

    def is_const(self):
        return True


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

    def is_const(self):
        return self.a.is_const() and self.b.is_const()


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

    def is_const(self):
        return all(a.is_const() for a in self.args)


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
        # The x, y, z of one vector-valued call (e.g. posePoint(...)): emit the call once.
        whole = getattr(self.c[0], "whole", None)
        if whole is not None and all(getattr(c, "whole", None) is whole and c.comp == k
                                     for k, c in enumerate(self.c)):
            return whole.glsl()
        return "vec3(" + ", ".join(c.glsl() for c in self.c) + ")"

    def value(self):
        return tuple(c.value() for c in self.c)

    def params(self):
        return [p for c in self.c for p in c.params()]

    def is_const(self):
        return all(c.is_const() for c in self.c)

    # Arithmetic stays vector-level in GLSL ("(a + b)", "(a * k)"), so a long vector
    # expression is not repeated once per component; per-component expressions are kept
    # for values, parameters and bounds.
    def __neg__(self):
        return _vec_op("neg", self, None, [-c for c in self.c])

    def __add__(self, o):
        o = vec3(o)
        return _vec_op("+", self, o, [a + b for a, b in zip(self.c, o.c)])

    def __sub__(self, o):
        o = vec3(o)
        return _vec_op("-", self, o, [a - b for a, b in zip(self.c, o.c)])

    def __mul__(self, k):
        k = wrap(k)
        return _vec_op("*", self, k, [c * k for c in self.c])

    __rmul__ = __mul__


class _VecOp:
    """A vector expression emitted whole: (a + b), (a - b), (a * k) or (-a)."""

    def __init__(self, op, a, b):
        self.op, self.a, self.b = op, a, b

    def glsl(self):
        if self.op == "neg":
            return f"(-{self.a.glsl()})"
        return f"({self.a.glsl()} {self.op} {self.b.glsl()})"


class VecComp(Expr):
    """Component of a whole vector expression; ``expanded()`` is the same value per component."""

    def __init__(self, whole, comp, expanded):
        self.whole, self.comp, self._expanded = whole, comp, expanded

    def glsl(self):
        return f"{self.whole.glsl()}.{'xyz'[self.comp]}"

    def value(self):
        return self._expanded.value()

    def params(self):
        return self._expanded.params()

    def is_const(self):
        return self._expanded.is_const()

    def expanded(self):
        return self._expanded


def _vec_op(op, a, b, comps):
    whole = _VecOp(op, a, b)
    return Vec3(*(VecComp(whole, k, c) for k, c in enumerate(comps)))


class SlotRef(Expr):
    """One component of an instance data slot (``vec4 s<k>`` in a template function)."""

    def __init__(self, var, comp):
        self.var, self.comp = var, comp

    def glsl(self):
        return f"{self.var}.{self.comp}"

    def value(self):
        raise RuntimeError("template slots have no value outside the shader")


class Vec3Slot(Vec3):
    def __init__(self, var):
        self.var = var
        self.c = tuple(SlotRef(var, k) for k in "xyz")

    def glsl(self):
        return f"{self.var}.xyz"


def vec3(v) -> Vec3:
    if isinstance(v, Vec3):
        return v
    if len(v) != 3:
        raise ValueError("expected 3 components")
    return Vec3(*(float(x) if not isinstance(x, Expr) else x for x in v))
