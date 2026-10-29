"""Instanced bodies: one graph structure, many instances with different constants.

A builder function ``build(b) -> [sdf_part, ...]`` asks a Binder for its
per-instance constants (``b.vec3(key)``, ``b.scalar(key)``, ``b.grid(key)``).

* Concrete mode (``Binder(values)``): returns real values, so every instance
  is an ordinary SDF graph (used for NumPy evaluation / meshing).
* Template mode (``Binder()``): returns slot references; the graph is compiled
  to GLSL once and each instance's constants are packed into one row of a
  float data texture. The shader loops over instances, so shader size does not
  grow with the number of instances (e.g. 23 discs).

Params (sliders) used by the builder are shared uniforms in both modes.
"""

from __future__ import annotations

import numpy as np

from .expr import Const, SlotRef, Vec3, Vec3Slot
from .sdf import Grid, GridSlot

BOUNDS_SLOTS = 2  # slots 0, 1 of every row hold the instance AABB (lo, hi)


class Binder:
    def __init__(self, values=None):
        self.values = values
        self.slots: dict[str, tuple[int, str]] = {}  # key -> (first slot, kind)
        self.cache: dict[str, object] = {}
        self.n = BOUNDS_SLOTS

    @property
    def template(self):
        return self.values is None

    def _alloc(self, key, kind, count=1):
        if key not in self.slots:
            self.slots[key] = (self.n, kind)
            self.n += count
        return self.slots[key][0]

    def vec3(self, key) -> Vec3:
        if key not in self.cache:
            if self.template:
                self.cache[key] = Vec3Slot(f"s{self._alloc(key, 'vec3')}")
            else:
                self.cache[key] = Vec3(*map(float, self.values[key]))
        return self.cache[key]

    def scalar(self, key):
        if key not in self.cache:
            if self.template:
                self.cache[key] = SlotRef(f"s{self._alloc(key, 'scalar')}", "x")
            else:
                self.cache[key] = Const(float(self.values[key]))
        return self.cache[key]

    def grid(self, key):
        if key not in self.cache:
            if self.template:
                k = self._alloc(key, "grid", 3)
                self.cache[key] = GridSlot(f"s{k}", f"s{k + 1}", f"s{k + 2}")
            else:
                g = self.values[key]
                if not isinstance(g, Grid):
                    raise TypeError(f"{key!r} must be a Grid")
                self.cache[key] = g
        return self.cache[key]


class Template:
    def __init__(self, name, build, instances, parts, bounds):
        """``parts`` = [(part name, color, group, visible), ...] matching build()'s output."""
        self.name, self.build, self.instances, self.parts = name, build, instances, parts
        self.bounds = bounds
        self.binder = Binder()
        self.shape = build(self.binder)
        if len(self.shape) != len(parts):
            raise ValueError("build() must return one SDF per part")
        self.first_body = None  # set by Scene
        self.row0 = None        # first data-texture row, set at compile time

    @property
    def width(self):
        return self.binder.n

    def rows(self) -> np.ndarray:
        """Per-instance slot data, float32 [n_instances, width, 4] (after atlas packing)."""
        out = np.zeros((len(self.instances), self.width, 4), np.float32)
        for i, inst in enumerate(self.instances):
            lo, hi = self.bounds[i]
            out[i, 0, :3], out[i, 1, :3] = lo, hi
            for key, (k, kind) in self.binder.slots.items():
                v = inst[key]
                if kind == "vec3":
                    out[i, k, :3] = v
                elif kind == "scalar":
                    out[i, k, 0] = v
                else:
                    out[i, k:k + 3] = GridSlot.slot_values(v)
        return out
