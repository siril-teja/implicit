"""A scene is a set of named bodies (bone, disc, cage, ...) sharing parameters."""

from __future__ import annotations

import hashlib
import json

import numpy as np

from .expr import Param
from .sdf import GLSL_LIB, SDF, Ctx


class Body:
    def __init__(self, name, sdf: SDF, color):
        self.name, self.sdf, self.color = name, sdf, tuple(float(c) for c in color)


class Scene:
    def __init__(self, name, bounds):
        """``bounds`` = ((xmin, ymin, zmin), (xmax, ymax, zmax)) enclosing all bodies."""
        self.name = name
        self.bounds = (tuple(map(float, bounds[0])), tuple(map(float, bounds[1])))
        self.bodies: list[Body] = []

    def add(self, name, sdf, color=(0.8, 0.8, 0.8)):
        if any(b.name == name for b in self.bodies):
            raise ValueError(f"duplicate body name {name!r}")
        self.bodies.append(Body(name, sdf, color))
        return sdf

    def params(self) -> list[Param]:
        seen: dict[str, Param] = {}
        for b in self.bodies:
            for p in b.sdf.params():
                q = seen.setdefault(p.name, p)
                if q is not p:
                    raise ValueError(f"two different params named {p.name!r}")
        return list(seen.values())

    def set_params(self, values: dict):
        by_name = {p.name: p for p in self.params()}
        for k, v in values.items():
            if k in by_name:
                by_name[k].set(v)

    def body(self, name) -> Body:
        for b in self.bodies:
            if b.name == name:
                return b
        raise KeyError(name)

    def glsl(self) -> str:
        """GLSL: uniforms, one ``float body_i(vec3)`` per body and ``vec2 sceneMap(vec3)``.

        Parameters are uniforms, so slider changes never recompile the shader.
        """
        out = [f"uniform float {p.uniform};" for p in self.params()]
        out += [f"uniform float u_vis{i};" for i in range(len(self.bodies))]
        out.append(GLSL_LIB)
        for i, b in enumerate(self.bodies):
            ctx = Ctx()
            d = b.sdf.emit(ctx, "p")
            out.append(f"float body{i}(vec3 p) {{  // {b.name}")
            out += ctx.lines
            out.append(f"    return {d};\n}}")
        out.append("vec2 sceneMap(vec3 p) {\n    vec2 r = vec2(1e10, -1.0);\n    float d;")
        for i in range(len(self.bodies)):
            out.append(
                f"    if (u_vis{i} > 0.5) {{ d = body{i}(p); if (d < r.x) r = vec2(d, {i}.0); }}"
            )
        out.append("    return r;\n}")
        return "\n".join(out)

    def to_json(self) -> dict:
        glsl = self.glsl()
        return {
            "name": self.name,
            "hash": hashlib.sha1(glsl.encode()).hexdigest()[:12],
            "glsl": glsl,
            "bounds": self.bounds,
            "params": [p.to_json() for p in self.params()],
            "bodies": [{"name": b.name, "color": b.color} for b in self.bodies],
        }

    def write_json(self, path):
        with open(path, "w") as f:
            json.dump(self.to_json(), f, indent=1)

    def sample(self, name, h, pad=2.0, chunk=1_000_000):
        """Evaluate body ``name`` on a regular grid of spacing ``h``.

        Returns (values[nx, ny, nz], origin, h).
        """
        lo = np.array(self.bounds[0]) - pad
        hi = np.array(self.bounds[1]) + pad
        n = np.ceil((hi - lo) / h).astype(int) + 1
        axes = [lo[i] + h * np.arange(n[i]) for i in range(3)]
        g = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
        sdf = self.body(name).sdf
        vals = np.concatenate([sdf.eval(g[i:i + chunk]) for i in range(0, len(g), chunk)])
        return vals.reshape(n), lo, h
