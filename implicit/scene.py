"""A scene is a set of named bodies (bone, disc, cage, ...) sharing parameters."""

from __future__ import annotations

import base64
import gzip
import hashlib
import json

import numpy as np

from .expr import Param
from .sdf import GLSL_LIB, SDF, Ctx, Grid, glsl_float
from .template import Binder, Template


class Body:
    def __init__(self, name, sdf: SDF, color, group, bounds, visible=True, share=None):
        self.name, self.sdf, self.group, self.visible = name, sdf, group, visible
        self.share = share
        self.template = None
        self.color = tuple(float(c) for c in color)
        self.bounds = None if bounds is None else (
            tuple(map(float, bounds[0])), tuple(map(float, bounds[1])))


def _walk(node, seen=None):
    seen = set() if seen is None else seen
    if id(node) in seen:
        return
    seen.add(id(node))
    yield node
    for c in node.children():
        yield from _walk(c, seen)


def _shelf_place(order, side):
    """Shelf placement in layers of side x side; returns ({id: (z, y, x)}, depth)."""
    place, layer_z, layer_d = {}, 0, 0
    row_y = row_h = col_x = 0
    for g in order:
        nx, ny, nz = g.shape
        if layer_d == 0:
            layer_d = nx
        if col_x + nz > side:  # next row
            row_y, col_x, row_h = row_y + row_h, 0, 0
        if row_y + ny > side:  # next layer
            layer_z, layer_d = layer_z + layer_d, nx
            row_y = col_x = row_h = 0
        place[id(g)] = (layer_z, row_y, col_x)
        col_x += nz
        row_h = max(row_h, ny)
    return place, layer_z + layer_d


def pack_grids(grids, min_side=64, max_dim=2048):
    """Shelf-pack grid blocks into one uint8 volume; sets each grid's atlas fields.

    The layer side is chosen to minimise the atlas volume (texture memory),
    keeping every dimension <= ``max_dim`` (the common WebGL2 3D-texture limit).
    NumPy axes (0, 1, 2) map to texture (depth, height, width), i.e. texture
    coordinates are the grid index reversed (zyx), see Grid.emit.
    """
    if not grids:
        return None
    order = sorted(grids, key=lambda g: (g.shape[0], g.shape[1]), reverse=True)
    first = max(min_side, max(max(g.shape[1], g.shape[2]) for g in grids))
    best = None
    for side in range(first, max_dim + 1, 8):
        place, depth = _shelf_place(order, side)
        vol = side * side * depth
        if depth <= max_dim and (best is None or vol < best[0]):
            best = (vol, side, place, depth)
    if best is None:
        raise ValueError("grids do not fit in a 3D texture atlas; use coarser spacing")
    _, side, place, depth = best
    atlas = np.full((depth, side, side), 255, dtype=np.uint8)
    for g in grids:
        o = place[id(g)]
        nx, ny, nz = g.shape
        atlas[o[0]:o[0] + nx, o[1]:o[1] + ny, o[2]:o[2] + nz] = g.q
        g.atlas_offset = (o[2], o[1], o[0])
        g.atlas_size = (side, side, depth)
    return atlas


class Scene:
    def __init__(self, name, bounds):
        """``bounds`` = ((xmin, ymin, zmin), (xmax, ymax, zmax)) enclosing all bodies."""
        self.name = name
        self.bounds = (tuple(map(float, bounds[0])), tuple(map(float, bounds[1])))
        self.bodies: list[Body] = []
        self.templates: list[Template] = []
        self.attribution = ""
        self.frame_offset = None  # optional: scene coordinates = source frame - offset (mm)

    def add(self, name, sdf, color=(0.8, 0.8, 0.8), group=None, bounds=None, visible=True,
            share=None):
        """Add a body. ``group`` sets the viewer visibility toggle (default: own name).

        ``bounds`` (lo, hi) must contain the body; it lets the shader skip the body
        when a ray is far from it, which matters for scenes with many bodies.
        ``visible`` is the group's initial state in the viewer.
        Bodies with the same ``share`` key are evaluated in one shader function so
        common subgraphs (e.g. a disc split into annulus and nucleus) run once.
        """
        if any(b.name == name for b in self.bodies):
            raise ValueError(f"duplicate body name {name!r}")
        self.bodies.append(Body(name, sdf, color, group or name, bounds, visible, share))
        return sdf

    def add_template(self, name, build, instances, parts, bounds):
        """Add many bodies that share one graph structure (see implicit.template).

        ``instances``: list of dicts with a "name" plus the keys build() asks for.
        ``parts``: [(part name, color, group, visible), ...] in build()'s output order.
        ``bounds``: one (lo, hi) per instance, enclosing all its parts.
        """
        t = Template(name, build, instances, parts, bounds)
        t.first_body = len(self.bodies)
        for i, inst in enumerate(instances):
            for (pname, color, group, visible), sdf in zip(parts, build(Binder(inst))):
                b = Body(f"{inst['name']} {pname}", sdf, color, group, bounds[i], visible)
                b.template = t
                self.bodies.append(b)
        self.templates.append(t)
        return t

    def groups(self):
        out = []
        for b in self.bodies:
            if b.group not in out:
                out.append(b.group)
        return out

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

    def grids(self) -> list[Grid]:
        seen, out = set(), []
        for b in self.bodies:
            for n in _walk(b.sdf, seen):
                if isinstance(n, Grid):
                    out.append(n)
        return out

    def glsl(self, atlas=True) -> str:
        """GLSL: uniforms, one ``vec2 unit_k(vec3)`` per body (or shared group) and ``sceneMap``.

        Parameters are uniforms, so slider changes never recompile the shader.
        """
        grids = self.grids()
        if atlas:
            pack_grids(grids)
        groups = self.groups()
        out = [f"uniform float {p.uniform};" for p in self.params()]
        out += [f"uniform float u_grp{i};" for i in range(len(groups))]
        out.append(GLSL_LIB)
        cols = ", ".join(f"vec3({', '.join(glsl_float(c) for c in b.color)})" for b in self.bodies)
        n = len(self.bodies)
        out.append(f"const vec3 BODY_COLORS[{n}] = vec3[{n}]({cols});")
        out.append("vec3 bodyColor(int i) { return BODY_COLORS[i]; }")
        units = self._units()
        for k, unit in enumerate(units):
            ctx = Ctx()
            ds = [ctx.sub(self.bodies[i].sdf, "p") for i in unit]
            names = ", ".join(self.bodies[i].name for i in unit)
            out.append(f"vec2 unit{k}(vec3 p) {{  // {names}")
            out += ctx.lines
            if len(unit) == 1:
                out.append(f"    return vec2({ds[0]}, {unit[0]}.0);\n}}")
                continue
            out.append("    vec2 r = vec2(1e10, -1.0);")
            for i, d in zip(unit, ds):
                g = groups.index(self.bodies[i].group)
                out.append(f"    if (u_grp{g} > 0.5 && {d} < r.x) r = vec2({d}, {i}.0);")
            out.append("    return r;\n}")
        row = 0
        for t, tmpl in enumerate(self.templates):
            tmpl.row0 = row
            row += len(tmpl.instances)
            ctx = Ctx()
            ds = [ctx.sub(part, "p") for part in tmpl.shape]
            out.append(f"vec2 tmpl{t}(vec3 p, int row) {{  // {tmpl.name}")
            out += [f"    vec4 s{k} = texelFetch(u_data, ivec2({k}, row), 0);"
                    for k in range(2, tmpl.width)]
            out += ctx.lines
            n_parts = len(tmpl.parts)
            out.append(f"    float id0 = float({tmpl.first_body} + (row - {tmpl.row0}) * {n_parts});")
            out.append("    vec2 r = vec2(1e10, -1.0);")
            for j, (d, (_, _, group, _)) in enumerate(zip(ds, tmpl.parts)):
                out.append(f"    if (u_grp{groups.index(group)} > 0.5 && {d} < r.x) r = vec2({d}, id0 + {j}.0);")
            out.append("    return r;\n}")
        out.append("vec2 sceneMap(vec3 p) {\n    vec2 r = vec2(1e10, -1.0);\n    vec2 d;")
        for t, tmpl in enumerate(self.templates):
            vis = " || ".join(sorted({f"u_grp{groups.index(g)} > 0.5" for _, _, g, _ in tmpl.parts}))
            out.append(f"    if ({vis}) for (int i = 0; i < {len(tmpl.instances)}; i++) {{")
            out.append(f"        int row = {tmpl.row0} + i;")
            out.append("        if (sdAABB(p, texelFetch(u_data, ivec2(0, row), 0).xyz,"
                       " texelFetch(u_data, ivec2(1, row), 0).xyz) < r.x) {")
            out.append(f"            d = tmpl{t}(p, row); if (d.x < r.x) r = d;\n        }}\n    }}")

        def box(bounds):
            lo = ", ".join(glsl_float(x) for x in bounds[0])
            hi = ", ".join(glsl_float(x) for x in bounds[1])
            return f"sdAABB(p, vec3({lo}), vec3({hi})) < r.x"

        def unit_bounds(unit):
            bs = [self.bodies[i].bounds for i in unit]
            if any(b is None for b in bs):
                return None
            return np.min([b[0] for b in bs], axis=0), np.max([b[1] for b in bs], axis=0)

        def unit_line(k, indent="    "):
            unit = units[k]
            vis = " || ".join(sorted({f"u_grp{groups.index(self.bodies[i].group)} > 0.5" for i in unit}))
            cond = f"({vis})"
            b = unit_bounds(unit)
            if b is not None:
                cond += " && " + box(b)
            return f"{indent}if ({cond}) {{ d = unit{k}(p); if (d.x < r.x) r = d; }}"

        for cluster in self._clusters([unit_bounds(u) for u in units]):
            if len(cluster) == 1 or any(unit_bounds(units[k]) is None for k in cluster):
                out += [unit_line(k) for k in cluster]
                continue
            bs = [unit_bounds(units[k]) for k in cluster]
            lo = np.min([b[0] for b in bs], axis=0)
            hi = np.max([b[1] for b in bs], axis=0)
            out.append(f"    if ({box((lo, hi))}) {{")
            out += [unit_line(k, "        ") for k in cluster]
            out.append("    }")
        out.append("    return r;\n}")
        return "\n".join(out)

    def _units(self):
        """Bodies grouped by ``share`` key (unshared bodies are their own unit)."""
        units, by_key = [], {}
        for i, b in enumerate(self.bodies):
            if b.template is not None:
                continue
            if b.share is None:
                units.append([i])
            elif b.share in by_key:
                by_key[b.share].append(i)
            else:
                by_key[b.share] = [i]
                units.append(by_key[b.share])
        return units

    def _clusters(self, bounds, size=6):
        """Spatial clusters of unit indices (bounded units sorted along the long axis)."""
        bounded = [k for k, b in enumerate(bounds) if b is not None]
        free = [[k] for k, b in enumerate(bounds) if b is None]
        if len(bounded) <= size:
            return ([bounded] if bounded else []) + free
        axis = int(np.argmax(np.subtract(self.bounds[1], self.bounds[0])))
        order = sorted(bounded, key=lambda k: bounds[k][0][axis] + bounds[k][1][axis])
        return [order[i:i + size] for i in range(0, len(order), size)] + free

    def to_json(self, atlas_url=None) -> dict:
        """Viewer JSON. The grid atlas is inlined (gzip+base64) unless ``atlas_url`` is given."""
        glsl = self.glsl()
        groups = self.groups()
        d = {
            "name": self.name,
            "glsl": glsl,
            "bounds": self.bounds,
            "params": [p.to_json() for p in self.params()],
            "groups": [
                {"name": g, "color": next(b.color for b in self.bodies if b.group == g),
                 "visible": next(b.visible for b in self.bodies if b.group == g)}
                for g in groups
            ],
            "bodies": [{"name": b.name, "group": groups.index(b.group)} for b in self.bodies],
            "attribution": self.attribution,
            "frame_offset": None if self.frame_offset is None else [float(x) for x in self.frame_offset],
        }
        h = hashlib.sha1(glsl.encode())
        atlas = pack_grids(self.grids())
        if atlas is not None:
            raw = gzip.compress(atlas.tobytes(), compresslevel=6, mtime=0)
            h.update(raw)
            size = [atlas.shape[2], atlas.shape[1], atlas.shape[0]]
            d["atlas"] = {"size": size, "url": atlas_url} if atlas_url else {
                "size": size, "gzip_b64": base64.b64encode(raw).decode()}
            self._atlas_gz = raw
        data = self.instance_data()
        if data is not None:
            h.update(data.tobytes())
            d["data"] = {"size": [data.shape[1], data.shape[0]],
                         "f32_b64": base64.b64encode(data.tobytes()).decode()}
        d["hash"] = h.hexdigest()[:12]
        return d

    def instance_data(self):
        """Float32 [rows, width, 4] texture of all template instances (after packing)."""
        if not self.templates:
            return None
        width = max(t.width for t in self.templates)
        rows = []
        for t in self.templates:
            r = t.rows()
            rows.append(np.pad(r, ((0, 0), (0, width - r.shape[1]), (0, 0))))
        return np.ascontiguousarray(np.concatenate(rows), dtype=np.float32)

    def atlas_gz(self) -> bytes | None:
        """Gzipped atlas bytes from the last ``to_json`` call."""
        return getattr(self, "_atlas_gz", None)

    def write_json(self, path, atlas_url=None):
        with open(path, "w") as f:
            json.dump(self.to_json(atlas_url), f)

    def sample(self, name, h, pad=2.0, chunk=1_000_000):
        """Evaluate body ``name`` on a regular grid of spacing ``h``.

        Uses the body's own bounds when it has them, else the scene bounds.
        Returns (values[nx, ny, nz], origin, h).
        """
        b = self.body(name)
        lo, hi = b.bounds or self.bounds
        lo = np.array(lo) - pad
        hi = np.array(hi) + pad
        n = np.ceil((hi - lo) / h).astype(int) + 1
        axes = [lo[i] + h * np.arange(n[i]) for i in range(3)]
        g = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
        vals = np.concatenate([b.sdf.eval(g[i:i + chunk]) for i in range(0, len(g), chunk)])
        return vals.reshape(n), lo, h
