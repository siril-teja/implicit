// Block graph -> scene shader, in the browser (mirrors implicit/graph.py build_graph
// and Scene.glsl). Block GLSL comes from blocks.json (traced from the Python
// emitters), bone GLSL from <scene>.library.json, so nothing is implemented twice.
// The result has the same shape as the Python scene JSON, so the viewer and the
// GPU check use it unchanged.
"use strict";

const Graph = (() => {
  const ACCEPTS = { body: ["body", "bone"], bone: ["bone"], scalar: ["scalar"], vec3: ["vec3"],
    plane: ["plane"], frame: ["frame"] };
  const WIRED = new Set(["body", "bone", "scalar", "vec3", "plane", "frame"]);
  const WIRE_ONLY = new Set(["body", "bone", "plane", "frame"]);

  const isNum = (v) => typeof v === "number" && Number.isFinite(v);

  function glslFloat(v) {
    if (!Number.isFinite(v)) throw new Error(`cannot emit ${v} as a GLSL literal`);
    let s = String(v);
    if (!/[.e]/.test(s)) s += ".0";
    return s;
  }

  function parseWire(v) {
    if (typeof v !== "string" || !v.startsWith("@")) return null;
    const [node, port] = v.slice(1).split(".");
    return { node, port: port || null };
  }

  function colorOf(v) {
    if (Array.isArray(v)) return v.map(Number);
    const s = String(v).replace(/^#/, "");
    if (!/^[0-9a-fA-F]{6}$/.test(s)) throw new Error(`colour must be #rrggbb, got ${v}`);
    return [0, 2, 4].map((i) => parseInt(s.slice(i, i + 2), 16) / 255);
  }

  // ------------------------------------------------------------ bounds (as graph.py)
  const add = (a, b) => a.map((v, i) => v + b[i]);
  const sub = (a, b) => a.map((v, i) => v - b[i]);
  const grow = (b, r) => (b && r !== null ? [b[0].map((v) => v - Math.max(r, 0)), b[1].map((v) => v + Math.max(r, 0))] : null);
  const unionB = (a, b) => (a && b ? [a[0].map((v, i) => Math.min(v, b[0][i])), a[1].map((v, i) => Math.max(v, b[1][i]))] : null);
  function intersectB(a, b) {
    if (!a) return b;
    if (!b) return a;
    const lo = a[0].map((v, i) => Math.max(v, b[0][i]));
    return [lo, a[1].map((v, i) => Math.max(Math.min(v, b[1][i]), lo[i]))];
  }
  // Bounds functions get ``iv(scalar)``: the scalar's interval over its slider range
  // (shader bounds, valid at any value) or at its current value (view bounds).
  const upper = (s, iv) => { const r = iv(s); return r ? r[1] : null; };
  const vecIv = (v, iv) => {
    const r = v.c.map(iv);
    return r.every(Boolean) ? [r.map((x) => x[0]), r.map((x) => x[1])] : null;
  };
  const child = (b, iv) => (b.bf ? b.bf(iv) : null);

  const BOUNDS = {
    sphere(x, iv) {
      const c = vecIv(x.center, iv), r = upper(x.radius, iv);
      return c && r !== null ? [c[0].map((v) => v - r), c[1].map((v) => v + r)] : null;
    },
    box(x, iv) {
      const c = vecIv(x.center, iv), s = vecIv(x.size, iv);
      if (!c || !s) return null;
      const hh = [0, 1, 2].map((i) => Math.max(s[0][i] * 0.5, s[1][i] * 0.5));
      return [sub(c[0], hh), add(c[1], hh)];
    },
    cylinder(x, iv) {
      const c = vecIv(x.center, iv), r = upper(x.radius, iv), h = upper(x.height, iv);
      if (!c || r === null || h === null) return null;
      const e = [r, r, h / 2];
      return [sub(c[0], e), add(c[1], e)];
    },
    union: (x, iv) => grow(unionB(child(x.a, iv), child(x.b, iv)), upper(x.blend, iv)),
    intersect: (x, iv) => grow(intersectB(child(x.a, iv), child(x.b, iv)), upper(x.blend, iv)),
    subtract: (x, iv) => grow(child(x.a, iv), upper(x.blend, iv)),
    translate(x, iv) {
      const t = vecIv(x.offset, iv), b = child(x.body, iv);
      return b && t ? [add(b[0], t[0]), add(b[1], t[1])] : null;
    },
    rotate(x, iv) {
      const b = child(x.body, iv), c = vecIv(x.center, iv);
      if (!b || !c) return null;
      if (iv.now) {  // exact: the box corners rotated by the current angle
        const [i, j] = { x: [1, 2], y: [2, 0], z: [0, 1] }[x.axis];
        const a = iv(x.angle)[0] * Math.PI / 180, ca = Math.cos(a), sa = Math.sin(a);
        const lo = [Infinity, Infinity, Infinity], hi = [-Infinity, -Infinity, -Infinity];
        for (const u of [0, 1]) for (const v of [0, 1]) for (const w of [0, 1]) {
          const q = [b[u][0], b[v][1], b[w][2]].map((val, k) => val - c[0][k]);
          const r = q.slice();
          r[i] = ca * q[i] - sa * q[j];
          r[j] = sa * q[i] + ca * q[j];
          r.forEach((val, k) => { lo[k] = Math.min(lo[k], val + c[0][k]); hi[k] = Math.max(hi[k], val + c[0][k]); });
        }
        return [lo, hi];
      }
      let r = 0;
      for (const i of [0, 1]) for (const j of [0, 1]) for (const k of [0, 1]) {
        const corner = [b[i][0], b[j][1], b[k][2]];
        const far = corner.map((v, a) => Math.max(Math.abs(v - c[0][a]), Math.abs(v - c[1][a])));
        r = Math.max(r, Math.hypot(...far));
      }
      return [c[0].map((v) => v - r), c[1].map((v) => v + r)];
    },
    offset: (x, iv) => grow(child(x.body, iv), upper(x.distance, iv)),
    shell: (x, iv) => child(x.body, iv),
    plane_cut: (x, iv) => child(x.body, iv),
    smooth: (x, iv) => child(x.bone, iv),
    show_plane(x, iv) {
      const p = vecIv(x.plane.point, iv), r = upper(x.radius, iv), t = upper(x.thickness, iv);
      return p && r !== null && t !== null ? [p[0].map((v) => v - r - t), p[1].map((v) => v + r + t)] : null;
    },
    show_frame(x, iv) {
      const o = vecIv(x.frame.origin, iv), l = upper(x.length, iv), r = upper(x.radius, iv);
      return o && l !== null && r !== null ? [o[0].map((v) => v - l - r), o[1].map((v) => v + l + r)] : null;
    },
  };
  const rangeIv = (s) => s.iv || null;
  const nowIv = (values) => Object.assign((s) => {
    if (!s.param) return s.iv || null;
    const v = s.param.name in values ? values[s.param.name] : s.param.value;
    return [v, v];
  }, { now: true });
  // Constant values (from the library): scalars and vectors that are not parameters.
  const cs = (v) => ({ g: glslFloat(v), iv: [v, v] });
  const cvec = (a) => ({ c: a.map(cs) });

  // ------------------------------------------------------------ evaluation
  function topoOrder(nodes, errors) {
    const order = [], state = {};
    const visit = (id, path) => {
      if (state[id] === 2) return true;
      if (state[id] === 1) { for (const n of path.slice(path.indexOf(id))) errors[n] = "part of a cycle"; return false; }
      state[id] = 1;
      let ok = true;
      for (const v of Object.values(nodes[id].in || {})) {
        const w = parseWire(v);
        if (w && nodes[w.node]) ok = visit(w.node, path.concat(id)) && ok;
      }
      state[id] = 2;
      order.push(id);
      return ok;
    };
    for (const id of Object.keys(nodes)) visit(id, []);
    return order;
  }

  function literal(id, title, port, v, params) {
    const label = `${title}: ${port.label || port.name}`;
    const mk = (name, val, lab) => {
      if (!isNum(val)) throw new Error(`${port.name}: expected a number`);
      const lo = Math.min(port.lo ?? val, val), hi = Math.max(port.hi ?? val, val);
      const p = { name, uniform: `u_${name}`, value: val, default: val, min: lo, max: hi,
        step: port.step || (hi - lo) / 200, label: lab, unit: port.unit || "", node: id, port: port.name };
      params.push(p);
      return { g: p.uniform, iv: [lo, hi], param: p };
    };
    switch (port.type) {
      case "scalar": return mk(`${id}_${port.name}`, Number(v), label);
      case "vec3":
        if (!Array.isArray(v) || v.length !== 3) throw new Error(`${port.name}: expected [x, y, z]`);
        return { c: ["x", "y", "z"].map((a, i) => mk(`${id}_${port.name}_${a}`, Number(v[i]), `${label} ${a}`)) };
      case "number": if (!isNum(Number(v))) throw new Error(`${port.name}: expected a number`); return Number(v);
      case "text": case "path": return String(v);
      case "bool": return Boolean(v);
      case "color": return colorOf(v);
      case "choice":
        if (!port.choices.includes(v)) throw new Error(`${port.name}: ${v} is not one of ${port.choices.join(", ")}`);
        return v;
      default: throw new Error(`${port.name}: a ${port.type} input needs a wire`);
    }
  }

  // Evaluate a graph: per-node values (or errors), params, output bodies, scene bounds.
  function evaluate(graph, manifest, library) {
    const blocks = Object.fromEntries(manifest.blocks.map((b) => [b.name, b]));
    const nodes = graph.nodes || {};
    const errors = {}, values = {}, params = [], outputs = [];
    const order = topoOrder(nodes, errors);
    for (const id of order) {
      if (errors[id]) continue;
      const raw = nodes[id], blk = blocks[raw.block];
      if (!blk) { errors[id] = `unknown block "${raw.block}"`; continue; }
      if (!/^[A-Za-z_][A-Za-z0-9_]*$/.test(id)) { errors[id] = "node id must be an identifier"; continue; }
      const title = raw.title || id, given = raw.in || {}, x = {};
      try {
        for (const port of blk.inputs) {
          const v = port.name in given ? given[port.name] : port.default;
          const w = parseWire(v);
          if (w) {
            const src = nodes[w.node];
            if (!src) throw new Error(`${port.name}: wire from unknown node ${w.node}`);
            const outs = blocks[src.block] ? blocks[src.block].outputs : [];
            const sp = w.port ? outs.find((o) => o.name === w.port) : outs[0];
            if (!sp) throw new Error(`${port.name}: ${w.node} has no output ${w.port || ""}`);
            if (!(ACCEPTS[port.type] || []).includes(sp.type)) throw new Error(`${port.name}: cannot connect ${sp.type} to ${port.type}`);
            if (!values[w.node]) throw new Error(`${port.name}: input from ${w.node} is not available`);
            x[port.name] = values[w.node][sp.name];
            if (x[port.name] === undefined || x[port.name] === null) {
              throw new Error(`${port.name}: ${w.node} has no ${sp.label || sp.name} (e.g. the sacrum has no inferior endplate)`);
            }
          } else if (WIRE_ONLY.has(port.type)) {
            throw new Error(`connect a ${port.type} to "${port.label || port.name}"`);
          } else if (v === null || v === undefined) {
            throw new Error(`${port.name}: missing value`);
          } else {
            x[port.name] = literal(id, title, port, v, params);
          }
        }
        values[id] = run(id, raw, blk, x, title, library, params);
        if (raw.block === "output") outputs.push({ id, x });
      } catch (e) {
        errors[id] = e.message;
      }
    }
    return { errors, values, params, outputs, blocks };
  }

  function run(id, raw, blk, x, title, library, params) {
    const name = raw.block;
    if (name === "number") {
      const lo = Math.min(x.min, x.value), hi = Math.max(x.max, x.value);
      const p = { name: id, uniform: `u_${id}`, value: x.value, default: x.value, min: lo, max: hi,
        step: x.step || (hi - lo) / 200, label: x.label || title, unit: x.unit, node: id, port: "value", number: true };
      params.push(p);
      return { value: { g: p.uniform, iv: [lo, hi], param: p } };
    }
    if (name === "vector") return { vector: { c: [x.x, x.y, x.z] } };
    if (name === "output") return {};
    if (name === "bone") {
      const b = library && library.bones[x.file];
      if (!b) throw new Error(`bone file not in this scene's library: ${x.file}`);
      return { bone: { node: id, tmpl: b.bone, inputs: {}, bf: () => b.bounds, lib: x.file } };
    }
    if (name === "landmarks") {
      const b = library && library.bones[x.bone.lib];
      const lm = b && b.landmarks;
      if (!lm) throw new Error("no landmarks for this bone (vertebrae and sacrum only)");
      const plane = (e) => (e ? { point: cvec(e.point), normal: cvec(e.normal) } : null);
      return {
        frame: { origin: cvec(lm.origin), x: cvec(lm.axes[0]), y: cvec(lm.axes[1]), z: cvec(lm.axes[2]) },
        superior: plane(lm.sup), inferior: plane(lm.inf),
        markers: { node: `${id}.markers`, tmpl: lm.markers, inputs: {}, bf: () => lm.markers_bounds },
      };
    }
    if (name === "plane") return { plane: { point: x.point, normal: x.normal } };
    if (name === "plane_parts") return { point: x.plane.point, normal: x.plane.normal };
    if (name === "frame_parts") return { origin: x.frame.origin, x: x.frame.x, y: x.frame.y, z: x.frame.z };
    if (name === "smooth") {
      const b = library.bones[x.bone.lib];
      if (!b) throw new Error("smooth needs a bone directly");
      return { body: { node: id, tmpl: b.smooth, inputs: x, bf: (iv) => BOUNDS.smooth(x, iv) } };
    }
    const key = blk.inputs.filter((p) => p.type === "choice").map((p) => `${p.name}=${x[p.name]}`).join(",");
    const tmpl = blk.glsl && blk.glsl[key];
    if (!tmpl) throw new Error(`no GLSL for block ${name} (${key})`);
    return { body: { node: id, tmpl, inputs: x, bf: BOUNDS[name] ? (iv) => BOUNDS[name](x, iv) : null } };
  }

  // ------------------------------------------------------------ GLSL
  function emitter() {
    const ctx = { lines: [], n: 0, memo: new Map() };
    ctx.var = (kind, expr) => {
      ctx.n += 1;
      const v = `${kind === "vec3" ? "p" : "d"}${ctx.n}`;
      ctx.lines.push(`    ${kind} ${v} = ${expr};`);
      return v;
    };
    ctx.sub = (body, p) => {
      const key = `${body.node}|${p}`;
      if (!ctx.memo.has(key)) ctx.memo.set(key, expand(ctx, body, p));
      return ctx.memo.get(key);
    };
    return ctx;
  }

  // "${radius}", "${center.x}", "${plane.normal.y}": walk the input value to a scalar.
  function scalarGlsl(inputs, name) {
    let v = inputs;
    for (const part of name.split(".")) {
      if (v && v.c && "xyz".includes(part) && part.length === 1) v = v.c["xyz".indexOf(part)];
      else v = v ? v[part] : undefined;
    }
    if (!v || typeof v.g !== "string") throw new Error(`template input ${name} missing`);
    return /^[A-Za-z_][A-Za-z0-9_]*$/.test(v.g) ? v.g : `(${v.g})`;
  }

  function expand(ctx, body, p) {
    const res = [];
    const subst = (s) => s.replace(/\$\{([^}]+)\}|\$(p|\d+)/g, (_, inp, tok) =>
      inp ? scalarGlsl(body.inputs, inp) : tok === "p" ? p : res[+tok]);
    for (const op of body.tmpl.ops) {
      if (op[0] === "var") res.push(ctx.var(op[1], subst(op[2])));
      else res.push(ctx.sub(body.inputs[op[1]], subst(op[2])));
    }
    return subst(body.tmpl.out);
  }

  // Scene JSON (same shape as implicit.scene.Scene.to_json) from an evaluated graph.
  function sceneOf(graph, manifest, library, ev) {
    const bodies = [], groups = [], groupInfo = [];
    const names = new Set();
    for (const o of ev.outputs) {
      const name = o.x.name || o.id;
      if (names.has(name)) { ev.errors[o.id] = `duplicate body name "${name}"`; continue; }
      names.add(name);
      const group = o.x.group || name;
      if (!groups.includes(group)) { groups.push(group); groupInfo.push({ name: group, color: o.x.color, visible: o.x.visible }); }
      const body = o.x.body;
      bodies.push({ name, group: groups.indexOf(group), node: o.id, body, color: o.x.color, bounds: child(body, rangeIv) });
    }
    const known = bodies.map((b) => b.bounds).filter(Boolean);
    let bounds;
    if (graph.bounds) bounds = graph.bounds;
    else if (known.length) bounds = known.reduce(unionB);
    else bounds = [[-50, -50, -50], [50, 50, 50]];

    // Only parameters used by output bodies become uniforms (as in Python).
    const used = new Set();
    const lines = [];
    const n = Math.max(bodies.length, 1);
    const cols = bodies.length ? bodies.map((b) => `vec3(${b.color.map(glslFloat).join(", ")})`) : ["vec3(1.0)"];
    const units = [];
    bodies.forEach((b, k) => {
      const ctx = emitter();
      const d = ctx.sub(b.body, "p");
      units.push(`vec2 unit${k}(vec3 p) {  // ${b.name}\n${ctx.lines.join("\n")}\n    return vec2(${d}, ${k}.0);\n}`);
      for (const m of (ctx.lines.join("\n") + d).matchAll(/\bu_[A-Za-z0-9_]+/g)) used.add(m[0]);
    });
    const params = ev.params.filter((p) => used.has(p.uniform));
    lines.push(...params.map((p) => `uniform float ${p.uniform};`));
    lines.push(...groups.map((_, i) => `uniform float u_grp${i};`));
    lines.push(manifest.glsl_lib);
    lines.push(`const vec3 BODY_COLORS[${n}] = vec3[${n}](${cols.join(", ")});`);
    lines.push("vec3 bodyColor(int i) { return BODY_COLORS[i]; }");
    lines.push(...units);
    lines.push("vec2 sceneMap(vec3 p) {\n    vec2 r = vec2(1e10, -1.0);\n    vec2 d;");
    bodies.forEach((b, k) => {
      let cond = `(u_grp${b.group} > 0.5)`;
      const bb = b.bounds;
      if (bb) cond += ` && sdAABB(p, vec3(${bb[0].map(glslFloat).join(", ")}), vec3(${bb[1].map(glslFloat).join(", ")})) < r.x`;
      lines.push(`    if (${cond}) { d = unit${k}(p); if (d.x < r.x) r = d; }`);
    });
    lines.push("    return r;\n}");
    const glsl = lines.join("\n");
    return {
      name: graph.name || "graph", glsl, bounds,
      params: params.map(({ name, uniform, value, default: def, min, max, step, label, unit, node, port, number }) =>
        ({ name, uniform, value, default: def, min, max, step, label, unit, node, port, number: !!number })),
      groups: groupInfo, bodies: bodies.map((b) => ({ name: b.name, group: b.group, node: b.node })),
      attribution: library ? library.attribution : "",
      frame_offset: library ? library.frame_offset : null,
      atlas: library ? library.atlas : null,
      graph: true, hash: hashOf(glsl),
      view_bounds: viewBounds(bodies, {}),
      _bodies: bodies,
    };
  }

  // Box around every body at the given slider values (for framing the camera).
  function viewBounds(bodies, values) {
    const bs = bodies.map((b) => child(b.body, nowIv(values))).filter(Boolean);
    return bs.length ? bs.reduce(unionB) : null;
  }

  function hashOf(s) {
    let h = 2166136261;
    for (let i = 0; i < s.length; i++) h = Math.imul(h ^ s.charCodeAt(i), 16777619);
    return (h >>> 0).toString(16);
  }

  // Node ids upstream of each node (for highlighting the bodies a node feeds).
  function upstream(graph) {
    const nodes = graph.nodes || {}, memo = {};
    const up = (id, seen = new Set()) => {
      if (memo[id]) return memo[id];
      if (seen.has(id)) return new Set();
      seen.add(id);
      const s = new Set([id]);
      for (const v of Object.values((nodes[id] || {}).in || {})) {
        const w = parseWire(v);
        if (w && nodes[w.node]) for (const u of up(w.node, seen)) s.add(u);
      }
      return (memo[id] = s);
    };
    return up;
  }

  function build(graph, manifest, library) {
    const ev = evaluate(graph, manifest, library);
    const scene = sceneOf(graph, manifest, library, ev);
    return { scene, errors: ev.errors, params: ev.params, upstream: upstream(graph) };
  }

  return { build, parseWire, ACCEPTS, WIRED, WIRE_ONLY, glslFloat, colorOf,
    viewBounds: (scene, values) => viewBounds(scene._bodies || [], values) };
})();

if (typeof module !== "undefined") module.exports = Graph;
