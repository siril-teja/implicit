// Block graph -> scene shader, in the browser (mirrors implicit/graph.py build_graph
// and Scene.glsl). Block GLSL comes from blocks.json (traced from the Python
// emitters), bone GLSL from <scene>.library.json, so nothing is implemented twice.
// The result has the same shape as the Python scene JSON, so the viewer and the
// GPU check use it unchanged.
"use strict";

const Graph = (() => {
  const ACCEPTS = { body: ["body", "bone"], bone: ["bone"], scalar: ["scalar"], vec3: ["vec3"],
    plane: ["plane"], frame: ["frame"], vertebra: ["vertebra"], alignment: ["alignment"] };
  const WIRED = new Set(["body", "bone", "scalar", "vec3", "plane", "frame", "vertebra", "alignment"]);
  const WIRE_ONLY = new Set(["body", "bone", "plane", "frame", "vertebra", "alignment"]);

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
    pose(x, iv) {
      const b = child(x.body, iv), o = vecIv(x.vertebra.frame.origin, iv), s = vecIv(x.shift, iv);
      if (!b || !o || !s) return null;
      const corners = [];
      for (const i of [0, 1]) for (const j of [0, 1]) for (const k of [0, 1]) corners.push([b[i][0], b[j][1], b[k][2]]);
      if (iv.now) {  // exact: the corners moved by the current pose
        const E = ["x", "y", "z"].map((a) => vecIv(x.vertebra.frame[a], iv));
        if (E.some((e) => !e)) return null;
        const e = E.map((r) => r[0]), oo = o[0], sh = s[0];
        const [f, l, a] = ["flexion", "lateral", "axial"].map((k) => iv(x[k])[0] * Math.PI / 180);
        const rot = (v, ax, ang) => {
          const [i, j] = [[1, 2], [2, 0], [0, 1]][ax], c = Math.cos(ang), sn = Math.sin(ang), r = v.slice();
          r[i] = c * v[i] - sn * v[j]; r[j] = sn * v[i] + c * v[j]; return r;
        };
        const dot = (u, v) => u[0] * v[0] + u[1] * v[1] + u[2] * v[2];
        const lo = [Infinity, Infinity, Infinity], hi = [-Infinity, -Infinity, -Infinity];
        for (const c of corners) {
          const q = c.map((v, k) => v - oo[k]);
          const loc = rot(rot(rot([dot(q, e[0]), dot(q, e[1]), dot(q, e[2])], 0, -f), 1, l), 2, a);
          for (let k = 0; k < 3; k++) {
            const w = oo[k] + e[0][k] * (loc[0] + sh[0]) + e[1][k] * (loc[1] + sh[1]) + e[2][k] * (loc[2] + sh[2]);
            lo[k] = Math.min(lo[k], w); hi[k] = Math.max(hi[k], w);
          }
        }
        return [lo, hi];
      }
      let r = 0;
      for (const c of corners) r = Math.max(r, Math.hypot(...c.map((v, a) => Math.max(Math.abs(v - o[0][a]), Math.abs(v - o[1][a])))));
      r += Math.hypot(...[0, 1, 2].map((a) => Math.max(Math.abs(s[0][a]), Math.abs(s[1][a]))));
      return [o[0].map((v) => v - r), o[1].map((v) => v + r)];
    },
    show_plane(x, iv) {
      const p = vecIv(x.plane.point, iv), r = upper(x.radius, iv), t = upper(x.thickness, iv);
      return p && r !== null && t !== null ? [p[0].map((v) => v - r - t), p[1].map((v) => v + r + t)] : null;
    },
    show_vertebra(x, iv) {
      const v = x.vertebra, r = upper(x.radius, iv), l = upper(x.length, iv);
      const parts = [vecIv(v.frame.origin, iv), ...[v.sup, v.inf].filter(Boolean).map((pl) => vecIv(pl.point, iv))];
      if (parts.some((q) => !q) || r === null || l === null) return null;
      const e = Math.max(r, l) + 0.6 + 0.35;
      return [[0, 1, 2].map((k) => Math.min(...parts.map((q) => q[0][k])) - e),
        [0, 1, 2].map((k) => Math.max(...parts.map((q) => q[1][k])) + e)];
    },
    show_frame(x, iv) {
      const o = vecIv(x.frame.origin, iv), l = upper(x.length, iv), r = upper(x.radius, iv);
      return o && l !== null && r !== null ? [o[0].map((v) => v - l - r), o[1].map((v) => v + l + r)] : null;
    },
  };
  const rangeIv = (s) => s.iv || null;
  const nowIv = (values) => Object.assign((s) => {
    if (s.param) {
      const v = s.param.name in values ? values[s.param.name] : s.param.value;
      return [v, v];
    }
    if (s.ev) {  // derived value (e.g. a posed frame): evaluate its expression
      const v = s.ev(values);
      return Number.isFinite(v) ? [v, v] : null;
    }
    return s.iv || null;
  }, { now: true });

  // JavaScript evaluator for a derived scalar's GLSL (arithmetic, sin, cos, radians only).
  function evaluator(g, params) {
    if (!/^[\w\s.+\-*/(),]*$/.test(g)) return null;
    const ids = new Set(g.match(/[A-Za-z_]\w*/g) || []);
    for (const w of ids) if (!["cos", "sin", "radians", "e"].includes(w) && !w.startsWith("u_")) return null;
    const byUniform = Object.fromEntries(params.map((p) => [p.uniform, p]));
    const js = g.replace(/\bu_\w+/g, (u) => `V(${JSON.stringify(u)})`).replace(/\bradians\(/g, "R(")
      .replace(/\bcos\(/g, "Math.cos(").replace(/\bsin\(/g, "Math.sin(");
    let f;
    try { f = new Function("V", "R", `return (${js});`); } catch { return null; }
    return (values) => f((u) => { const p = byUniform[u]; return p ? (p.name in values ? values[p.name] : p.value) : NaN; },
      (d) => d * Math.PI / 180);
  }
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
          } else if (WIRE_ONLY.has(port.type) && port.optional) {
            x[port.name] = null;
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
      const frame = { origin: cvec(lm.origin), x: cvec(lm.axes[0]), y: cvec(lm.axes[1]), z: cvec(lm.axes[2]) };
      return {
        frame, superior: plane(lm.sup), inferior: plane(lm.inf),
        vertebra: { frame, sup: plane(lm.sup), inf: plane(lm.inf) },
        markers: { node: `${id}.markers`, tmpl: lm.markers, inputs: {}, bf: () => lm.markers_bounds },
      };
    }
    if (name === "plane") return { plane: { point: x.point, normal: x.normal } };
    if (name === "plane_parts") return { point: x.plane.point, normal: x.plane.normal };
    if (name === "frame_parts") return { origin: x.frame.origin, x: x.frame.x, y: x.frame.y, z: x.frame.z };
    if (name === "vertebra_parts") return { frame: x.vertebra.frame, superior: x.vertebra.sup, inferior: x.vertebra.inf };
    if (name === "smooth") {
      const b = library.bones[x.bone.lib];
      if (!b) throw new Error("smooth needs a bone directly");
      return { body: { node: id, tmpl: b.smooth, inputs: x, bf: (iv) => BOUNDS.smooth(x, iv) } };
    }
    if (name === "spinopelvic") return { alignment: spinopelvic(id, x, library) };
    if (name === "align") return align(id, x, params);
    if (name === "spine_curve") return { body: spineCurve(id, x) };
    if (name === "tps_warp") return { body: tpsWarp(id, x, params) };
    if (name === "field_view") return { body: fieldView(id, x, params) };
    // Template variant: choice inputs, then optional parts present (1) or absent (0).
    const key = [...blk.inputs.filter((p) => p.type === "choice").map((p) => `${p.name}=${x[p.name]}`),
      ...Object.entries(blk.optional || {}).flatMap(([inp, parts]) => parts.map((pt) => `${inp}.${pt}=${x[inp] && x[inp][pt] ? 1 : 0}`))]
      .join(",");
    const tmpls = blk.glsl && blk.glsl[key];
    if (!tmpls) throw new Error(`no GLSL for block ${name} (${key})`);
    const out = {};
    for (const op of blk.outputs) {
      const tmpl = tmpls[op.name];
      if (op.type === "body") {
        out[op.name] = { node: op.name === "body" ? id : `${id}:${op.name}`, tmpl, inputs: x,
          bf: BOUNDS[name] ? (iv) => BOUNDS[name](x, iv, op.name) : null };
      } else {
        out[op.name] = valueFrom(tmpl, x, params);
      }
    }
    return out;
  }

  // ------------------------------------------------------------ spinopelvic alignment
  // As implicit/graph.py: the chain model runs on the CPU (solvers.js) and each Align
  // block reads its segment's motion (cos a, sin a, t_y, t_z) as computed uniforms.
  const KNOB_NAMES = ["PI", "PT", "LL", "TK", "MT", "TL", "AR", "PO", "CB"];
  function spinopelvic(id, x, library) {
    const b = library && library.bones[x.bone.lib];
    if (!b || !b.spine) throw new Error("no spine and pelvis landmarks for this subject");
    const inputs = {}, knobs = {};
    for (const k of KNOB_NAMES) {
      const s = x[k];
      if (!s || !s.param || s.param.computed) throw new Error(`${k}: needs a value or a Number block`);
      inputs[k] = s.param.name;
      knobs[s.param.name] = s.param;
    }
    const curves = {};
    for (const k of ["MT", "TL"]) {  // curves whose vertebrae the subject lacks are left out
      const c = ["upper", "apex", "lower"].map((part) => x[`${k}_${part}`]);
      if (c.every((lv) => b.spine.levels.includes(lv))) curves[k] = c;
    }
    const solver = { id, kind: "spinopelvic", data: { ...b.spine, curves, field: x.field }, inputs };
    Solvers.run(solver, (n) => knobs[n].value);  // reports a bad curve (end/apex order) here
    return { solver, knobs, motions: {}, samples: null, tables: {} };
  }

  const sampled = (a) => a.samples
    || (a.samples = Solvers.samples(a.solver, (n) => { const p = a.knobs[n]; return [p.min, p.value, p.max]; }));
  const computedParam = (s, name, key, index, value, vals, params) => {
    const p = { name, uniform: `u_${name}`, value, default: value, min: Math.min(...vals), max: Math.max(...vals),
      step: 0, label: name, unit: "", node: s.id, port: key, computed: { solver: s.id, key, index }, solver: s };
    params.push(p);
    return p;
  };
  const rigNp = (m, q) => [0, 1, 2].map((i) => m[i] * q[0] + m[3 + i] * q[1] + m[6 + i] * q[2] + m[9 + i]);

  function align(id, x, params) {
    const a = x.alignment, seg = x.segment, s = a.solver;
    const res = Solvers.run(s, (n) => a.knobs[n].value);
    if (!(seg in res) || seg === "tps") throw new Error(`segment "${seg}" is not in this alignment`);
    if (!a.motions[seg]) {
      a.motions[seg] = res[seg].map((v, i) => {
        const p = computedParam(s, `${s.id}_${seg}_${i}`, seg, i, v, sampled(a).map((r) => r[seg][i]), params);
        return { g: p.uniform, iv: [p.min, p.max], param: p };
      });
    }
    const m = a.motions[seg], g = m.map((q) => q.g);
    const bf = (iv) => {
      const bb = child(x.body, iv);
      if (!bb) return null;
      const ms = iv.now ? [m.map((q) => iv(q)[0])] : sampled(a).map((r) => r[seg]);
      const lo = [Infinity, Infinity, Infinity], hi = [-Infinity, -Infinity, -Infinity];
      for (const mm of ms) for (const i of [0, 1]) for (const j of [0, 1]) for (const k of [0, 1]) {
        rigNp(mm, [bb[i][0], bb[j][1], bb[k][2]]).forEach((v, n) => { lo[n] = Math.min(lo[n], v); hi[n] = Math.max(hi[n], v); });
      }
      return [lo, hi];
    };
    const mv = [0, 3, 6, 9].map((i) => `vec3(${g[i]}, ${g[i + 1]}, ${g[i + 2]})`).join(", ");
    const box = child(x.body, rangeIv);
    const out = {
      body: { node: id, inputs: { body: x.body }, bf, local: box ? { m: mv, box } : null,  // local box: culling (scene.py)
        tmpl: { ops: [["var", "vec3", `rigInv($p, ${mv})`], ["sub", "body", "$0"]], out: "$1" } },
      vertebra: null,
    };
    const v = x.vertebra;
    if (v) {
      // One rigPoint/rigDir call per vector (components .x/.y/.z, collapsed downstream).
      const vecG = (vec) => {
        const gg = vec.c.map((q) => q.g.replace(/^\((.*)\)$/, "$1")), m0 = gg[0].match(/^(.*)\.x$/);
        return m0 && gg[1] === `${m0[1]}.y` && gg[2] === `${m0[1]}.z` ? m0[1] : `vec3(${gg.join(", ")})`;
      };
      const rig = (vec, move) => {
        const call = `${move ? "rigPoint" : "rigDir"}(${vecG(vec)}, ${mv})`;
        return { c: ["x", "y", "z"].map((c) => ({ g: `${call}.${c}`, iv: null, ev: null })) };
      };
      const plane = (pl) => (pl ? { point: rig(pl.point, true), normal: rig(pl.normal, false) } : null);
      const f = v.frame;
      out.vertebra = { frame: { origin: rig(f.origin, true), x: rig(f.x, false), y: rig(f.y, false), z: rig(f.z, false) },
        sup: plane(v.sup), inf: plane(v.inf) };
    }
    return out;
  }

  // The subject's spine curve, unmoved (a TPS warp carries it with the alignment).
  function spineCurve(id, x) {
    const P = x.alignment.solver.data.curve, pts = [];
    for (let i = 0; i < P.length; i += 3) pts.push(`vec3(${[P[i], P[i + 1], P[i + 2]].map(glslFloat).join(", ")})`);
    // As Tube in sdf.py: chunks of segments skipped when their box is farther than the
    // nearest segment so far (exact).
    const ops = [], CHUNK = 8;
    let d = "1e10";
    for (let i = 0; i + 1 < pts.length; i += CHUNK) {
      const c = pts.slice(i, i + CHUNK + 1);
      const fold = (f) => `${f}(`.repeat(c.length - 1) + c[0] + c.slice(1).map((q) => `, ${q})`).join("");
      ops.push(["var", "vec3", fold("min")], ["var", "vec3", fold("max")]);
      const lo = `$${ops.length - 2}`, hi = `$${ops.length - 1}`;
      let caps = d;
      for (let j = 0; j + 1 < c.length; j++) caps = `min(${caps}, sdCap($p, ${c[j]}, ${c[j + 1]}, \${radius}))`;
      ops.push(["var", "float", `sdAABB($p, ${lo} - \${radius}, ${hi} + \${radius}) < ${d} ? ${caps} : ${d}`]);
      d = `$${ops.length - 1}`;
    }
    const bf = (iv) => {
      const r = upper(x.radius, iv);
      if (r === null) return null;
      const lo = [Infinity, Infinity, Infinity], hi = [-Infinity, -Infinity, -Infinity];
      P.forEach((v, i) => { lo[i % 3] = Math.min(lo[i % 3], v - r); hi[i % 3] = Math.max(hi[i % 3], v + r); });
      return [lo, hi];
    };
    return { node: id, inputs: { radius: x.radius }, bf, tmpl: { ops, out: `$${ops.length - 1}` } };
  }

  // Computed tables of an alignment ("tps", "chain", "grid") as texture params.
  function tableParam(a, key, params) {
    const s = a.solver;
    if (!a.tables[key]) {
      const width = { tps: 3, chain: 7, flow_segs: 512 }[key] || { grid: s.data.grid.n[0], flow_cells: s.data.flow.n[0] }[key];
      a.tables[key] = { name: `${s.id}_${key}`, uniform: `u_${s.id}_${key}`, value: 0, default: 0, min: 0, max: 0, step: 0,
        label: `${s.id}_${key}`, unit: "", node: s.id, port: key, texture: { solver: s.id, key, width }, solver: s };
      params.push(a.tables[key]);
    }
    return a.tables[key];
  }
  // The soft-tissue field: the GLSL of its inverse map at a point (as Warp in sdf.py).
  function fieldInverse(a, params) {
    const d = a.solver.data, tps = d.field !== "polyrigid";
    const u = tableParam(a, tps ? "tps" : "chain", params).uniform;
    const n = tps ? d.tps_points.length : 1 + d.levels.length;
    return (p) => `${tps ? "tpsInv" : "polyInv"}(${p}, ${u}, ${n})`;
  }
  // Largest control point displacement, now or over the slider ranges.
  function fieldDisp(a, iv) {
    const s = a.solver;
    const res = iv.now ? [Solvers.run(s, (nm) => iv({ param: a.knobs[nm] })[0])] : sampled(a);
    let mx = 0;
    for (const r of res) for (const q of s.data.tps_points) {
      const y = rigNp(r[q.segment], q.point);
      mx = Math.max(mx, Math.hypot(y[0] - q.point[0], y[1] - q.point[1], y[2] - q.point[2]));
    }
    return mx;
  }
  const WARP_LIP = 0.8;
  function tpsWarp(id, x, params) {
    const a = x.alignment, inv = fieldInverse(a, params);
    const bf = (iv) => { const bb = child(x.body, iv); return bb ? grow(bb, fieldDisp(a, iv) + 2.0) : null; };
    const ops = [["var", "vec3", inv("$p")], ["sub", "body", "$0"], ["var", "float", `${glslFloat(WARP_LIP)} * $1`]];
    return { node: id, inputs: { body: x.body }, bf, tmpl: { ops, out: "$2" } };
  }

  // Field view (as graph.py): lattice carried by the field, arrows of the display grid, or
  // a slice coloured by the displacement (``colormap``: per-point colour).
  function fieldView(id, x, params) {
    const a = x.alignment, g = a.solver.data.grid, inv = fieldInverse(a, params);
    const lo = g.lo, hi = g.lo.map((v, i) => v + g.step * (g.n[i] - 1));
    const v3 = (q) => `vec3(${q.map(glslFloat).join(", ")})`, box = [lo, hi];
    const sg = (s) => (/^[A-Za-z_]\w*$/.test(s.g) ? s.g : `(${s.g})`);
    if (x.mode === "flow") {
      const f = a.solver.data.flow, C = tableParam(a, "flow_cells", params).uniform, S = tableParam(a, "flow_segs", params).uniform;
      const call = (p, r) => `flowLines(${p}, ${C}, ${S}, ${v3(f.lo)}, ${glslFloat(f.cell)}, ivec3(${f.n.join(", ")}), ${r})`;
      const fb = [f.lo, f.lo.map((v, i) => v + f.cell * f.n[i])];
      return { node: id, inputs: x, bf: () => fb, tmpl: { ops: [["var", "float", `${call("$p", "${radius}")}.x`]], out: "$0" },
        colormap: { glsl: (p) => `${call(p, sg(x.radius))}.y`, max: x.max } };
    }
    if (x.mode === "lattice") {
      const ops = [["var", "vec3", inv("$p")],
        ["var", "float", `${glslFloat(WARP_LIP)} * sdLattice($0, ${v3(lo)}, ${v3(hi)}, \${spacing}, \${radius})`]];
      return { node: id, inputs: x, bf: (iv) => grow(box, fieldDisp(a, iv) + 2.0), tmpl: { ops, out: "$1" } };
    }
    if (x.mode === "arrows") {
      const u = tableParam(a, "grid", params).uniform;
      const call = (p, r) => `arrows(${p}, ${u}, ${v3(lo)}, ${glslFloat(g.step)}, ivec3(${g.n.join(", ")}), ${r})`;
      return { node: id, inputs: x, bf: () => grow(box, g.step), tmpl: { ops: [["var", "float", `${call("$p", "${radius}")}.x`]], out: "$0" },
        colormap: { glsl: (p) => `${call(p, sg(x.radius))}.y`, max: x.max } };
    }
    const wide = grow(box, 20.0), c = "xyz"["xyz".indexOf(x.axis)];  // the slice spans the display grid
    const ops = [["var", "float", `max(abs($p.${c} - \${offset}) - 0.75, sdAABB($p, ${v3(wide[0])}, ${v3(wide[1])}))`]];
    return { node: id, inputs: x, bf: () => wide, tmpl: { ops, out: "$0" },
      colormap: { glsl: (p) => `length(${inv(p)} - ${p})`, max: x.max } };
  }

  // A value output (vector, plane, frame, vertebra) from its component expressions; a part
  // whose inputs are missing (e.g. the sacrum's inferior endplate) becomes null.
  function valueFrom(t, inputs, params) {
    if (t === null || t === undefined) return null;
    const scalar = (s) => {
      const g = collapseVec(s, inputs).replace(/\$\{([^}]+)\}/g, (_, inp) => scalarGlsl(inputs, inp));
      return { g, iv: null, ev: evaluator(g, params) };
    };
    if (typeof t === "string") return scalar(t);
    if (Array.isArray(t)) return { c: t.map(scalar) };
    const out = {};
    for (const [k, v] of Object.entries(t)) {
      try { out[k] = valueFrom(v, inputs, params); } catch (e) {
        if (v && !Array.isArray(v) && typeof v === "object") out[k] = null; else throw e;
      }
    }
    return out;
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

  // vec3(${a.x}, ${a.y}, ${a.z}) where the three are the parts of one vector call
  // (e.g. posePoint(...).x/.y/.z): emit the call once, as the Python side does.
  function collapseVec(s, inputs) {
    return s.replace(/vec3\(\$\{([^}]+)\.x\}, \$\{\1\.y\}, \$\{\1\.z\}\)/g, (whole, base) => {
      const g = ["x", "y", "z"].map((a) => scalarGlsl(inputs, `${base}.${a}`).replace(/^\((.*)\)$/, "$1"));
      const m = g[0].match(/^(.*)\.x$/);
      return m && g[1] === `${m[1]}.y` && g[2] === `${m[1]}.z` ? m[1] : whole;
    });
  }

  function expand(ctx, body, p) {
    const res = [];
    const subst = (s) => collapseVec(s, body.inputs).replace(/\$\{([^}]+)\}|\$(p|\d+)/g, (_, inp, tok) =>
      inp ? scalarGlsl(body.inputs, inp) : tok === "p" ? p : res[+tok]);
    for (const op of body.tmpl.ops) {
      if (op[0] === "raw") { ctx.lines.push(subst(op[1])); res.push(null); } else if (op[0] === "var") res.push(ctx.var(op[1], subst(op[2])));
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
    // Field-coloured bodies (as scene.py): their colour code, before the uniforms it uses.
    const colorLines = ["vec3 bodyColor(int i, vec3 p) {"];
    bodies.forEach((b, k) => {
      const cm = b.body.colormap;
      if (!cm) return;
      const mx = /^[A-Za-z_]\w*$/.test(cm.max.g) ? cm.max.g : `(${cm.max.g})`;
      const line = `    if (i == ${k}) return fieldColor((${cm.glsl("p")}) / max(${mx}, 1e-6));`;
      colorLines.push(line);
      for (const m of line.matchAll(/\bu_[A-Za-z0-9_]+/g)) used.add(m[0]);
    });
    colorLines.push("    return BODY_COLORS[i];\n}");
    // Computed uniforms need their solver's inputs (sliders, not in the shader).
    const solvers = {};
    for (const p of ev.params) {
      if (!(p.computed || p.texture) || !used.has(p.uniform)) continue;
      solvers[p.solver.id] = { kind: p.solver.kind, data: p.solver.data, inputs: p.solver.inputs };
      for (const n of Object.values(p.solver.inputs)) used.add(`u_${n}`);
    }
    const params = ev.params.filter((p) => used.has(p.uniform));
    lines.push(...params.map((p) => `uniform ${p.texture ? "highp sampler2D" : "float"} ${p.uniform};`));
    lines.push(...groups.map((_, i) => `uniform float u_grp${i};`));
    lines.push(manifest.glsl_lib);
    lines.push(`const vec3 BODY_COLORS[${n}] = vec3[${n}](${cols.join(", ")});`, ...colorLines);
    lines.push(...units);
    lines.push("vec2 sceneMap(vec3 p) {\n    vec2 r = vec2(1e10, -1.0);\n    vec2 d;");
    bodies.forEach((b, k) => {
      let cond = `(u_grp${b.group} > 0.5)`;
      const bb = b.bounds, loc = b.body.local;
      const v3 = (a) => `vec3(${a.map(glslFloat).join(", ")})`;
      if (loc) cond += ` && sdAABB(rigInv(p, ${loc.m}), ${v3(loc.box[0])}, ${v3(loc.box[1])}) < r.x`;  // moved body: own box
      else if (bb) cond += ` && sdAABB(p, ${v3(bb[0])}, ${v3(bb[1])}) < r.x`;
      lines.push(`    if (${cond}) { d = unit${k}(p); if (d.x < r.x) r = d; }`);
    });
    lines.push("    return r;\n}");
    const glsl = lines.join("\n");
    return {
      name: graph.name || "graph", glsl, bounds,
      params: params.map(({ name, uniform, value, default: def, min, max, step, label, unit, node, port, number, computed, texture }) =>
        ({ name, uniform, value, default: def, min, max, step, label, unit, node, port, number: !!number,
          ...(computed ? { computed } : {}), ...(texture ? { texture } : {}) })),
      solvers,
      groups: groupInfo, bodies: bodies.map((b) => ({ name: b.name, group: b.group, node: b.node,
        ...(b.body.colormap ? { colormap: { max: b.body.colormap.max.param ? b.body.colormap.max.param.name : b.body.colormap.max.iv[0] } } : {}) })),
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

  // Box the rays march in: the visible bodies at the current values (``vis``: per group),
  // or null when one of them has no bounds. Tighter than the scene bounds, which hold
  // every slider value (an alignment can move the spine far).
  function rayBounds(bodies, values, vis) {
    let box = null;
    for (const b of bodies) {
      if (!vis[b.group]) continue;
      const bb = child(b.body, nowIv(values));
      if (!bb) return null;
      box = box ? unionB(box, bb) : bb;
    }
    return box && grow(box, 0.5);
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
    viewBounds: (scene, values) => viewBounds(scene._bodies || [], values),
    rayBounds: (scene, values, vis) => (scene._bodies ? rayBounds(scene._bodies, values, vis) : null) };
})();

if (typeof module !== "undefined") module.exports = Graph;
