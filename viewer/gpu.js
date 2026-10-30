// Shared WebGL2 code for the viewer and the diagnostics page: shader sources,
// program building, texture uploads, GPU info. All functions take the GL context.
"use strict";

const GPU = (() => {
  const VERT = `#version 300 es
in vec2 a_pos;
void main() { gl_Position = vec4(a_pos, 0.0, 1.0); }`;

  const FRAG_HEAD = `#version 300 es
precision highp float;
precision highp sampler3D;
out vec4 outColor;
uniform vec2 u_res;
uniform vec3 u_eye, u_fwd, u_right, u_up;
uniform float u_fov;
uniform vec3 u_bmin, u_bmax;
uniform vec4 u_clip;     // xyz normal, w offset
uniform float u_clipOn;
uniform float u_pick;    // 1: write the body id under u_pickPx instead of a colour
uniform vec2 u_pickPx;
uniform highp uvec4 u_hl[2];  // highlighted bodies, one bit per body (first 256)
uniform float u_fast;    // 1 while the view is moving: cheaper shading (no ambient occlusion)
`;

  const FRAG_TAIL = `
vec2 mapClip(vec3 p) {
  vec2 m = sceneMap(p);
  if (u_clipOn > 0.5) m.x = max(m.x, dot(p, u_clip.xyz) - u_clip.w);
  return m;
}
vec2 boxHit(vec3 ro, vec3 rd) {
  vec3 inv = 1.0 / rd;
  vec3 t0 = (u_bmin - ro) * inv, t1 = (u_bmax - ro) * inv;
  vec3 tmin = min(t0, t1), tmax = max(t0, t1);
  return vec2(max(max(tmin.x, tmin.y), tmin.z), min(min(tmax.x, tmax.y), tmax.z));
}
bool highlighted(int i) {
  if (i < 0 || i >= 256) return false;
  uvec4 w = u_hl[i >> 7];
  int j = (i >> 5) & 3;
  uint bits = j == 0 ? w.x : j == 1 ? w.y : j == 2 ? w.z : w.w;
  return ((bits >> uint(i & 31)) & 1u) != 0u;
}
void main() {
  vec2 fc = u_pick > 0.5 ? u_pickPx : gl_FragCoord.xy;
  vec2 uv = (2.0 * fc - u_res) / u_res.y;
  vec3 rd = normalize(u_fwd + uv.x * u_fov * u_right + uv.y * u_fov * u_up);
  vec3 ro = u_eye;
  vec3 bg = mix(vec3(0.13, 0.14, 0.17), vec3(0.24, 0.26, 0.31), 0.5 + 0.5 * uv.y);
  vec2 bh = boxHit(ro, rd);
  if (bh.x > bh.y || bh.y < 0.0) { outColor = u_pick > 0.5 ? vec4(0.0) : vec4(pow(bg, vec3(0.4545)), 1.0); return; }
  // One loop, one call of the scene: march to the surface, then 3 samples for the
  // normal (forward differences), then 3 for ambient occlusion. Drivers inline every
  // call site of sceneMap, so a single site keeps compile time and code size down in
  // scenes of many bodies (D3D especially).
  float t = max(bh.x, 0.0);
  float id = -1.0, d0 = 0.0;
  float tol = u_fast > 0.5 ? 0.004 : 0.002;  // hit tolerance per mm of ray length
  const float e = 0.02;
  int last = u_pick > 0.5 ? 0 : (u_fast > 0.5 ? 3 : 6);  // final phase
  int phase = 0;  // 0 march, 1-3 normal, 4-6 occlusion
  vec3 p = ro, g = vec3(0.0), n = vec3(0.0, 0.0, 1.0);
  float occ = 0.0, w = 1.0;
  for (int i = 0; i < 310; i++) {
    vec3 q = phase == 0 ? ro + rd * t
           : phase <= 3 ? p + e * vec3(phase == 1, phase == 2, phase == 3)
           : p + n * (0.9 * float(phase - 3));
    vec2 m = mapClip(q);
    if (phase == 0) {
      if (m.x < tol * t + 0.01) { id = m.y; d0 = m.x; p = q; if (last == 0) break; phase = 1; continue; }
      t += 0.85 * m.x;
      if (t > bh.y || i >= 299) break;
    } else if (phase <= 3) {
      g[phase - 1] = m.x;
      if (phase == 3) n = normalize(g - d0);
      if (phase == last) break;
      phase++;
    } else {
      float h = 0.9 * float(phase - 3);
      occ += w * (h - m.x);
      w *= 0.7;
      if (phase == last) break;
      phase++;
    }
  }
  if (u_pick > 0.5) {
    float v = id + 1.0;
    outColor = vec4(mod(v, 256.0) / 255.0, floor(v / 256.0) / 255.0, 0.0, 1.0);
    return;
  }
  vec3 col = bg;
  if (id >= 0.0) {
    vec3 base = bodyColor(int(id));
    bool cut = u_clipOn > 0.5 && abs(dot(p, u_clip.xyz) - u_clip.w) < 0.05;
    if (cut) base *= 0.75 + 0.25 * step(0.5, fract((p.x + p.y + p.z) * 0.35));
    vec3 l1 = normalize(-u_fwd + u_up * 0.6 + u_right * 0.4);
    float dif = max(dot(n, l1), 0.0);
    float hemi = 0.5 + 0.5 * n.z;
    float ao = u_fast > 0.5 ? 0.85 : clamp(1.0 - 0.15 * occ, 0.0, 1.0);
    float spec = pow(max(dot(reflect(rd, n), l1), 0.0), 32.0);
    col = base * (0.85 * dif + 0.35 * hemi * ao) + 0.15 * spec;
    if (highlighted(int(id))) {
      float rim = pow(1.0 - max(dot(n, -rd), 0.0), 2.0);
      col = mix(col, vec3(0.42, 0.66, 1.0), 0.28) + vec3(0.35, 0.55, 1.0) * rim * 0.9;
    }
  }
  outColor = vec4(pow(col, vec3(0.4545)), 1.0);
}`;


  function compile(gl, type, src) {
    const s = gl.createShader(type);
    gl.shaderSource(s, src);
    gl.compileShader(s);
    if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) {
      const log = gl.getShaderInfoLog(s);
      gl.deleteShader(s);
      throw new Error(log);
    }
    return s;
  }

  // Program from a complete fragment shader source; returns { prog, uniforms, ms }.
  function program(gl, fragSrc) {
    const t0 = performance.now();
    const p = gl.createProgram();
    gl.attachShader(p, compile(gl, gl.VERTEX_SHADER, VERT));
    gl.attachShader(p, compile(gl, gl.FRAGMENT_SHADER, fragSrc));
    gl.bindAttribLocation(p, 0, "a_pos");
    gl.linkProgram(p);
    if (!gl.getProgramParameter(p, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(p));
    const uniforms = {};
    const n = gl.getProgramParameter(p, gl.ACTIVE_UNIFORMS);
    for (let i = 0; i < n; i++) {
      const info = gl.getActiveUniform(p, i);
      uniforms[info.name.replace("[0]", "")] = gl.getUniformLocation(p, info.name);
    }
    return { prog: p, uniforms, ms: performance.now() - t0 };
  }

  // Shared code for bodies built alike. Shader compilers inline every function at
  // every call site, and D3D's is slow on large programs, so a scene of many bodies
  // (40 in the average spine) can take most of a minute to compile on Windows, long
  // enough for the browser to give up. Units whose code differs only in float
  // literals and uniform names (vertebrae, their markers, left/right bones) become
  // one function ``grpK(p, row)`` called from one loop in sceneMap. Per-instance
  // numbers come from a float texture (u_inst; row: box min + visibility uniform,
  // box max, for a moved body the uniform indices of its motion, then the
  // instance's literals and uniform indices), uniforms through the
  // array u_SP. Only texture fetches and uniform-array reads: D3D compiles such
  // loops as loops (local arrays made it unroll them). ``unitN`` remain as one-line
  // wrappers (used by the diagnostics).
  // Returns { glsl, sp: uniform names in u_SP order, table: { data, w, h } | null,
  // where: { unit: [group, row] } for grouped units }.
  const FLOAT_RE = /\bu_\w+\b|(?:(?<=[(, ])-)?(?<![\w.])(?:(?:\d+\.\d*|\.\d+)(?:[eE][-+]?\d+)?|\d+[eE][-+]?\d+)(?![\w.])/g;  // a unary minus belongs to the literal
  function instanced(glsl) {
    const none = { glsl, sp: [], table: null, where: {} };
    const floatUniforms = new Set([...glsl.matchAll(/uniform float (u_\w+);/g)].map((m) => m[1]));
    const mapAt = glsl.indexOf("vec2 sceneMap(vec3 p) {");
    if (mapAt < 0) return none;
    // Units, and their line in sceneMap (visibility uniform, bounds).
    const units = new Map();
    for (const m of glsl.matchAll(/vec2 unit(\d+)\(vec3 p\) \{[^\n]*\n/g)) {
      const end = glsl.indexOf("\n}", m.index) + 2;
      units.set(+m[1], { start: m.index, end, body: glsl.slice(m.index + m[0].length, end - 2) });
    }
    // Box test in world space, or in the body's own frame for a moved body (rigInv:
    // the columns of its rotation and its translation, 12 uniforms).
    const lineRe = /^\s*if \(\((u_grp\d+) > 0\.5\)(?: && sdAABB\((?:p|rigInv\(p, ((?:vec3\([^)]*\)(?:, )?){4})\)), vec3\(([^)]*)\), vec3\(([^)]*)\)\) < r\.x)?\) \{ d = unit(\d+)\(p\); if \(d\.x < r\.x\) r = d; \}$/;
    const mapLines = glsl.slice(mapAt).split("\n");
    for (const [i, line] of mapLines.entries()) {
      const m = line.match(lineRe);
      if (!m || !units.has(+m[5])) continue;
      const box = (s, inf) => (s ? s.split(",").map(Number) : [inf, inf, inf]);
      const move = m[2] ? m[2].replace(/vec3\(|\)/g, "").split(",").map((s) => s.trim()) : null;
      if (move && !move.every((u) => floatUniforms.has(u))) continue;
      Object.assign(units.get(+m[5]), { line: i, vis: m[1], move, lo: box(m[3], -1e30), hi: box(m[4], 1e30) });
    }
    // Group by code skeleton (literals and float uniforms taken out).
    const groups = new Map();
    for (const [k, u] of units) {
      if (u.line === undefined || !floatUniforms.has(u.vis)) continue;
      u.slots = [];
      const skel = u.body.replace(FLOAT_RE, (s) => {
        if (s.startsWith("u_") && !floatUniforms.has(s)) return s;
        u.slots.push(s);
        return s.startsWith("u_") ? "\u0001U" : "\u0001F";
      });
      const key = (u.move ? "moved|" : "|") + skel;  // a group culls all its bodies alike
      if (!groups.has(key)) groups.set(key, []);
      groups.get(key).push(k);
    }
    const sp = new Map();  // uniform -> index in u_SP
    const spIndex = (name) => { if (!sp.has(name)) sp.set(name, sp.size); return sp.get(name); };
    const funcs = [], loops = [], drop = new Set(), wrappers = new Map(), rows = [], where = {};
    let g = 0;
    for (const [key, ks] of groups) {
      if (ks.length < 2) continue;
      const skel = key.slice(key.indexOf("|") + 1);
      const inst = ks.map((k) => units.get(k)), row0 = rows.length, moved = !!inst[0].move;
      const base = moved ? 5 : 2;  // first column of the literals (2-4: the motion's 12 uniforms)
      // Columns of per-instance values; slots equal in all instances stay inline,
      // identical columns are stored once.
      const cols = new Map(), expr = [];
      for (let s = 0; s < inst[0].slots.length; s++) {
        const vals = inst.map((u) => u.slots[s]);
        const isU = vals[0].startsWith("u_");
        if (isU ? vals.every((v) => v === vals[0]) : vals.every((v) => +v === +vals[0])) { expr.push(vals[0]); continue; }
        const key = (isU ? "U" : "F") + vals.join(",");
        if (!cols.has(key)) cols.set(key, { k: cols.size, isU, vals });
        expr.push(cols.get(key));
      }
      for (const [i, u] of inst.entries()) {
        const r = [...u.lo, spIndex(u.vis), ...u.hi, 0, ...(moved ? u.move.map(spIndex) : [])];
        for (const c of cols.values()) r.push(c.isU ? spIndex(c.vals[i]) : +c.vals[i]);
        rows.push(r);
      }
      const texel = (k) => `t${(k >> 2) + base}.${"xyzw"[k & 3]}`;
      const used = new Set([...cols.values()].map((c) => (c.k >> 2) + base));
      let s = 0;
      const body = skel.replace(/\u0001[FU]/g, () => {
        const e = expr[s++];
        if (typeof e === "string") return e;
        return e.isU ? `spGet(int(${texel(e.k)}))` : texel(e.k);
      });
      const fetches = [...used].sort((x, y) => x - y)
        .map((c) => `    vec4 t${c} = texelFetch(u_inst, ivec2(${c}, row), 0);`);
      funcs.push(`vec2 grp${g}(vec3 p, int row) {  // ${inst.length} bodies\n${fetches.join("\n")}\n${body}\n}`);
      const sp = (t) => `spGet(int(${t}))`, v3 = (a, b, c) => `vec3(${sp(a)}, ${sp(b)}, ${sp(c)})`;
      const at = moved ? `rigInv(p, ${v3("m0.x", "m0.y", "m0.z")}, ${v3("m0.w", "m1.x", "m1.y")}, `
        + `${v3("m1.z", "m1.w", "m2.x")}, ${v3("m2.y", "m2.z", "m2.w")})` : "p";
      // One visibility switch for the whole group (usual) is tested once, outside the loop.
      const oneVis = inst.every((u) => u.vis === inst[0].vis);
      const visTest = oneVis ? "" : "spGet(int(lo.w)) > 0.5 && ";
      loops.push(`    ${oneVis ? `if (${inst[0].vis} > 0.5) ` : ""}for (int row = ${row0}; row < ${row0 + inst.length}; row++) {
        vec4 lo = texelFetch(u_inst, ivec2(0, row), 0), hi = texelFetch(u_inst, ivec2(1, row), 0);${moved ? `
        vec4 m0 = texelFetch(u_inst, ivec2(2, row), 0), m1 = texelFetch(u_inst, ivec2(3, row), 0), m2 = texelFetch(u_inst, ivec2(4, row), 0);` : ""}
        if (${visTest}sdAABB(${at}, lo.xyz, hi.xyz) < r.x) { d = grp${g}(p, row); if (d.x < r.x) r = d; }
    }`);
      ks.forEach((k, i) => { drop.add(units.get(k).line); wrappers.set(k, `vec2 unit${k}(vec3 p) { return grp${g}(p, ${row0 + i}); }`); where[k] = [g, row0 + i]; });
      g++;
    }
    if (!g) return none;
    // Reassemble: unit definitions (grouped ones replaced by wrappers placed after
    // the shared code), then sceneMap without the grouped lines plus the loops.
    let head = "", at = 0;
    for (const [k, u] of [...units].sort((x, y) => x[1].start - y[1].start)) {
      head += glsl.slice(at, u.start);
      if (!wrappers.has(k)) head += glsl.slice(u.start, u.end);
      at = u.end;
    }
    head += glsl.slice(at, mapAt);
    const n4 = Math.ceil(sp.size / 4);
    const lib = `uniform vec4 u_SP[${n4}];
uniform highp sampler2D u_inst;
float spGet(int k) {
    vec4 v = u_SP[k >> 2];
    int c = k & 3;
    return c == 0 ? v.x : c == 1 ? v.y : c == 2 ? v.z : v.w;
}`;
    const kept = mapLines.filter((_, i) => !drop.has(i));
    kept.splice(kept.lastIndexOf("    return r;"), 0, ...loops);
    const w = Math.max(...rows.map((r) => Math.ceil(r.length / 4))), h = rows.length;
    const data = new Float32Array(w * h * 4);
    rows.forEach((r, i) => data.set(r, i * w * 4));
    return { glsl: [head, lib, ...funcs, ...wrappers.values(), kept.join("\n")].join("\n"), sp: [...sp.keys()], table: { data, w, h }, where };
  }

  // Compile a fragment source made with instanced(): the program plus its instance
  // table texture (texture unit 5) and the u_SP order.
  function instancedProgram(gl, head, inst, tail) {
    const built = program(gl, head + inst.glsl + tail);
    built.sp = inst.sp;
    if (inst.table) {
      const { data, w, h } = inst.table;
      built.instTex = gl.createTexture();
      gl.activeTexture(gl.TEXTURE5);
      gl.bindTexture(gl.TEXTURE_2D, built.instTex);
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA32F, w, h, 0, gl.RGBA, gl.FLOAT, data);
      for (const k of [gl.TEXTURE_MIN_FILTER, gl.TEXTURE_MAG_FILTER]) gl.texParameteri(gl.TEXTURE_2D, k, gl.NEAREST);
    }
    return built;
  }

  // Bind the instance table and fill u_SP; ``value(uniformName)`` gives each value.
  // Call with the program in use, after setting the scene's own uniforms.
  function bindInstances(gl, built, value) {
    if (!built.instTex || !built.uniforms.u_SP) return;
    gl.activeTexture(gl.TEXTURE5);
    gl.bindTexture(gl.TEXTURE_2D, built.instTex);
    gl.uniform1i(built.uniforms.u_inst, 5);
    const v = new Float32Array(Math.ceil(built.sp.length / 4) * 4);
    built.sp.forEach((name, i) => { v[i] = value(name); });
    gl.uniform4fv(built.uniforms.u_SP, v);
  }

  // Computed tables (Solvers.tables) as RGBA32F textures from unit 6 on; a texture is
  // refilled only when its table changes (the solver caches its result).
  const tableTex = new Map();
  function bindTables(gl, U, tabs) {
    let unit = 6;
    for (const [u, { data, width }] of Object.entries(tabs)) {
      if (!U[u]) continue;
      let e = tableTex.get(u);
      if (!e || e.gl !== gl) tableTex.set(u, (e = { gl, tex: gl.createTexture(), data: null }));
      gl.activeTexture(gl.TEXTURE0 + unit);
      gl.bindTexture(gl.TEXTURE_2D, e.tex);
      if (e.data !== data) {
        const rows = data.length / (4 * width);
        gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA32F, width, rows, 0, gl.RGBA, gl.FLOAT, new Float32Array(data));
        for (const k of [gl.TEXTURE_MIN_FILTER, gl.TEXTURE_MAG_FILTER]) gl.texParameteri(gl.TEXTURE_2D, k, gl.NEAREST);
        e.data = data;
      }
      gl.uniform1i(U[u], unit++);
    }
  }

  // The ray-marching program for a scene's GLSL (implicit.scene.Scene.glsl).
  function sceneProgram(gl, sceneGlsl) {
    return instancedProgram(gl, FRAG_HEAD, instanced(sceneGlsl), FRAG_TAIL);
  }

  // Template instance data (RGBA32F), or a 1x1 placeholder.
  function dataTexture(gl, scene) {
    const d = scene.data;
    const [w, h] = d ? d.size : [1, 1];
    const buf = d ? new Float32Array(Uint8Array.from(atob(d.f32_b64), (c) => c.charCodeAt(0)).buffer)
      : new Float32Array(4);
    const tex = gl.createTexture();
    gl.activeTexture(gl.TEXTURE2);
    gl.bindTexture(gl.TEXTURE_2D, tex);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA32F, w, h, 0, gl.RGBA, gl.FLOAT, buf);
    for (const [k, v] of [[gl.TEXTURE_MIN_FILTER, gl.NEAREST], [gl.TEXTURE_MAG_FILTER, gl.NEAREST]])
      gl.texParameteri(gl.TEXTURE_2D, k, v);
    return tex;
  }

  // The grid atlas (R8 3D texture) of a scene, inlined or fetched from scenes/.
  async function atlasTexture(gl, scene, base = "scenes/") {
    const a = scene.atlas;
    let gz;
    if (a.url) gz = await (await fetch(base + a.url, { cache: "no-store" })).arrayBuffer();
    else gz = Uint8Array.from(atob(a.gzip_b64), (c) => c.charCodeAt(0)).buffer;
    const stream = new Blob([gz]).stream().pipeThrough(new DecompressionStream("gzip"));
    const data = new Uint8Array(await new Response(stream).arrayBuffer());
    const [w, h, d] = a.size;
    const max = gl.getParameter(gl.MAX_3D_TEXTURE_SIZE);
    if (Math.max(w, h, d) > max) throw new Error(`atlas ${w}x${h}x${d} exceeds MAX_3D_TEXTURE_SIZE ${max}`);
    while (gl.getError() !== gl.NO_ERROR) { /* clear stale errors */ }
    const tex = gl.createTexture();
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_3D, tex);
    gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);
    gl.texImage3D(gl.TEXTURE_3D, 0, gl.R8, w, h, d, 0, gl.RED, gl.UNSIGNED_BYTE, data);
    for (const [k, v] of [[gl.TEXTURE_MIN_FILTER, gl.LINEAR], [gl.TEXTURE_MAG_FILTER, gl.LINEAR],
      [gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE], [gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE], [gl.TEXTURE_WRAP_R, gl.CLAMP_TO_EDGE]])
      gl.texParameteri(gl.TEXTURE_3D, k, v);
    const err = gl.getError();
    if (err !== gl.NO_ERROR) throw new Error(`3D texture upload failed (WebGL error 0x${err.toString(16)}, `
      + `${(w * h * d / 1e6).toFixed(0)} MB): the GPU may be out of memory`);
    return { tex, mb: w * h * d / 1e6 };
  }

  function info(gl) {
    const ext = gl.getExtension("WEBGL_debug_renderer_info");
    const renderer = ext ? gl.getParameter(ext.UNMASKED_RENDERER_WEBGL) : gl.getParameter(gl.RENDERER);
    const hp = gl.getShaderPrecisionFormat(gl.FRAGMENT_SHADER, gl.HIGH_FLOAT);
    return { renderer, floatBits: hp ? hp.precision : 0, max3D: gl.getParameter(gl.MAX_3D_TEXTURE_SIZE) };
  }

  // Bind a full-screen triangle to attribute 0.
  function fullscreenTriangle(gl) {
    const buf = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, buf);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 3, -1, -1, 3]), gl.STATIC_DRAW);
    gl.enableVertexAttribArray(0);
    gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0);
  }

  return { VERT, FRAG_HEAD, FRAG_TAIL, program, sceneProgram, instanced, instancedProgram, bindInstances, bindTables, dataTexture, atlasTexture, info, fullscreenTriangle };
})();
