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
  sceneInit();
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
  // one function ``grpK(p, i)``: per-instance numbers come from constant tables,
  // uniforms from a table SP filled once per pixel by sceneInit(), and sceneMap loops
  // over the instances. ``unitN`` stay as one-line wrappers (used by diagnostics).
  const FLOAT_RE = /\bu_\w+\b|(?:(?<=[(, ])-)?(?<![\w.])(?:(?:\d+\.\d*|\.\d+)(?:[eE][-+]?\d+)?|\d+[eE][-+]?\d+)(?![\w.])/g;  // a unary minus belongs to the literal
  function instanced(glsl) {
    const floatUniforms = new Set([...glsl.matchAll(/uniform float (u_\w+);/g)].map((m) => m[1]));
    const mapAt = glsl.indexOf("vec2 sceneMap(vec3 p) {");
    if (mapAt < 0) return glsl + "\nvoid sceneInit() {}\n";
    // Units, and their line in sceneMap (visibility uniform, bounds).
    const units = new Map();
    for (const m of glsl.matchAll(/vec2 unit(\d+)\(vec3 p\) \{[^\n]*\n/g)) {
      const end = glsl.indexOf("\n}", m.index) + 2;
      units.set(+m[1], { start: m.index, end, body: glsl.slice(m.index + m[0].length, end - 2) });
    }
    const lineRe = /^\s*if \(\((u_grp\d+) > 0\.5\)(?: && sdAABB\(p, (vec3\([^)]*\)), (vec3\([^)]*\))\) < r\.x)?\) \{ d = unit(\d+)\(p\); if \(d\.x < r\.x\) r = d; \}$/;
    const NO_BOX = ["vec3(-1e30)", "vec3(1e30)"];  // unbounded unit
    const mapLines = glsl.slice(mapAt).split("\n");
    for (const [i, line] of mapLines.entries()) {
      const m = line.match(lineRe);
      if (m && units.has(+m[4])) Object.assign(units.get(+m[4]), { line: i, vis: m[1], lo: m[2] || NO_BOX[0], hi: m[3] || NO_BOX[1] });
    }
    // Group by code skeleton (literals and float uniforms taken out).
    const groups = new Map();
    for (const [k, u] of units) {
      if (u.line === undefined || !floatUniforms.has(u.vis)) continue;
      u.slots = [];
      const skel = u.body.replace(FLOAT_RE, (s) => {
        if (s.startsWith("u_")) {
          if (!floatUniforms.has(s)) return s;
          u.slots.push({ u: s });
          return "\u0001U";
        }
        u.slots.push({ f: s });
        return "\u0001F";
      });
      if (!groups.has(skel)) groups.set(skel, []);
      groups.get(skel).push(k);
    }
    const sp = new Map();  // uniform -> index in SP
    const spIndex = (name) => { if (!sp.has(name)) sp.set(name, sp.size); return sp.get(name); };
    const out = [], drop = new Set(), loops = [], wrappers = new Map();
    let g = 0;
    for (const [skel, ks] of groups) {
      if (ks.length < 2) continue;
      const n = ks.length, inst = ks.map((k) => units.get(k));
      const nSlots = inst[0].slots.length;
      // Columns of per-instance values; identical columns are stored once.
      const fcols = new Map(), ucols = new Map(), expr = [];
      for (let s = 0; s < nSlots; s++) {
        const vals = inst.map((u) => u.slots[s]);
        if (vals[0].f !== undefined) {
          const fs = vals.map((v) => v.f);
          if (fs.every((v) => +v === +fs[0])) { expr.push(fs[0]); continue; }
          const key = fs.join(",");
          if (!fcols.has(key)) fcols.set(key, fs);
          expr.push({ f: [...fcols.keys()].indexOf(key) });
        } else {
          const us = vals.map((v) => v.u);
          if (us.every((v) => v === us[0])) { expr.push(us[0]); continue; }
          const key = us.join(",");
          if (!ucols.has(key)) ucols.set(key, us);
          expr.push({ u: [...ucols.keys()].indexOf(key) });
        }
      }
      const nf = fcols.size, nu = ucols.size;
      const fdata = [], udata = [];
      for (let i = 0; i < n; i++) {
        for (const col of fcols.values()) fdata.push(col[i]);
        for (const col of ucols.values()) udata.push(spIndex(col[i]));
      }
      while (fdata.length % 4) fdata.push("0.0");
      if (nf) {
        const vecs = [];
        for (let j = 0; j < fdata.length; j += 4) vecs.push(`vec4(${fdata.slice(j, j + 4).map((v) => (/[.eE]/.test(v) ? v : v + ".0")).join(", ")})`);
        out.push(`const vec4 G${g}F[${vecs.length}] = vec4[${vecs.length}](${vecs.join(", ")});`);
      }
      if (nu) out.push(`const int G${g}U[${udata.length}] = int[${udata.length}](${udata.join(", ")});`);
      let s = 0;
      const body = skel.replace(/\u0001[FU]/g, () => {
        const e = expr[s++];
        if (typeof e === "string") return e;
        if (e.f !== undefined) return `G${g}F[(i * ${nf} + ${e.f}) >> 2][(i * ${nf} + ${e.f}) & 3]`;
        return `SP[G${g}U[i * ${nu} + ${e.u}]]`;
      });
      out.push(`vec2 grp${g}(vec3 p, int i) {  // ${n} bodies\n${body}\n}`);
      const vis = inst.map((u) => spIndex(u.vis));
      out.push(`const int G${g}V[${n}] = int[${n}](${vis.join(", ")});`);
      out.push(`const vec3 G${g}LO[${n}] = vec3[${n}](${inst.map((u) => u.lo).join(", ")});`);
      out.push(`const vec3 G${g}HI[${n}] = vec3[${n}](${inst.map((u) => u.hi).join(", ")});`);
      loops.push(`    for (int i = 0; i < ${n}; i++) {\n        if (SP[G${g}V[i]] > 0.5 && sdAABB(p, G${g}LO[i], G${g}HI[i]) < r.x) { d = grp${g}(p, i); if (d.x < r.x) r = d; }\n    }`);
      ks.forEach((k, i) => { drop.add(units.get(k).line); wrappers.set(k, `vec2 unit${k}(vec3 p) { return grp${g}(p, ${i}); }`); });
      g++;
    }
    if (!g) return glsl + "\nvoid sceneInit() {}\n";
    // Reassemble: unit definitions (grouped ones replaced by wrappers placed after
    // the shared code), then sceneMap without the grouped lines plus the loops.
    let head = "", at = 0;
    for (const [k, u] of [...units].sort((a, b) => a[1].start - b[1].start)) {
      head += glsl.slice(at, u.start);
      if (!wrappers.has(k)) head += glsl.slice(u.start, u.end);
      at = u.end;
    }
    head += glsl.slice(at, mapAt);
    const init = `float SP[${sp.size}];\nvoid sceneInit() {\n${[...sp].map(([name, i]) => `    SP[${i}] = ${name};`).join("\n")}\n}`;
    const kept = mapLines.filter((_, i) => !drop.has(i));
    const ret = kept.lastIndexOf("    return r;");
    kept.splice(ret, 0, ...loops);
    return [head, init, ...out, ...wrappers.values(), kept.join("\n")].join("\n");
  }

  // The ray-marching program for a scene's GLSL (implicit.scene.Scene.glsl).
  function sceneProgram(gl, sceneGlsl) {
    return program(gl, FRAG_HEAD + instanced(sceneGlsl) + FRAG_TAIL);
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

  return { VERT, FRAG_HEAD, FRAG_TAIL, program, sceneProgram, instanced, dataTexture, atlasTexture, info, fullscreenTriangle };
})();
