// Real-time SDF ray marcher. Scene GLSL comes from Python (implicit.scene.Scene.glsl).
// Params are uniforms: moving a slider sets a uniform and redraws, no recompile.
"use strict";

const VERT = `#version 300 es
in vec2 a_pos;
void main() { gl_Position = vec4(a_pos, 0.0, 1.0); }`;

const FRAG_HEAD = `#version 300 es
precision highp float;
out vec4 outColor;
uniform vec2 u_res;
uniform vec3 u_eye, u_fwd, u_right, u_up;
uniform float u_fov;
uniform vec3 u_bmin, u_bmax;
uniform vec4 u_clip;     // xyz normal, w offset
uniform float u_clipOn;
uniform vec3 u_colors[16];
`;

const FRAG_TAIL = `
vec2 mapClip(vec3 p) {
  vec2 m = sceneMap(p);
  if (u_clipOn > 0.5) m.x = max(m.x, dot(p, u_clip.xyz) - u_clip.w);
  return m;
}
vec3 calcNormal(vec3 p) {
  const vec2 k = vec2(1.0, -1.0);
  float e = 0.02;
  return normalize(k.xyy * mapClip(p + k.xyy * e).x + k.yyx * mapClip(p + k.yyx * e).x +
                   k.yxy * mapClip(p + k.yxy * e).x + k.xxx * mapClip(p + k.xxx * e).x);
}
float calcAO(vec3 p, vec3 n) {
  float occ = 0.0, w = 1.0;
  for (int i = 1; i <= 5; i++) {
    float h = 0.6 * float(i);
    occ += w * (h - mapClip(p + n * h).x);
    w *= 0.7;
  }
  return clamp(1.0 - 0.12 * occ, 0.0, 1.0);
}
vec2 boxHit(vec3 ro, vec3 rd) {
  vec3 inv = 1.0 / rd;
  vec3 t0 = (u_bmin - ro) * inv, t1 = (u_bmax - ro) * inv;
  vec3 tmin = min(t0, t1), tmax = max(t0, t1);
  return vec2(max(max(tmin.x, tmin.y), tmin.z), min(min(tmax.x, tmax.y), tmax.z));
}
void main() {
  vec2 uv = (2.0 * gl_FragCoord.xy - u_res) / u_res.y;
  vec3 rd = normalize(u_fwd + uv.x * u_fov * u_right + uv.y * u_fov * u_up);
  vec3 ro = u_eye;
  vec3 bg = mix(vec3(0.13, 0.14, 0.17), vec3(0.24, 0.26, 0.31), 0.5 + 0.5 * uv.y);
  vec2 bh = boxHit(ro, rd);
  if (bh.x > bh.y || bh.y < 0.0) { outColor = vec4(pow(bg, vec3(0.4545)), 1.0); return; }
  float t = max(bh.x, 0.0);
  float id = -1.0;
  for (int i = 0; i < 300; i++) {
    vec2 m = mapClip(ro + rd * t);
    if (m.x < 0.002 * t + 0.01) { id = m.y; break; }
    t += 0.85 * m.x;
    if (t > bh.y) break;
  }
  vec3 col = bg;
  if (id >= 0.0) {
    vec3 p = ro + rd * t;
    vec3 n = calcNormal(p);
    vec3 base = u_colors[int(id)];
    bool cut = u_clipOn > 0.5 && abs(dot(p, u_clip.xyz) - u_clip.w) < 0.05;
    if (cut) base *= 0.75 + 0.25 * step(0.5, fract((p.x + p.y + p.z) * 0.35));
    vec3 l1 = normalize(-u_fwd + u_up * 0.6 + u_right * 0.4);
    float dif = max(dot(n, l1), 0.0);
    float hemi = 0.5 + 0.5 * n.z;
    float ao = calcAO(p, n);
    float spec = pow(max(dot(reflect(rd, n), l1), 0.0), 32.0);
    col = base * (0.85 * dif + 0.35 * hemi * ao) + 0.15 * spec;
  }
  outColor = vec4(pow(col, vec3(0.4545)), 1.0);
}`;

const $ = (s) => document.querySelector(s);
const canvas = $("#gl");
const gl = canvas.getContext("webgl2", { antialias: false });

const state = {
  scene: null, prog: null, uniforms: {}, values: {}, vis: [],
  cam: { yaw: -0.9, pitch: 0.25, dist: 260, target: [0, 0, 0] },
  clip: { on: false, axis: 0, offset: 0 },
  dirty: true, interacting: 0, server: false,
};

function compile(type, src) {
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

function buildProgram(sceneGlsl) {
  const t0 = performance.now();
  const p = gl.createProgram();
  gl.attachShader(p, compile(gl.VERTEX_SHADER, VERT));
  gl.attachShader(p, compile(gl.FRAGMENT_SHADER, FRAG_HEAD + sceneGlsl + FRAG_TAIL));
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

function setStatus(msg, err = false) {
  const el = $("#status");
  el.textContent = msg;
  el.className = err ? "err" : "";
}

function loadScene(scene) {
  let built;
  try {
    built = buildProgram(scene.glsl);
  } catch (e) {
    setStatus("Shader error:\n" + e.message, true);
    console.error(e);
    return;
  }
  const first = !state.scene;
  if (state.prog) gl.deleteProgram(state.prog);
  state.prog = built.prog;
  state.uniforms = built.uniforms;
  const oldVals = state.values;
  state.values = {};
  for (const p of scene.params) state.values[p.name] = p.name in oldVals ? oldVals[p.name] : p.value;
  if (state.vis.length !== scene.bodies.length) state.vis = scene.bodies.map(() => true);
  state.scene = scene;
  if (first) fitView();
  buildPanel();
  setStatus(`${scene.name} · shader compiled in ${built.ms.toFixed(0)} ms`);
  state.dirty = true;
}

// ---------------------------------------------------------------- panel
function el(tag, attrs = {}, ...kids) {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k.startsWith("on")) e.addEventListener(k.slice(2), v);
    else if (k in e) e[k] = v;
    else e.setAttribute(k, v);
  }
  for (const k of kids) e.append(k);
  return e;
}

function fmt(v, step) {
  const d = step >= 1 ? 0 : Math.min(3, Math.ceil(-Math.log10(step)));
  return Number(v).toFixed(d);
}

function buildPanel() {
  const s = state.scene;
  $("#title").textContent = s.name;
  const params = $("#params");
  params.replaceChildren();
  for (const p of s.params) {
    const out = el("output", { value: fmt(state.values[p.name], p.step) });
    const range = el("input", {
      type: "range", min: p.min, max: p.max, step: p.step, value: state.values[p.name],
      oninput: (e) => {
        state.values[p.name] = parseFloat(e.target.value);
        out.value = fmt(state.values[p.name], p.step);
        state.dirty = true;
      },
    });
    const reset = el("button", {
      className: "reset", title: "Reset to default", textContent: "↺",
      onclick: () => {
        state.values[p.name] = p.default;
        range.value = p.default;
        out.value = fmt(p.default, p.step);
        state.dirty = true;
      },
    });
    params.append(el("div", { className: "param" },
      el("label", { textContent: p.label + (p.unit ? ` (${p.unit})` : "") }),
      el("div", { className: "row" }, range, out, reset)));
  }
  const bodies = $("#bodies");
  bodies.replaceChildren();
  s.bodies.forEach((b, i) => {
    const sw = el("span", { className: "swatch" });
    sw.style.background = `rgb(${b.color.map((c) => Math.round(c * 255)).join(",")})`;
    const cb = el("input", {
      type: "checkbox", checked: state.vis[i],
      onchange: (e) => { state.vis[i] = e.target.checked; state.dirty = true; },
    });
    bodies.append(el("label", { className: "body" }, cb, sw, b.name));
  });
  const [lo, hi] = s.bounds;
  const ax = state.clip.axis;
  const off = $("#clipOffset");
  off.min = lo[ax]; off.max = hi[ax]; off.step = 0.1;
  off.value = state.clip.offset;
}

$("#clipOn").addEventListener("change", (e) => { state.clip.on = e.target.checked; state.dirty = true; });
$("#clipAxis").addEventListener("change", (e) => {
  state.clip.axis = +e.target.value;
  state.clip.offset = 0;
  buildPanel();
  state.dirty = true;
});
$("#clipFlip").addEventListener("change", () => { state.dirty = true; });
$("#clipOffset").addEventListener("input", (e) => { state.clip.offset = +e.target.value; state.dirty = true; });
$("#resetView").addEventListener("click", () => { fitView(); state.dirty = true; });
$("#copyParams").addEventListener("click", async () => {
  const txt = JSON.stringify(state.values, null, 1);
  try { await navigator.clipboard.writeText(txt); setStatus("Params copied to clipboard"); }
  catch { setStatus(txt); }
});
$("#exportStl").addEventListener("click", async () => {
  setStatus("Meshing on the Python side…");
  const r = await fetch("/api/export", {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ scene: $("#scenePick").value, params: state.values, visible: state.vis }),
  });
  const j = await r.json();
  setStatus(r.ok ? "Wrote:\n" + j.files.join("\n") : "Export failed: " + j.error, !r.ok);
});

// ---------------------------------------------------------------- camera
function fitView() {
  const [lo, hi] = state.scene.bounds;
  state.cam.target = lo.map((v, i) => (v + hi[i]) / 2);
  const diag = Math.hypot(...lo.map((v, i) => hi[i] - v));
  state.cam.dist = diag * 0.95;
  state.cam.yaw = 0.35;
  state.cam.pitch = 0.12;
}

function camBasis() {
  const { yaw, pitch, dist, target } = state.cam;
  const dir = [Math.cos(pitch) * Math.cos(yaw), Math.cos(pitch) * Math.sin(yaw), Math.sin(pitch)];
  const eye = target.map((t, i) => t + dir[i] * dist);
  const fwd = dir.map((d) => -d);
  let right = [fwd[1], -fwd[0], 0];
  const rl = Math.hypot(...right) || 1;
  right = right.map((v) => v / rl);
  const up = [right[1] * fwd[2] - right[2] * fwd[1], right[2] * fwd[0] - right[0] * fwd[2],
    right[0] * fwd[1] - right[1] * fwd[0]];
  return { eye, fwd, right, up };
}

let drag = null;
canvas.addEventListener("pointerdown", (e) => {
  canvas.setPointerCapture(e.pointerId);
  drag = { x: e.clientX, y: e.clientY, pan: e.button === 2 || e.shiftKey };
  state.interacting++;
});
canvas.addEventListener("pointermove", (e) => {
  if (!drag) return;
  const dx = e.clientX - drag.x, dy = e.clientY - drag.y;
  drag.x = e.clientX; drag.y = e.clientY;
  if (drag.pan) {
    const { right, up } = camBasis();
    const k = state.cam.dist * 0.0015;
    state.cam.target = state.cam.target.map((t, i) => t - right[i] * dx * k + up[i] * dy * k);
  } else {
    state.cam.yaw -= dx * 0.008;
    state.cam.pitch = Math.max(-1.55, Math.min(1.55, state.cam.pitch + dy * 0.008));
  }
  state.dirty = true;
});
const endDrag = () => { if (drag) { drag = null; state.interacting--; state.dirty = true; } };
canvas.addEventListener("pointerup", endDrag);
canvas.addEventListener("pointercancel", endDrag);
canvas.addEventListener("contextmenu", (e) => e.preventDefault());
canvas.addEventListener("wheel", (e) => {
  e.preventDefault();
  state.cam.dist *= Math.exp(e.deltaY * 0.001);
  state.dirty = true;
}, { passive: false });

// ---------------------------------------------------------------- render
const quad = gl.createBuffer();
gl.bindBuffer(gl.ARRAY_BUFFER, quad);
gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 3, -1, -1, 3]), gl.STATIC_DRAW);
gl.enableVertexAttribArray(0);
gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0);

let lastFull = 0;
function frame() {
  requestAnimationFrame(frame);
  if (!state.prog || !state.dirty) return;
  const busy = state.interacting > 0 || performance.now() - lastFull < 120;
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  const scale = busy ? 0.5 : 1.0;
  const w = Math.max(1, Math.floor(canvas.clientWidth * dpr * scale));
  const h = Math.max(1, Math.floor(canvas.clientHeight * dpr * scale));
  if (canvas.width !== w || canvas.height !== h) { canvas.width = w; canvas.height = h; }
  gl.viewport(0, 0, w, h);
  gl.useProgram(state.prog);
  const U = state.uniforms, s = state.scene;
  const c = camBasis();
  gl.uniform2f(U.u_res, w, h);
  gl.uniform3fv(U.u_eye, c.eye);
  gl.uniform3fv(U.u_fwd, c.fwd);
  gl.uniform3fv(U.u_right, c.right);
  gl.uniform3fv(U.u_up, c.up);
  gl.uniform1f(U.u_fov, 0.45);
  gl.uniform3fv(U.u_bmin, s.bounds[0]);
  gl.uniform3fv(U.u_bmax, s.bounds[1]);
  const n = [0, 0, 0];
  const sign = $("#clipFlip").checked ? -1 : 1;
  n[state.clip.axis] = sign;
  gl.uniform4f(U.u_clip, n[0], n[1], n[2], sign * state.clip.offset);
  gl.uniform1f(U.u_clipOn, state.clip.on ? 1 : 0);
  const cols = new Float32Array(48);
  s.bodies.forEach((b, i) => cols.set(b.color, i * 3));
  if (U.u_colors) gl.uniform3fv(U.u_colors, cols);
  for (const p of s.params) if (U[p.uniform]) gl.uniform1f(U[p.uniform], state.values[p.name]);
  s.bodies.forEach((_, i) => { if (U["u_vis" + i]) gl.uniform1f(U["u_vis" + i], state.vis[i] ? 1 : 0); });
  const t0 = performance.now();
  gl.drawArrays(gl.TRIANGLES, 0, 3);
  if (!busy) {
    state.dirty = false;
    gl.finish();
    $("#fps").textContent = `${w}×${h} · ${(performance.now() - t0).toFixed(0)} ms`;
  } else {
    lastFull = state.interacting > 0 ? performance.now() : lastFull;
  }
}
window.addEventListener("resize", () => { state.dirty = true; });

// ---------------------------------------------------------------- loading
async function fetchJson(url) {
  const r = await fetch(url, { cache: "no-store" });
  if (!r.ok) throw new Error(`${url}: ${r.status}`);
  return r.json();
}

async function main() {
  if (!gl) { setStatus("WebGL2 is not available in this browser.", true); return; }
  if (window.SCENES) {  // single-file build: scenes inlined
    initPicker(Object.keys(window.SCENES), (k) => loadScene(window.SCENES[k]));
    return requestAnimationFrame(frame);
  }
  try {
    const ping = await fetch("/api/ping", { cache: "no-store" });
    state.server = ping.ok;
  } catch { state.server = false; }
  $("#exportStl").hidden = !state.server;
  const index = await fetchJson("scenes/index.json");
  const qs = new URLSearchParams(location.search).get("scene");
  initPicker(index, async (k) => loadScene(await fetchJson(`scenes/${k}.json`)), qs);
  requestAnimationFrame(frame);
  if (state.server) {  // live reload when the Python script changes
    setInterval(async () => {
      try {
        const k = $("#scenePick").value;
        const sc = await fetchJson(`scenes/${k}.json`);
        if (state.scene && sc.hash !== state.scene.hash) loadScene(sc);
      } catch { /* server restarting */ }
    }, 1000);
  }
}

function initPicker(keys, load, initial) {
  const pick = $("#scenePick");
  pick.replaceChildren(...keys.map((k) => el("option", { value: k, textContent: k })));
  if (initial && keys.includes(initial)) pick.value = initial;
  pick.hidden = keys.length < 2;
  pick.addEventListener("change", () => { state.scene = null; load(pick.value); });
  load(pick.value);
}

main();
