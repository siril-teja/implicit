// Real-time SDF ray marcher. Scene GLSL comes from Python (implicit.scene.Scene.glsl)
// or, for block-graph scenes, is built here from the graph (graph.js) so the block
// canvas (editor.js) can change it. Params are uniforms: moving a slider sets a
// uniform and redraws, no recompile; only rewiring blocks rebuilds the shader.
// Shaders and GL helpers live in gpu.js (shared with the diagnostics page).
"use strict";

const $ = (s) => document.querySelector(s);
const canvas = $("#gl");
// "high-performance" asks dual-GPU laptops for the discrete GPU (a hint: on Windows
// the browser's GPU is set per app in Settings > System > Display > Graphics).
const gl = canvas.getContext("webgl2", { antialias: false, preserveDrawingBuffer: true, powerPreference: "high-performance" });

const state = {
  scene: null, prog: null, uniforms: {}, values: {}, vis: [], atlas: null, atlasKey: null,
  cam: { yaw: -0.9, pitch: 0.25, dist: 260, target: [0, 0, 0] },
  clip: { on: false, axis: 0, offset: 0 },
  dirty: true, interacting: 0, server: false,
  key: null, g: null,  // block-graph state: { graph, manifest, library, built, published, editor }
  hl: new Uint32Array(8),
};
const LS = (k) => `implicit.graph.${k}`;
function lsGet(k) { try { return localStorage.getItem(k); } catch { return null; } }
function lsSet(k, v) { try { if (v === null) localStorage.removeItem(k); else localStorage.setItem(k, v); } catch { /* private mode */ } }

function buildProgram(sceneGlsl) {
  return GPU.sceneProgram(gl, sceneGlsl);
}

function setStatus(msg, err = false) {
  const el = $("#status");
  el.textContent = msg;
  el.className = err ? "err" : "";
}

async function uploadAtlas(scene) {
  if (state.dataTex) gl.deleteTexture(state.dataTex);
  state.dataTex = GPU.dataTexture(gl, scene);
  const key = scene.atlas && (scene.atlas.url || scene.hash);
  if (!scene.atlas || state.atlasKey === key) return;
  if (state.atlas) gl.deleteTexture(state.atlas);
  const { tex, mb } = await GPU.atlasTexture(gl, scene);
  state.atlas = tex;
  state.atlasKey = key;
  state.atlasMB = mb;
}

function showDiag() {
  const g = GPU.info(gl);
  const atlas = state.scene && state.scene.atlas
    ? ` · atlas ${state.scene.atlas.size.join("×")} (${(state.atlasMB || 0).toFixed(0)} MB)` : "";
  $("#diag").textContent = `GPU: ${g.renderer} · float ${g.floatBits} bit · max 3D ${g.max3D}${atlas}`;
  if (g.floatBits < 23) setStatus("Warning: this GPU has no full-precision fragment floats; "
    + "the implicit bodies may render with speckle.", true);
}

async function loadScene(scene, key) {
  if (scene.graph && !window.SCENES) return loadGraphScene(scene, key);
  state.g = null;
  state.key = key;
  setGraphUI(null);
  try {
    setStatus("Loading anatomy…");
    await uploadAtlas(scene);
  } catch (e) {
    setStatus("Atlas error: " + e.message, true);
    console.error(e);
    return;
  }
  useScene(scene, true);
}

// Compile and show a scene; keeps slider values (by name) and visibility when ``keep``.
function useScene(scene, keep) {
  let built;
  try {
    built = buildProgram(scene.glsl);
  } catch (e) {
    setStatus("Shader error:\n" + e.message, true);
    console.error(e);
    return false;
  }
  const first = !state.scene;
  if (state.prog) gl.deleteProgram(state.prog);
  if (state.built && state.built.instTex) gl.deleteTexture(state.built.instTex);
  state.built = built;
  state.prog = built.prog;
  state.uniforms = built.uniforms;
  const oldVals = state.values;
  state.values = {};
  for (const p of scene.params) state.values[p.name] = keep && p.name in oldVals ? oldVals[p.name] : p.value;
  const oldGroups = state.scene ? state.scene.groups.map((g) => g.name) : [];
  state.vis = scene.groups.map((g, i) => {
    const j = oldGroups.indexOf(g.name);
    return keep && j >= 0 && state.vis[j] !== undefined ? state.vis[j] : g.visible !== false;
  });
  state.scene = scene;
  if (first) fitView();
  buildPanel();
  setStatus(`${scene.name} · shader compiled in ${built.ms.toFixed(0)} ms`);
  showDiag();
  state.dirty = true;
  return true;
}

// ---------------------------------------------------------------- block graphs
async function loadGraphScene(pyScene, key) {
  try {
    setStatus("Loading blocks and anatomy…");
    const [manifest, published, library] = await Promise.all([
      fetchJson("scenes/blocks.json"), fetchJson(`scenes/${key}.graph.json`), fetchJson(`scenes/${key}.library.json`)]);
    await uploadAtlas({ ...pyScene, atlas: library.atlas });
    let graph = published, note = "";
    const saved = lsGet(LS(key));
    if (saved) {
      try {
        const s = JSON.parse(saved);
        if (s.published === JSON.stringify(published)) { graph = s.graph; note = "Your edits were restored (Reset discards them)."; }
        else lsSet(LS(key), null);  // the published graph changed: drop stale edits
      } catch { lsSet(LS(key), null); }
    }
    state.key = key;
    state.g = { graph: JSON.parse(JSON.stringify(graph)), manifest, library, published, built: null, editor: null };
    setGraphUI(state.g);
    rebuild(false);
    state.g.editor.fit();
    if (note) $("#graphNote").textContent = note;
  } catch (e) {
    setStatus("Could not load the block graph: " + e.message, true);
    console.error(e);
  }
}

function rebuild(keep = true) {
  const g = state.g;
  g.built = Graph.build(g.graph, g.manifest, g.library);
  const ok = useScene(g.built.scene, keep);
  const nerr = Object.keys(g.built.errors).length;
  $("#graphNote").textContent = nerr ? `${nerr} block${nerr > 1 ? "s" : ""} with problems (red)` : "";
  g.editor.render();
  g.editor.sidePanel();
  updateHighlight();
  return ok;
}

let saveTimer = null;
function graphChanged(kind) {
  const g = state.g;
  if (!g) return;
  if (kind === "structure") rebuild(true);
  else if (kind === "value") { g.editor.render(); state.dirty = true; }
  else g.editor.render();
  clearTimeout(saveTimer);
  saveTimer = setTimeout(() => {
    const same = JSON.stringify(g.graph) === JSON.stringify(g.published);
    lsSet(LS(state.key), same ? null : JSON.stringify({ published: JSON.stringify(g.published), graph: g.graph }));
  }, 400);
}

function setGraphUI(g) {
  const has = !!g;
  for (const id of ["gAdd", "gUndo", "gRedo", "gFit", "gOpen", "gDownload", "gReset"]) $("#" + id).disabled = !has;
  $("#gSave").hidden = !(has && state.server);
  $("#noGraph").hidden = has;
  $("#editor").style.visibility = has ? "visible" : "hidden";
  $("#graphNote").textContent = "";
  $("#editor").replaceChildren();
  if (!has) { setTab("scene"); $("#selPane").replaceChildren(); return; }
  setTab("sel");
  g.editor = Editor.create({
    svg: $("#editor"), side: $("#selPane"), manifest: g.manifest, library: g.library,
    getGraph: () => g.graph, getBuilt: () => g.built,
    getValue: (name) => state.values[name],
    onChange: graphChanged,
    onValue: (name, v) => { state.values[name] = v; state.dirty = true; },
    onSelect: () => updateHighlight(),
    replaceGraph: (ng) => { g.graph = ng; graphChanged("structure"); },
    status: (msg) => { $("#graphNote").textContent = msg; },
  });
}

// Bodies to highlight in 3D: the clicked body, or every body the selected block feeds.
function updateHighlight() {
  state.hl.fill(0);
  const g = state.g;
  const set = (i) => { if (i >= 0 && i < 256) state.hl[i >> 5] |= 1 << (i & 31); };
  if (g && g.built && g.editor) {
    if (g.editor.focusBody !== null) set(g.editor.focusBody);
    else if (g.editor.selected) {
      g.built.scene.bodies.forEach((b, i) => { if (g.built.upstream(b.node).has(g.editor.selected)) set(i); });
    }
  } else if (state.pickedBody !== undefined && state.pickedBody !== null) set(state.pickedBody);
  state.dirty = true;
}

function setTab(name) {
  document.querySelectorAll(".tabs button").forEach((b) => b.classList.toggle("on", b.dataset.tab === name));
  $("#selPane").hidden = name !== "sel";
  $("#scenePane").hidden = name !== "scene";
}
document.querySelectorAll(".tabs button").forEach((b) => b.addEventListener("click", () => setTab(b.dataset.tab)));

$("#gAdd").addEventListener("click", () => state.g && state.g.editor.addAtCenter());
$("#gUndo").addEventListener("click", () => state.g && state.g.editor.undo());
$("#gRedo").addEventListener("click", () => state.g && state.g.editor.redo());
$("#gFit").addEventListener("click", () => state.g && state.g.editor.fit());
$("#gReset").addEventListener("click", () => {
  const g = state.g;
  if (!g || !confirm("Discard your edits to this graph?")) return;
  g.editor.snapshot();
  g.graph = JSON.parse(JSON.stringify(g.published));
  lsSet(LS(state.key), null);
  graphChanged("structure");
});
$("#gDownload").addEventListener("click", () => {
  const g = state.g;
  if (!g) return;
  const blob = new Blob([JSON.stringify(g.graph, null, 1) + "\n"], { type: "application/json" });
  const a = el("a", { href: URL.createObjectURL(blob), download: `${state.key}.json` });
  document.body.append(a); a.click(); a.remove();
  setTimeout(() => URL.revokeObjectURL(a.href), 1000);
});
$("#gOpen").addEventListener("click", () => $("#gFile").click());
$("#gFile").addEventListener("change", async (e) => {
  const f = e.target.files[0];
  e.target.value = "";
  if (!f || !state.g) return;
  try {
    const ng = JSON.parse(await f.text());
    if (ng.format !== "implicit.graph") throw new Error("not an implicit.graph file");
    state.g.editor.snapshot();
    state.g.graph = ng;
    graphChanged("structure");
    state.g.editor.fit();
  } catch (err) { setStatus("Could not open the graph: " + err.message, true); }
});
$("#gSave").addEventListener("click", async () => {
  const r = await fetch("/api/graph", { method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ scene: state.key, graph: state.g.graph }) });
  const j = await r.json();
  if (r.ok) { state.g.published = JSON.parse(JSON.stringify(state.g.graph)); lsSet(LS(state.key), null); }
  $("#graphNote").textContent = r.ok ? `Saved to ${j.file}` : "Save failed: " + j.error;
});

// ---------------------------------------------------------------- dock
function setDock(open) {
  $("#dock").hidden = !open;
  $("#dockToggle").textContent = open ? "Hide panel" : "Show panel";
  lsSet("implicit.dock", open ? "1" : "0");
  state.dirty = true;
}
$("#dockToggle").addEventListener("click", () => setDock($("#dock").hidden));
document.addEventListener("keydown", (e) => {
  const typing = /^(INPUT|SELECT|TEXTAREA)$/.test(document.activeElement && document.activeElement.tagName);
  if (e.key === "Tab" && !typing && !e.ctrlKey && !e.altKey && !e.metaKey) { e.preventDefault(); setDock($("#dock").hidden); }
});
(() => {
  const h = lsGet("implicit.dockH");
  if (h) document.documentElement.style.setProperty("--dock-h", h);
  if (lsGet("implicit.dock") === "0") setDock(false);
  const grip = $("#grip");
  let start = null;
  grip.addEventListener("pointerdown", (e) => {
    grip.setPointerCapture(e.pointerId);
    start = { y: e.clientY, h: $("#dock").getBoundingClientRect().height };
  });
  grip.addEventListener("pointermove", (e) => {
    if (!start) return;
    const hh = Math.max(120, Math.min(window.innerHeight - 140, start.h - (e.clientY - start.y)));
    document.documentElement.style.setProperty("--dock-h", hh + "px");
    state.dirty = true;
  });
  grip.addEventListener("pointerup", () => {
    if (!start) return;
    start = null;
    lsSet("implicit.dockH", getComputedStyle(document.documentElement).getPropertyValue("--dock-h"));
  });
})();

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
  $("#attribution").textContent = s.attribution || "";
  const params = $("#params");
  params.replaceChildren();
  // Graph scenes: block settings are edited on the blocks; here only the named Number sliders.
  const shown = s.graph ? s.params.filter((p) => p.number) : s.params;
  $("#paramsHead").hidden = !shown.length;
  for (const p of shown) {
    const out = el("output", { value: fmt(state.values[p.name], p.step) });
    const range = el("input", {
      type: "range", min: p.min, max: p.max, step: p.step, value: state.values[p.name],
      oninput: (e) => {
        state.values[p.name] = parseFloat(e.target.value);
        out.value = fmt(state.values[p.name], p.step);
        state.dirty = true;
        syncGraphValue(p);
      },
    });
    const reset = el("button", {
      className: "reset", title: "Reset to default", textContent: "↺",
      onclick: () => {
        state.values[p.name] = p.default;
        range.value = p.default;
        out.value = fmt(p.default, p.step);
        state.dirty = true;
        syncGraphValue(p);
      },
    });
    params.append(el("div", { className: "param" },
      el("label", { textContent: p.label + (p.unit ? ` (${p.unit})` : "") }),
      el("div", { className: "row" }, range, out, reset)));
  }
  const bodies = $("#bodies");
  bodies.replaceChildren();
  s.groups.forEach((b, i) => {
    const sw = el("span", { className: "swatch" });
    sw.style.background = `rgb(${b.color.map((c) => Math.round(c * 255)).join(",")})`;
    const cb = el("input", {
      type: "checkbox", checked: state.vis[i],
      onchange: (e) => { state.vis[i] = e.target.checked; state.dirty = true; },
    });
    bodies.append(el("label", { className: "body" }, cb, sw, b.name));
  });
  const [lo, hi] = viewBox();
  const ax = state.clip.axis;
  const off = $("#clipOffset");
  off.min = lo[ax]; off.max = hi[ax]; off.step = 0.1;
  off.value = state.clip.offset;
}

// A Number slider moved in the Scene tab: store the value in the graph too.
function syncGraphValue(p) {
  if (!state.g || !p.number) return;
  const n = state.g.graph.nodes[p.node];
  if (n) { n.in = n.in || {}; n.in.value = state.values[p.name]; graphChanged("value"); }
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
$("#quality").value = state.quality;
$("#quality").addEventListener("change", (e) => {
  state.quality = e.target.value;
  lsSet("implicit.quality", state.quality);
  state.dirty = true;
});
$("#copyParams").addEventListener("click", async () => {
  const txt = JSON.stringify(state.values, null, 1);
  try { await navigator.clipboard.writeText(txt); setStatus("Params copied to clipboard"); }
  catch { setStatus(txt); }
});
$("#exportStl").addEventListener("click", async () => {
  setStatus("Meshing on the Python side…");
  const r = await fetch("/api/export", {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ scene: $("#scenePick").value, params: state.values, visible: state.vis,
      graph: state.g ? state.g.graph : undefined }),
  });
  const j = await r.json();
  setStatus(r.ok ? "Wrote:\n" + j.files.join("\n") : "Export failed: " + j.error, !r.ok);
});

// ---------------------------------------------------------------- camera
// Where the bodies are now (graph scenes: at the current slider values), else the scene box.
function viewBox() {
  const s = state.scene;
  return (s._bodies && Graph.viewBounds(s, state.values)) || s.view_bounds || s.bounds;
}

function fitView() {
  const [lo, hi] = viewBox();
  state.cam.target = lo.map((v, i) => (v + hi[i]) / 2);
  const ext = hi.map((v, i) => v - lo[i]);
  const aspect = canvas.clientWidth / Math.max(1, canvas.clientHeight);
  const fitH = ext[2] / 2 / 0.45, fitW = Math.hypot(ext[0], ext[1]) / 2 / (0.45 * aspect);
  state.cam.dist = 1.1 * Math.max(fitH, fitW) + Math.hypot(ext[0], ext[1]) / 2;
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
  drag = { x: e.clientX, y: e.clientY, x0: e.clientX, y0: e.clientY, pan: e.button === 2 || e.shiftKey, button: e.button };
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
canvas.addEventListener("pointerup", (e) => {
  const click = drag && drag.button === 0 && Math.abs(e.clientX - drag.x0) + Math.abs(e.clientY - drag.y0) < 5;
  endDrag();
  if (click) pickAt(e);
});
canvas.addEventListener("pointercancel", endDrag);
canvas.addEventListener("contextmenu", (e) => e.preventDefault());
canvas.addEventListener("wheel", (e) => {
  e.preventDefault();
  state.cam.dist *= Math.exp(e.deltaY * 0.001);
  state.dirty = true;
}, { passive: false });

// ---------------------------------------------------------------- render
GPU.fullscreenTriangle(gl);

// Progressive rendering: every change draws a cheap low-res preview into an
// offscreen buffer (upscaled to the canvas); once input settles the full-res
// image is refined in horizontal tiles, one tile per animation frame, so no
// single draw call is long enough to trip the GPU watchdog.
const low = { fb: null, tex: null, w: 0, h: 0, scale: 0.35 };
const refine = { active: false, row: 0, tileH: 64, lastT: 0, start: 0, lastInput: 0 };

function ensureLow(w, h) {
  if (low.fb && low.w === w && low.h === h) return;
  if (low.fb) { gl.deleteFramebuffer(low.fb); gl.deleteTexture(low.tex); }
  low.tex = gl.createTexture();
  gl.activeTexture(gl.TEXTURE1);
  gl.bindTexture(gl.TEXTURE_2D, low.tex);
  gl.texStorage2D(gl.TEXTURE_2D, 1, gl.RGBA8, w, h);
  low.fb = gl.createFramebuffer();
  gl.bindFramebuffer(gl.FRAMEBUFFER, low.fb);
  gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, low.tex, 0);
  gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  low.w = w; low.h = h;
}

// Body under the mouse: render one pixel with the pick flag (id + 1 in red/green).
const pickFb = { fb: null, tex: null };
function pickAt(e) {
  if (!state.prog) return;
  if (!pickFb.fb) {
    pickFb.tex = gl.createTexture();
    gl.activeTexture(gl.TEXTURE1);
    gl.bindTexture(gl.TEXTURE_2D, pickFb.tex);
    gl.texStorage2D(gl.TEXTURE_2D, 1, gl.RGBA8, 1, 1);
    pickFb.fb = gl.createFramebuffer();
    gl.bindFramebuffer(gl.FRAMEBUFFER, pickFb.fb);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, pickFb.tex, 0);
  }
  const r = canvas.getBoundingClientRect();
  const sx = canvas.width / r.width, sy = canvas.height / r.height;
  gl.bindFramebuffer(gl.FRAMEBUFFER, pickFb.fb);
  gl.viewport(0, 0, 1, 1);
  setUniforms(canvas.width, canvas.height);
  gl.uniform1f(state.uniforms.u_pick, 1);
  gl.uniform2f(state.uniforms.u_pickPx, (e.clientX - r.left) * sx, canvas.height - (e.clientY - r.top) * sy);
  gl.drawArrays(gl.TRIANGLES, 0, 3);
  const px = new Uint8Array(4);
  gl.readPixels(0, 0, 1, 1, gl.RGBA, gl.UNSIGNED_BYTE, px);
  gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  const id = px[0] + 256 * px[1] - 1;
  selectBody(id >= 0 ? id : null);
}

function selectBody(i) {
  const b = i !== null && state.scene ? state.scene.bodies[i] : null;
  if (state.g) {
    state.g.editor.selectBody(b ? i : null);
    if (b) { setTab("sel"); if ($("#dock").hidden) setDock(true); }
  } else {
    state.pickedBody = b ? i : null;
    setStatus(b ? `Selected: ${b.name}` : "");
  }
  updateHighlight();
}

function setUniforms(w, h) {
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
  if (U.u_atlasSize && s.atlas) gl.uniform3fv(U.u_atlasSize, s.atlas.size);
  if (U.u_data && state.dataTex) {
    gl.activeTexture(gl.TEXTURE2);
    gl.bindTexture(gl.TEXTURE_2D, state.dataTex);
    gl.uniform1i(U.u_data, 2);
  }
  if (U.u_atlas && state.atlas) {
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_3D, state.atlas);
    gl.uniform1i(U.u_atlas, 0);
  }
  for (const p of s.params) if (U[p.uniform]) gl.uniform1f(U[p.uniform], state.values[p.name]);
  if (U.u_pick) gl.uniform1f(U.u_pick, 0);
  if (U.u_fast) gl.uniform1f(U.u_fast, 0);
  if (U.u_hl) gl.uniform4uiv(U.u_hl, state.hl);
  s.groups.forEach((_, i) => { if (U["u_grp" + i]) gl.uniform1f(U["u_grp" + i], state.vis[i] ? 1 : 0); });
  const byUniform = Object.fromEntries(s.params.map((p) => [p.uniform, state.values[p.name]]));
  GPU.bindInstances(gl, state.built, (u) => (u.startsWith("u_grp") ? (state.vis[+u.slice(5)] ? 1 : 0) : byUniform[u]));
}

// Quality: the pixel density of the sharp (refined) image on high-DPI screens.
const QUALITY = { high: 2, balanced: 1.5, fast: 1 };
state.quality = QUALITY[lsGet("implicit.quality")] ? lsGet("implicit.quality") : "balanced";

// Frame-rate readout (top right): frames per second while the view changes, then
// how long the sharp image took. Updated a few times per second.
const perf = { frames: [], shown: 0 };
function showFps(t, text) {
  if (text === undefined) {
    perf.frames.push(t);
    while (perf.frames.length && perf.frames[0] < t - 1000) perf.frames.shift();
    if (t - perf.shown < 250) return;
    const fps = perf.frames.length;
    text = `${fps} fps · preview ${Math.round(low.scale * 100)}%`;
    $("#fps").className = fps < 15 ? "slow" : fps < 30 ? "mid" : "";
  } else {
    perf.frames.length = 0;
    $("#fps").className = "";
  }
  perf.shown = t;
  $("#fps").textContent = text;
}

function frame(t) {
  requestAnimationFrame(frame);
  if (!state.prog) return;
  const dt = t - (refine.lastT || t);
  refine.lastT = t;
  const dpr = Math.min(window.devicePixelRatio || 1, QUALITY[state.quality]);
  const W = Math.max(1, Math.floor(canvas.clientWidth * dpr));
  const H = Math.max(1, Math.floor(canvas.clientHeight * dpr));
  if (canvas.width !== W || canvas.height !== H) { canvas.width = W; canvas.height = H; state.dirty = true; }

  if (state.dirty) {
    // Preview resolution adapts to keep moving the view at ~30 fps or better.
    if (state.previewed && dt > 45) low.scale = Math.max(0.12, low.scale * 0.85);
    else if (dt < 25) low.scale = Math.min(0.6, low.scale * 1.05);
    const lw = Math.max(1, Math.round(W * low.scale)), lh = Math.max(1, Math.round(H * low.scale));
    ensureLow(lw, lh);
    gl.bindFramebuffer(gl.FRAMEBUFFER, low.fb);
    gl.viewport(0, 0, lw, lh);
    setUniforms(lw, lh);
    gl.uniform1f(state.uniforms.u_fast, 1);  // cheaper shading while moving
    gl.drawArrays(gl.TRIANGLES, 0, 3);
    gl.bindFramebuffer(gl.READ_FRAMEBUFFER, low.fb);
    gl.bindFramebuffer(gl.DRAW_FRAMEBUFFER, null);
    gl.blitFramebuffer(0, 0, lw, lh, 0, 0, W, H, gl.COLOR_BUFFER_BIT, gl.LINEAR);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    state.dirty = false;
    state.previewed = true;
    if (refine.fence) { gl.deleteSync(refine.fence); refine.fence = null; }
    Object.assign(refine, { active: true, row: 0, lastInput: t, start: 0 });
    showFps(t);
    return;
  }
  if (refine.fence && gl.getSyncParameter(refine.fence, gl.SYNC_STATUS) === gl.SIGNALED) {
    // The GPU has finished the last tile (draw calls return before the GPU is done).
    showFps(t, `sharp in ${(t - refine.start).toFixed(0)} ms · ${W}×${H}`);
    gl.deleteSync(refine.fence);
    refine.fence = null;
  }
  if (!refine.active || state.interacting > 0 || t - refine.lastInput < 100) return;
  if (refine.row === 0) refine.start = t;
  else if (dt > 60) refine.tileH = Math.max(8, refine.tileH >> 1);
  else if (dt < 22) refine.tileH = Math.min(H, refine.tileH * 2);
  gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  gl.viewport(0, 0, W, H);
  setUniforms(W, H);
  gl.enable(gl.SCISSOR_TEST);
  gl.scissor(0, H - refine.row - refine.tileH, W, refine.tileH);  // top to bottom
  gl.drawArrays(gl.TRIANGLES, 0, 3);
  gl.disable(gl.SCISSOR_TEST);
  refine.row += refine.tileH;
  if (refine.row >= H) {
    refine.active = false;
    if (refine.fence) gl.deleteSync(refine.fence);
    refine.fence = gl.fenceSync(gl.SYNC_GPU_COMMANDS_COMPLETE, 0);
    gl.flush();
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
  showAbout();
  if (!gl) { setStatus("WebGL2 is not available in this browser.", true); return; }
  if (window.SCENES) {  // single-file build: scenes inlined
    initPicker(Object.keys(window.SCENES), (k) => loadScene(window.SCENES[k], k));
    return requestAnimationFrame(frame);
  }
  state.server = false;
  if (["localhost", "127.0.0.1", "[::1]"].includes(location.hostname)) {  // local dev server?
    try { state.server = (await fetch("/api/ping", { cache: "no-store" })).ok; } catch { /* static */ }
  }
  $("#exportStl").hidden = !state.server;
  const index = await fetchJson("scenes/index.json");
  const qs = new URLSearchParams(location.search).get("scene");
  initPicker(index, async (k) => loadScene(await fetchJson(`scenes/${k}.json`), k), qs);
  requestAnimationFrame(frame);
  if (state.server) {  // live reload when the Python script changes
    setInterval(async () => {
      try {
        const k = $("#scenePick").value;
        const sc = await fetchJson(`scenes/${k}.json`);
        if (state.lastPyHash && sc.hash !== state.lastPyHash) loadScene(sc, k);
        state.lastPyHash = sc.hash;
      } catch { /* server restarting */ }
    }, 1000);
  }
}

function showAbout() {
  const app = window.APP || {};
  const box = $("#about");
  box.replaceChildren();
  const line = (...parts) => box.append(el("div", {}, ...parts));
  const link = (text, href) => el("a", { href, textContent: text, target: "_blank", rel: "noopener" });
  const version = app.version ? `v${app.version}` : "development build";
  const commit = app.commit ? ` · ${app.commit.slice(0, 7)}` : "";
  line(`implicit ${version}${commit}`);
  if (app.repo) {
    const base = `https://github.com/${app.repo}`;
    const ref = app.version ? `v${app.version}` : (app.commit || "main");
    line(link("Release notes", app.version ? `${base}/releases/tag/v${app.version}` : `${base}/releases`),
      " · ", link("MIT licence", `${base}/blob/${ref}/LICENSE`),
      " · ", link("Notices & attribution", `${base}/blob/${ref}/NOTICE.md`),
      " · ", link("Source", `${base}/tree/${ref}`));
  }
  if (!window.SCENES) line(link("GPU check", "diagnostics.html"), " (does this computer render the models correctly?)");
}

function initPicker(keys, load, initial) {
  const pick = $("#scenePick");
  pick.replaceChildren(...keys.map((k) => el("option", { value: k, textContent: k })));
  if (initial && keys.includes(initial)) pick.value = initial;
  pick.hidden = keys.length < 2;
  pick.addEventListener("change", () => { state.scene = null; state.lastPyHash = null; load(pick.value); });
  load(pick.value);
}

main();
