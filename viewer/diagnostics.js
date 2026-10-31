// GPU diagnostics: does this GPU evaluate the scene shaders like NumPy does?
// For every shader unit: distances at probe points (GPU vs NumPy reference from
// implicit.diagnostics) and one ray traced to the unit's deepest point. Uses the
// viewer's own shader code (gpu.js), so it tests exactly what the viewer runs.
// Result: window.DIAG = { done, pass, gpu, failures, checks } (read by CI).
"use strict";

const $ = (s) => document.querySelector(s);
const TOL = (d) => 0.15 + 0.002 * Math.abs(d);  // mm; trilinear filtering differs slightly per GPU
const lines = [];
const log = (s) => { lines.push(s); $("#report").value = lines.join("\n"); console.log("diag: " + s); };
window.DIAG = { done: false, pass: false, failures: [], checks: 0 };


// Value of a scene uniform: a parameter (moved or default) or a group's visibility (on).
function uniformValue(scene, values, u) {
  if (u.startsWith("u_grp")) return 1;
  const p = scene.params.find((q) => q.uniform === u);
  return p.name in values ? values[p.name] : p.value;
}

function setUniforms(gl, U, scene, atlas, values = {}) {
  if (U.u_atlas) { gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_3D, atlas); gl.uniform1i(U.u_atlas, 0); }
  if (U.u_data) gl.uniform1i(U.u_data, 2);
  if (U.u_atlasSize && scene.atlas) gl.uniform3fv(U.u_atlasSize, scene.atlas.size);
  for (const p of scene.params) if (U[p.uniform] && !p.texture) gl.uniform1f(U[p.uniform], p.name in values ? values[p.name] : p.value);
  GPU.bindTables(gl, U, Solvers.tables(scene, values));
  scene.groups.forEach((_, i) => { if (U["u_grp" + i]) gl.uniform1f(U["u_grp" + i], 1); });
  if (U.u_bmin) { gl.uniform3fv(U.u_bmin, scene.bounds[0]); gl.uniform3fv(U.u_bmax, scene.bounds[1]); }
}

// One output pixel per probe. Probe i reads its point and shader unit from a data
// texture (RGBA32F: xyz, unit) and calls that unit through one dispatch function, so
// each unit's code appears once in the program however many probes there are (large
// scenes otherwise make programs too big for some drivers). mode 0: field value at the
// point; mode 1: ray from ``eye`` toward the point with that unit alone.
// Probes run in programs of at most UNITS_PER_PROGRAM own units: functions a program
// never calls are dropped before the driver compiles it, which keeps each program
// small enough for slow compilers (D3D) in scenes of many bodies.
const UNITS_PER_PROGRAM = 8;
function runProbes(gl, scene, atlas, rows, values, mode, eye = [0, 0, 0]) {
  const px = new Float32Array(rows.length * 4);
  const units = [...new Set(rows.map((r) => r[0]))];
  // Units sharing code (GPU.instanced) cost one function per group: they all go in
  // the first program; the others at most UNITS_PER_PROGRAM per program.
  const where = GPU.instanced(scene.glsl).where;
  const shared = units.filter((u) => u in where), own = units.filter((u) => !(u in where));
  const chunks = [];
  for (let k = 0; k < Math.max(own.length, 1); k += UNITS_PER_PROGRAM) chunks.push(own.slice(k, k + UNITS_PER_PROGRAM));
  chunks[0].push(...shared);
  for (const c of chunks) {
    if (!c.length) continue;
    const chunk = new Set(c);
    const idx = rows.map((r, i) => (chunk.has(r[0]) ? i : -1)).filter((i) => i >= 0);
    const out = runProbeProgram(gl, scene, atlas, idx.map((i) => rows[i]), values, mode, eye, [...chunk]);
    idx.forEach((i, j) => px.set(out.subarray(j * 4, j * 4 + 4), i * 4));
  }
  return px;
}

function runProbeProgram(gl, scene, atlas, rows, values, mode, eye, units) {
  const n = rows.length;
  // Each shared function (grpK) is called from one place, with the row of the unit.
  const inst = GPU.instanced(scene.glsl);
  const grouped = units.filter((k) => k in inst.where), single = units.filter((k) => !(k in inst.where));
  const gs = [...new Set(grouped.map((k) => inst.where[k][0]))];
  const dispatch = "vec2 unitAt(int u, vec3 p) {\n"
    + single.map((k) => `  if (u == ${k}) return unit${k}(p);`).join("\n") + "\n"
    + "  int g = -1, row = 0;\n"
    + grouped.map((k) => `  if (u == ${k}) { g = ${inst.where[k][0]}; row = ${inst.where[k][1]}; }`).join("\n") + "\n"
    + gs.map((g) => `  if (g == ${g}) return grp${g}(p, row);`).join("\n") + "\n"
    + "  return vec2(1e10, -1.0);\n}\n";
  // Compiled as the viewer compiles it (shared code for bodies built alike).
  const built = GPU.instancedProgram(gl, GPU.FRAG_HEAD, inst, "\n" + dispatch + PROBE_LIB);
  const { prog, uniforms } = built;
  const data = new Float32Array(n * 4);
  rows.forEach(([u, x, y, z], i) => data.set([x, y, z, u], i * 4));
  const dtex = gl.createTexture();
  gl.activeTexture(gl.TEXTURE4);
  gl.bindTexture(gl.TEXTURE_2D, dtex);
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA32F, n, 1, 0, gl.RGBA, gl.FLOAT, data);
  for (const [k, v] of [[gl.TEXTURE_MIN_FILTER, gl.NEAREST], [gl.TEXTURE_MAG_FILTER, gl.NEAREST]]) gl.texParameteri(gl.TEXTURE_2D, k, v);
  const tex = gl.createTexture();
  gl.activeTexture(gl.TEXTURE3);
  gl.bindTexture(gl.TEXTURE_2D, tex);
  gl.texStorage2D(gl.TEXTURE_2D, 1, gl.RGBA32F, n, 1);
  const fb = gl.createFramebuffer();
  gl.bindFramebuffer(gl.FRAMEBUFFER, fb);
  gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, tex, 0);
  gl.useProgram(prog);
  const all = Solvers.apply(scene, { ...values });  // computed values from the moved sliders, as in the viewer
  setUniforms(gl, uniforms, scene, atlas, all);
  GPU.bindInstances(gl, built, (u) => uniformValue(scene, all, u));
  gl.uniform1i(uniforms.u_probe, 4);
  gl.uniform1f(uniforms.u_mode, mode);
  if (uniforms.u_eye0) gl.uniform3fv(uniforms.u_eye0, eye);
  gl.viewport(0, 0, n, 1);
  gl.drawArrays(gl.TRIANGLES, 0, 3);
  const px = new Float32Array(n * 4);
  gl.readPixels(0, 0, n, 1, gl.RGBA, gl.FLOAT, px);
  gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  gl.deleteFramebuffer(fb); gl.deleteTexture(tex); gl.deleteTexture(dtex); gl.deleteProgram(prog);
  if (built.instTex) gl.deleteTexture(built.instTex);
  return px;
}

// Ray march toward a target with one unit alone: (iterations, t, min distance, hit).
const PROBE_LIB = `
uniform highp sampler2D u_probe;
uniform float u_mode;
uniform vec3 u_eye0;
vec2 probeBox(vec3 ro, vec3 rd) {
  vec3 inv = 1.0 / rd;
  vec3 t0 = (u_bmin - ro) * inv, t1 = (u_bmax - ro) * inv;
  vec3 tmin = min(t0, t1), tmax = max(t0, t1);
  return vec2(max(max(tmin.x, tmin.y), tmin.z), min(min(tmax.x, tmax.y), tmax.z));
}
void main() {
  vec4 a = texelFetch(u_probe, ivec2(int(gl_FragCoord.x), 0), 0);
  int u = int(a.w + 0.5);
  if (u_mode < 0.5) { outColor = vec4(unitAt(u, a.xyz).x, 0.0, 0.0, 0.0); return; }
  vec3 ro = u_eye0, rd = normalize(a.xyz - ro);
  vec2 bh = probeBox(ro, rd);
  float t = max(bh.x, 0.0), mind = 1e9, it = 0.0, hit = 0.0;
  for (int k = 0; k < 300; k++) {
    float d = unitAt(u, ro + rd * t).x;
    mind = min(mind, d); it += 1.0;
    if (d < 0.002 * t + 0.01) { hit = 1.0; break; }
    t += 0.85 * d;
    if (t > bh.y) break;
  }
  outColor = vec4(it, t, mind, hit);
}
`;

const getJson = async (url) => (await fetch(url, { cache: "no-store" })).json();

// Block-graph scenes are checked twice: the shader Python wrote, and the one the
// viewer builds in the browser from the graph (graph.js), as used when editing.
async function diagnose(gl, key) {
  const scene = await getJson(`scenes/${key}.json`);
  const probes = await getJson(`scenes/${key}.probes.json`);
  const up = scene.atlas ? await GPU.atlasTexture(gl, scene) : null, atlas = up && up.tex;
  GPU.dataTexture(gl, scene);
  await check(gl, key, scene, probes, atlas, "");
  if (scene.graph) {
    const [manifest, graph, library] = await Promise.all([
      getJson("scenes/blocks.json"), getJson(`scenes/${key}.graph.json`), getJson(`scenes/${key}.library.json`)]);
    const built = Graph.build(graph, manifest, library);
    const errs = Object.entries(built.errors);
    if (errs.length) throw new Error(`browser build: ${errs.map(([k, v]) => `${k}: ${v}`).join("; ")}`);
    await check(gl, key, built.scene, probes, atlas, ", built in the browser");
    if (up) checkClearance(key, built.scene, library, probes, up.data, scene.atlas.size);
  }
  gl.deleteTexture(atlas);
}

// Clearance of neighbouring bones: viewer/clearance.js against implicit.anatomy.clearance.
const CLEARANCE_TOL = 0.01;  // mm
function checkClearance(key, scene, library, probes, atlasData, size) {
  for (const [b, batch] of probes.batches.entries()) {
    if (!batch.clearance || !batch.clearance.length) continue;
    const values = Solvers.apply(scene, { ...batch.params });
    const got = Clearance.compute(atlasData, size, Graph.rigidBodies(scene, library, values));
    const byPair = Object.fromEntries(got.map(([a, c, d]) => [`${a}|${c}`, d]));
    let bad = 0;
    for (const [a, c, ref] of batch.clearance) {
      window.DIAG.checks++;
      const v = byPair[`${a}|${c}`];
      if (v === undefined || !(Math.abs(v - ref) <= CLEARANCE_TOL)) {
        bad++;
        window.DIAG.failures.push(`${key} (${b ? "sliders moved" : "slider defaults"}) clearance ${a}-${c}: browser ${v === undefined ? "missing" : v.toFixed(4)} vs NumPy ${ref.toFixed(4)} mm`);
      }
    }
    log(`clearance (${b ? "sliders moved" : "slider defaults"}): ${batch.clearance.length - bad}/${batch.clearance.length} pairs match NumPy`);
  }
  for (const [b, batch] of probes.batches.entries()) {
    if (!batch.discs || !batch.discs.length) continue;
    const values = Solvers.apply(scene, { ...batch.params });
    const bodies = Graph.rigidBodies(scene, library, values), thick = Discs.thickness(atlasData, size, bodies);
    const got = Discs.heights(bodies);
    const byPair = Object.fromEntries(got.map(([u, l, ...h]) => [`${u}|${l}`, [...h, thick[`${u}|${l}`]]]));
    let bad = 0;
    for (const [u, l, ...ref] of batch.discs) {
      window.DIAG.checks++;
      const h = byPair[`${u}|${l}`];
      if (!h || ref.some((x, i) => !(Math.abs(h[i] - x) <= CLEARANCE_TOL))) {
        bad++;
        window.DIAG.failures.push(`${key} (${b ? "sliders moved" : "slider defaults"}) disc ${u}-${l}: browser ${h ? h.map((x) => x.toFixed(3)).join("/") : "missing"} vs NumPy ${ref.map((x) => x.toFixed(3)).join("/")} mm`);
      }
    }
    log(`disc heights (${b ? "sliders moved" : "slider defaults"}): ${batch.discs.length - bad}/${batch.discs.length} discs match NumPy`);
  }
}

async function check(gl, key, scene, probes, atlas, how) {
  for (const [b, batch] of probes.batches.entries()) {
    const label = (b === 0 ? "slider defaults" : "sliders moved") + how;
    log(`\n== ${scene.name} (${key}), ${label}: ${batch.units.length} units`);
    // 1) distances at probe points
    const rows = [], meta = [];
    for (const u of batch.units) for (const p of u.points) {
      rows.push([u.unit, p[0], p[1], p[2]]);
      meta.push([u, p]);
    }
    const px = runProbes(gl, scene, atlas, rows, batch.params, 0);
    let bad = 0;
    meta.forEach(([u, p], i) => {
      const gpu = px[i * 4], ref = p[3];
      window.DIAG.checks++;
      if (!(Math.abs(gpu - ref) <= TOL(ref))) {
        bad++;
        window.DIAG.failures.push(`${key} (${label}) ${u.name}: distance at (${p.slice(0, 3).map((v) => v.toFixed(1))}) GPU ${gpu.toFixed(3)} vs NumPy ${ref.toFixed(3)}`);
      }
    });
    log(`distance probes: ${meta.length - bad}/${meta.length} match NumPy`);
    // 2) one ray per unit toward its deepest point, from a fixed eye outside the scene
    const [lo, hi] = scene.view_bounds || scene.bounds;
    const c = lo.map((v, i) => (v + hi[i]) / 2);
    const ext = Math.max(...hi.map((v, i) => v - lo[i]));
    const eye = [c[0] + 1.6 * ext, c[1] + 0.6 * ext, c[2] + 0.3 * ext];
    const targets = batch.units.filter((u) => u.target);
    const rays = runProbes(gl, scene, atlas, targets.map((u) => [u.unit, ...u.target]), batch.params, 1, eye);
    let missed = 0;
    targets.forEach((u, i) => {
      window.DIAG.checks++;
      const [it, t, mind, hit] = rays.slice(i * 4, i * 4 + 4);
      if (hit !== 1) {
        missed++;
        window.DIAG.failures.push(`${key} (${label}) ${u.name}: ray missed (steps ${it}, travelled ${t.toFixed(0)} mm, closest ${mind.toFixed(2)} mm)`);
      }
    });
    log(`ray probes: ${targets.length - missed}/${targets.length} hit`);
  }
}

async function main() {
  const canvas = document.createElement("canvas");
  canvas.width = canvas.height = 4;
  const gl = canvas.getContext("webgl2", { antialias: false, powerPreference: "high-performance" });
  if (!gl) { verdict(false, "WebGL2 is not available in this browser."); return; }
  const g = GPU.info(gl);
  window.DIAG.gpu = g.renderer;
  log(`GPU: ${g.renderer} · float ${g.floatBits} bit · max 3D ${g.max3D}`);
  log(`Browser: ${navigator.userAgent}`);
  if (!gl.getExtension("EXT_color_buffer_float")) { verdict(false, "EXT_color_buffer_float missing: cannot read results."); return; }
  GPU.fullscreenTriangle(gl);
  const keys = await (await fetch("scenes/index.json", { cache: "no-store" })).json();
  for (const key of keys) {
    try { await diagnose(gl, key); }
    catch (e) { window.DIAG.failures.push(`${key}: ${e.message}`); log(`${key}: ERROR ${e.message}`); }
  }
  const pass = window.DIAG.failures.length === 0;
  if (!pass) log("\nFailures:\n" + window.DIAG.failures.slice(0, 50).join("\n"));
  verdict(pass, pass ? `All ${window.DIAG.checks} checks match NumPy on this GPU.`
    : `${window.DIAG.failures.length} of ${window.DIAG.checks} checks failed on this GPU.`);
}

function verdict(pass, msg) {
  window.DIAG.pass = pass;
  window.DIAG.done = true;
  const v = $("#verdict");
  v.textContent = (pass ? "PASS · " : "FAIL · ") + msg;
  v.className = pass ? "pass" : "fail";
  log(`\n${pass ? "PASS" : "FAIL"}: ${msg}`);
}

main().catch((e) => { window.DIAG.failures.push(e.message); verdict(false, e.message); });
