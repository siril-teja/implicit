// GPU diagnostics: does this GPU evaluate the scene shaders like NumPy does?
// For every shader unit: distances at probe points (GPU vs NumPy reference from
// implicit.diagnostics) and one ray traced to the unit's deepest point. Uses the
// viewer's own shader code (gpu.js), so it tests exactly what the viewer runs.
// Result: window.DIAG = { done, pass, gpu, failures, checks } (read by CI).
"use strict";

const $ = (s) => document.querySelector(s);
const TOL = (d) => 0.15 + 0.002 * Math.abs(d);  // mm; trilinear filtering differs slightly per GPU
const lines = [];
const log = (s) => { lines.push(s); $("#report").value = lines.join("\n"); };
window.DIAG = { done: false, pass: false, failures: [], checks: 0 };

const f = (v) => v.toFixed(4);
const vec3 = (p) => `vec3(${f(p[0])}, ${f(p[1])}, ${f(p[2])})`;

function setUniforms(gl, U, scene, atlas) {
  if (U.u_atlas) { gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_3D, atlas); gl.uniform1i(U.u_atlas, 0); }
  if (U.u_data) gl.uniform1i(U.u_data, 2);
  if (U.u_atlasSize && scene.atlas) gl.uniform3fv(U.u_atlasSize, scene.atlas.size);
  for (const p of scene.params) if (U[p.uniform]) gl.uniform1f(U[p.uniform], p.value);
  scene.groups.forEach((_, i) => { if (U["u_grp" + i]) gl.uniform1f(U["u_grp" + i], 1); });
  if (U.u_bmin) { gl.uniform3fv(U.u_bmin, scene.bounds[0]); gl.uniform3fv(U.u_bmax, scene.bounds[1]); }
}

// One output pixel per probe; each fragment runs its probe statement (sets vec4 o).
function runProbes(gl, scene, atlas, stmts) {
  const n = stmts.length;
  let body = "";
  stmts.forEach((st, i) => { body += `  if (i == ${i}) { ${st} }\n`; });
  const frag = GPU.FRAG_HEAD + scene.glsl + PROBE_LIB
    + `\nvoid main() {\n  int i = int(gl_FragCoord.x);\n  vec4 o = vec4(-999.0);\n${body}  outColor = o;\n}`;
  const { prog, uniforms } = GPU.program(gl, frag);
  const tex = gl.createTexture();
  gl.activeTexture(gl.TEXTURE3);
  gl.bindTexture(gl.TEXTURE_2D, tex);
  gl.texStorage2D(gl.TEXTURE_2D, 1, gl.RGBA32F, n, 1);
  const fb = gl.createFramebuffer();
  gl.bindFramebuffer(gl.FRAMEBUFFER, fb);
  gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, tex, 0);
  gl.useProgram(prog);
  setUniforms(gl, uniforms, scene, atlas);
  gl.viewport(0, 0, n, 1);
  gl.drawArrays(gl.TRIANGLES, 0, 3);
  const px = new Float32Array(n * 4);
  gl.readPixels(0, 0, n, 1, gl.RGBA, gl.FLOAT, px);
  gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  gl.deleteFramebuffer(fb); gl.deleteTexture(tex); gl.deleteProgram(prog);
  return px;
}

// Ray march toward a target with one unit alone: (iterations, t, min distance, hit).
const PROBE_LIB = `
vec2 probeBox(vec3 ro, vec3 rd) {
  vec3 inv = 1.0 / rd;
  vec3 t0 = (u_bmin - ro) * inv, t1 = (u_bmax - ro) * inv;
  vec3 tmin = min(t0, t1), tmax = max(t0, t1);
  return vec2(max(max(tmin.x, tmin.y), tmin.z), min(min(tmax.x, tmax.y), tmax.z));
}
#define MARCH(UNIT, RO, TARGET) { vec3 rd = normalize(TARGET - RO); vec2 bh = probeBox(RO, rd); float t = max(bh.x, 0.0), mind = 1e9, it = 0.0, hit = 0.0; for (int k = 0; k < 300; k++) { float d = UNIT(RO + rd * t).x; mind = min(mind, d); it += 1.0; if (d < 0.002 * t + 0.01) { hit = 1.0; break; } t += 0.85 * d; if (t > bh.y) break; } o = vec4(it, t, mind, hit); }
`;

async function diagnose(gl, key) {
  const scene = await (await fetch(`scenes/${key}.json`, { cache: "no-store" })).json();
  const probes = await (await fetch(`scenes/${key}.probes.json`, { cache: "no-store" })).json();
  const atlas = scene.atlas ? (await GPU.atlasTexture(gl, scene)).tex : null;
  GPU.dataTexture(gl, scene);
  log(`\n== ${scene.name} (${key}): ${probes.units.length} units`);
  // 1) distances at probe points
  const exprs = [], meta = [];
  for (const u of probes.units) for (const p of u.points) {
    exprs.push(`o = vec4(unit${u.unit}(${vec3(p)}).x, 0.0, 0.0, 0.0);`);
    meta.push([u, p]);
  }
  const px = runProbes(gl, scene, atlas, exprs);
  let bad = 0;
  meta.forEach(([u, p], i) => {
    const gpu = px[i * 4], ref = p[3];
    window.DIAG.checks++;
    if (!(Math.abs(gpu - ref) <= TOL(ref))) {
      bad++;
      window.DIAG.failures.push(`${key} ${u.name}: distance at (${p.slice(0, 3).map((v) => v.toFixed(1))}) GPU ${gpu.toFixed(3)} vs NumPy ${ref.toFixed(3)}`);
    }
  });
  log(`distance probes: ${meta.length - bad}/${meta.length} match NumPy`);
  // 2) one ray per unit toward its deepest point, from a fixed eye outside the scene
  const [lo, hi] = scene.bounds;
  const c = lo.map((v, i) => (v + hi[i]) / 2);
  const ext = Math.max(...hi.map((v, i) => v - lo[i]));
  const eye = [c[0] + 1.6 * ext, c[1] + 0.6 * ext, c[2] + 0.3 * ext];
  const targets = probes.units.filter((u) => u.target);
  const rays = runProbes(gl, scene, atlas,
    targets.map((u) => `MARCH(unit${u.unit}, ${vec3(eye)}, ${vec3(u.target)})`));
  let missed = 0;
  targets.forEach((u, i) => {
    window.DIAG.checks++;
    const [it, t, mind, hit] = rays.slice(i * 4, i * 4 + 4);
    if (hit !== 1) {
      missed++;
      window.DIAG.failures.push(`${key} ${u.name}: ray missed (steps ${it}, travelled ${t.toFixed(0)} mm, closest ${mind.toFixed(2)} mm)`);
    }
  });
  log(`ray probes: ${targets.length - missed}/${targets.length} hit`);
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
