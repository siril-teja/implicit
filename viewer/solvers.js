// CPU computations whose results reach the shader as "computed" uniforms (Python:
// implicit.expr.Solver / Computed). Same arithmetic as the Python side, which the GPU
// check confirms: it compares the shader at moved sliders with NumPy.
"use strict";

const Solvers = (() => {
  // ------------------------------------------------------------ spine alignment
  // Port of implicit/anatomy/alignment.py solve(); see there for the model.
  const LORDOSIS = ["L1", "L2", "L3", "L4", "L5"];
  const KYPHOSIS = ["T4", "T5", "T6", "T7", "T8", "T9", "T10", "T11"];
  const X = [1, 0, 0], Y = [0, 1, 0];
  const rad = (d) => d * Math.PI / 180, deg = (r) => r * 180 / Math.PI;

  const mv = (R, v) => [0, 1, 2].map((i) => R[i][0] * v[0] + R[i][1] * v[1] + R[i][2] * v[2]);
  function rot(c, axis, a) {  // turn by a (rad) about the axis through c (Rodrigues)
    const [x, y, z] = axis, ca = Math.cos(a), sa = Math.sin(a), k = 1 - ca;
    const R = [[ca + x * x * k, x * y * k - z * sa, x * z * k + y * sa],
      [y * x * k + z * sa, ca + y * y * k, y * z * k - x * sa],
      [z * x * k - y * sa, z * y * k + x * sa, ca + z * z * k]];
    const Rc = mv(R, c);
    return [R, [0, 1, 2].map((i) => c[i] - Rc[i])];
  }
  function compose([Ra, ta], [Rb, tb]) {  // A after B
    const R = [0, 1, 2].map((i) => [0, 1, 2].map((j) => Ra[i][0] * Rb[0][j] + Ra[i][1] * Rb[1][j] + Ra[i][2] * Rb[2][j]));
    const u = mv(Ra, tb);
    return [R, [0, 1, 2].map((i) => u[i] + ta[i])];
  }
  const apply = (T, p) => { const u = mv(T[0], p); return [0, 1, 2].map((i) => u[i] + T[1][i]); };
  const ptSs = (m, n, H) => [deg(Math.atan2(H[0] - m[0], m[1] - H[1])), deg(Math.atan2(n[0], n[1]))];
  function piAfter(d, beta) {
    const T = rot(d.si, X, beta), m = apply(T, d.s1_point), n = mv(T[0], d.s1_normal);
    const [pt, ss] = ptSs([m[1], m[2]], [n[1], n[2]], [d.hip[1], d.hip[2]]);
    return [pt + ss, pt];
  }
  function bisect(f, lo, hi, target) {
    let fLo = f(lo) - target;
    for (let i = 0; i < 60; i++) {
      const mid = 0.5 * (lo + hi), fMid = f(mid) - target;
      if ((fMid > 0) === (fLo > 0)) { lo = mid; fLo = fMid; } else hi = mid;
    }
    return 0.5 * (lo + hi);
  }
  function weights(discs, uppers, sign) {
    const w = discs.map((d) => (uppers.includes(d.upper) ? Math.max(sign * d.angle, 1.0) : 0.0));
    const s = w.reduce((a, b) => a + b, 0);
    return w.map((x) => (s > 0 ? x / s : 0));
  }
  function curvePlan(levels, curves) {
    const idx = Object.fromEntries(levels.map((lv, i) => [lv, i])), plans = {};
    for (const [k, [u, a, lo]] of Object.entries(curves)) {
      if (![u, a, lo].every((v) => v in idx)) throw new Error(`${k} curve: ${u}, ${a}, ${lo} must all be vertebrae of this spine`);
      const iu = idx[u], ia = idx[a], il = idx[lo];
      if (!(iu > ia && ia > il)) throw new Error(`${k} curve: need upper end above apex above lower end (${u}, ${a}, ${lo})`);
      const low = [], up = [];
      for (let j = il + 1; j <= ia; j++) low.push(j);
      for (let j = ia + 1; j <= iu; j++) up.push(j);
      const share = levels.map(() => 0), shape = levels.map(() => 0);
      for (const [half, other] of [[low, up], [up, low]]) for (const j of half) share[j] = (other.length ? 0.5 : 1.0) / half.length;
      for (let i = il; i <= iu; i++) shape[i] = 1 - Math.abs(i - ia) / (i <= ia ? ia - il : iu - ia);
      plans[k] = [share, shape, il];
    }
    return plans;
  }

  function spinopelvic(d, v) {
    const base = d.base, meas = d.measured;
    const rel = (k, x) => (k in base ? x + meas[k] - base[k] : x);
    const PI = rel("PI", v.PI), PT = rel("PT", v.PT), CB = rel("CB", v.CB ?? 0);
    const delta = (k, x) => rad(x - (k in base ? base[k] : x));
    const PO = v.PO ?? 0, AR = v.AR ?? 0, levels = d.levels, discs = d.discs;
    const beta = Math.abs(PI - meas.PI) < 1e-9 ? 0 : bisect((b) => piAfter(d, b)[0], -rad(60), rad(60), PI);
    const alpha = rad(PT - piAfter(d, beta)[1]);
    const legs = rot(d.hip, Y, -rad(PO)), pelvis = compose(legs, rot(d.hip, X, alpha));
    const sacrum = compose(pelvis, rot(d.si, X, beta));
    const dLL = delta("LL", v.LL), dTK = delta("TK", v.TK);
    const wl = weights(discs, LORDOSIS, 1), wk = weights(discs, KYPHOSIS, -1);
    const sag = wl.map((a, i) => dLL * a - dTK * wk[i]);
    const plans = curvePlan(levels, d.curves), cor = discs.map(() => 0), axial = levels.map(() => 0);
    const targets = { MT: v.MT ?? 0, TL: v.TL ?? 0 };
    for (const [k, [share, shape]] of Object.entries(plans)) {
      const dk = delta(k, targets[k]);
      for (let j = 0; j < discs.length; j++) { cor[j] -= dk * share[j]; axial[j] -= AR * dk * shape[j]; }
    }
    const lows = Object.values(plans).map((p) => p[2]);
    const lowest = lows.length ? Math.min(...lows) : -1;
    const frac = discs.map((_, j) => j).filter((j) => j <= lowest);
    const build = (bal) => {
      let T = sacrum;
      const out = {};
      discs.forEach((disc, j) => {
        const c = disc.center, turn = cor[j] + (frac.includes(j) ? bal / frac.length : 0);
        T = compose(T, compose(rot(c, X, sag[j]), rot(c, Y, turn)));
        out[disc.upper] = T;
      });
      return out;
    };
    const top = levels[levels.length - 1];
    const offset = (bal) => apply(build(bal)[top], d.frames[top].center)[0] - apply(sacrum, d.s1_point)[0];
    const moved = cor.some((x) => Math.abs(x) > 0) || Math.abs(PO) > 0 || Math.abs(alpha) > 0 || Math.abs(beta) > 0
      || sag.some((x) => x !== 0);
    let bal = 0;
    if (frac.length && (moved || Math.abs(CB - meas.CB) > 1e-9)) bal = bisect(offset, -rad(30), rad(30), CB);
    const out = build(bal);
    levels.forEach((lv, i) => { const f = d.frames[lv]; out[lv] = compose(out[lv], rot(f.center, f.axes[2], axial[i])); });
    Object.assign(out, { legs, pelvis, sacrum });
    const res = {};
    for (const [k, [R, t]] of Object.entries(out)) {
      res[k] = [R[0][0], R[1][0], R[2][0], R[0][1], R[1][1], R[2][1], R[0][2], R[1][2], R[2][2], ...t];
    }
    res.tps = tpsTable(d.tps_points.map((q) => [q.point, apply(out[q.segment], q.point)]));
    return res;
  }

  // Inverse TPS (alignment.py tps_table): kernel |r| plus affine, from moved to original.
  function tpsTable(pairs) {
    const n = pairs.length, N = n + 4;
    const Yp = pairs.map((p) => p[1]), Xp = pairs.map((p) => p[0]);
    const M = Array.from({ length: N }, () => new Float64Array(N + 3));
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) M[i][j] = Math.hypot(Yp[i][0] - Yp[j][0], Yp[i][1] - Yp[j][1], Yp[i][2] - Yp[j][2]);
      M[i][n] = M[n][i] = 1;
      for (let k = 0; k < 3; k++) { M[i][n + 1 + k] = M[n + 1 + k][i] = Yp[i][k]; M[i][N + k] = Xp[i][k]; }
    }
    for (let c = 0; c < N; c++) {  // Gaussian elimination, partial pivoting
      let piv = c;
      for (let r = c + 1; r < N; r++) if (Math.abs(M[r][c]) > Math.abs(M[piv][c])) piv = r;
      [M[c], M[piv]] = [M[piv], M[c]];
      for (let r = 0; r < N; r++) {
        if (r === c || M[r][c] === 0) continue;
        const f = M[r][c] / M[c][c];
        for (let k = c; k < N + 3; k++) M[r][k] -= f * M[c][k];
      }
    }
    const sol = (i, k) => M[i][N + k] / M[i][i];
    const rows = [0, 1, 2].map((c) => [sol(n + 1, c), sol(n + 2, c), sol(n + 3, c), sol(n, c)]);
    for (let i = 0; i < n; i++) rows.push([...Yp[i], 0], [sol(i, 0), sol(i, 1), sol(i, 2), 0], [0, 0, 0, 0]);
    return rows.flat();
  }

  const KINDS = { spinopelvic };

  // ------------------------------------------------------------ plumbing
  // A solver: { kind, data, inputs: { input: param name } }; ``value(name)`` gives a
  // param's current value.
  const cache = new WeakMap();
  function run(solver, value) {
    const vals = {};
    for (const [k, name] of Object.entries(solver.inputs)) vals[k] = value(name);
    const key = JSON.stringify(vals);
    let c = cache.get(solver);
    if (!c || c.key !== key) cache.set(solver, (c = { key, res: KINDS[solver.kind](solver.data, vals) }));
    return c.res;
  }

  // Results at the current values, each input alone at its low and high end, and all
  // low / all high (for bounds), as expr.Solver.samples.
  function samples(solver, range) {
    const cur = {}, lo = {}, hi = {};
    for (const [k, name] of Object.entries(solver.inputs)) [lo[k], cur[k], hi[k]] = range(name);
    const combos = [cur, lo, hi];
    for (const k of Object.keys(solver.inputs)) combos.push({ ...cur, [k]: lo[k] }, { ...cur, [k]: hi[k] });
    return combos.map((vals) => KINDS[solver.kind](solver.data, vals));
  }

  // Fill in the computed params of a scene (scene.params entries with ``computed``)
  // from the other values; returns ``values`` (changed in place).
  function applyScene(scene, values) {
    if (!scene || !scene.solvers) return values;
    const byName = Object.fromEntries(scene.params.map((p) => [p.name, p]));
    const value = (n) => (n in values ? values[n] : byName[n] ? byName[n].value : NaN);
    const results = {};
    for (const [id, s] of Object.entries(scene.solvers)) results[id] = run(s, value);
    for (const p of scene.params) {
      if (p.computed) values[p.name] = results[p.computed.solver][p.computed.key][p.computed.index];
    }
    return values;
  }

  // Tables (computed textures) of a scene at the given values: { uniform: { data, width } }.
  function tables(scene, values) {
    const out = {};
    if (!scene || !scene.solvers) return out;
    const byName = Object.fromEntries(scene.params.map((p) => [p.name, p]));
    const value = (n) => (n in values ? values[n] : byName[n] ? byName[n].value : NaN);
    for (const p of scene.params) {
      if (p.texture) out[p.uniform] = { data: run(scene.solvers[p.texture.solver], value)[p.texture.key], width: p.texture.width };
    }
    return out;
  }

  return { run, samples, apply: applyScene, tables, KINDS };
})();

if (typeof module !== "undefined") module.exports = Solvers;
