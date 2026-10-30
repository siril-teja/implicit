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

  function spinopelvic(d, v, light = false) {
    const out = motions(d, v), res = {};
    for (const [k, [R, t]] of Object.entries(out)) {
      res[k] = [R[0][0], R[1][0], R[2][0], R[0][1], R[1][1], R[2][1], R[0][2], R[1][2], R[2][2], ...t];
    }
    if (light) return res;
    const pairs = d.tps_points.map((q) => [q.point, apply(out[q.segment], q.point)]);
    res.tps = tpsTable(pairs);
    res.chain = chainTable(d, out);
    const G = gridPoints(d.grid), grid = new Array(G.length * 4), fwd = forward(d, out);
    G.forEach((g, i) => {
      const y = fwd(g), dd = [y[0] - g[0], y[1] - g[1], y[2] - g[2]];
      grid[4 * i] = dd[0]; grid[4 * i + 1] = dd[1]; grid[4 * i + 2] = dd[2]; grid[4 * i + 3] = Math.hypot(...dd);
    });
    res.grid = grid;
    [res.flow_cells, res.flow_segs] = flowTables(d, v);
    return res;
  }

  // The forward soft-tissue map (rest point -> moved) for segment motions ``out``.
  function forward(d, out) {
    if (d.field === "polyrigid") { const T = chainTable(d, out); return (x) => polyForward(T, x); }
    const F = tpsTable(d.tps_points.map((q) => [apply(out[q.segment], q.point), q.point]));  // fitted from the rest points
    return (x) => tpsEval(F, x);
  }

  // Flow lines (alignment.py flow_tables): seed paths from the subject's values to ``v``.
  const FLOW_STEPS = 10, FLOW_MARGIN = 6.0, FLOW_RMAX = 3.0, FLOW_ROW = 512;
  function flowTables(d, v) {
    const f = d.flow, S = f.seeds, start = {};
    for (const k of ["PI", "PT", "LL", "TK", "MT", "TL", "CB"]) start[k] = k in d.base ? d.base[k] : (v[k] ?? 0);
    start.PO = 0;
    const P = [S];
    for (let k = 1; k <= FLOW_STEPS; k++) {
      const s = k / FLOW_STEPS, vk = { AR: v.AR ?? 0 };
      for (const key of Object.keys(start)) vk[key] = start[key] + s * ((v[key] ?? 0) - start[key]);
      const fwd = forward(d, motions(d, vk));
      P.push(S.map(fwd));
    }
    const lo = f.lo, cs = f.cell, n = f.n, cells = Array.from({ length: n[0] * n[1] * n[2] }, () => []), segs = [];
    const grow = FLOW_MARGIN + FLOW_RMAX, dist = (a, b) => Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]);
    const cellOf = (x, i) => Math.min(Math.max(Math.floor((x - lo[i]) / cs), 0), n[i] - 1);
    S.forEach((seed, i) => {
      if (dist(P[FLOW_STEPS][i], seed) < 0.5) return;  // (nearly) still: no line
      for (let k = 0; k < FLOW_STEPS; k++) {
        const a = P[k][i], b = P[k + 1][i], e = segs.length;
        const c0 = [0, 1, 2].map((j) => cellOf(Math.min(a[j], b[j]) - grow, j));
        const c1 = [0, 1, 2].map((j) => cellOf(Math.max(a[j], b[j]) + grow, j));
        segs.push([a, dist(a, seed), b, dist(b, seed)]);
        for (let z = c0[2]; z <= c1[2]; z++) for (let y = c0[1]; y <= c1[1]; y++) {
          for (let x = c0[0]; x <= c1[0]; x++) cells[x + n[0] * (y + n[1] * z)].push(e);
        }
      }
    });
    const empty = emptyLayers(cells.map((c) => c.length > 0), n);
    const table = [], entries = [];
    cells.forEach((lst, i) => {
      table.push(entries.length / 8, lst.length, empty[i], 0);
      for (const e of lst) { const [a, va, b, vb] = segs[e]; entries.push(...a, va, ...b, vb); }
    });
    const pad = (-entries.length % (4 * FLOW_ROW) + 4 * FLOW_ROW) % (4 * FLOW_ROW);
    for (let i = 0; i < (pad || 4 * FLOW_ROW); i++) entries.push(0);
    return [table, entries];
  }

  // Empty cell layers around each cell (alignment.py _empty_layers), x fastest.
  const FLOW_SKIP = 4;
  function emptyLayers(full, n) {
    const at = (x, y, z) => x + n[0] * (y + n[1] * z);
    const dilate = (m) => {
      let out = m.slice();
      for (let axis = 0; axis < 3; axis++) {
        const a = out.slice(), st = [1, n[0], n[0] * n[1]][axis];
        for (let z = 0; z < n[2]; z++) for (let y = 0; y < n[1]; y++) for (let x = 0; x < n[0]; x++) {
          const c = [x, y, z][axis], i = at(x, y, z);
          if (c > 0 && out[i - st]) a[i] = true;
          if (c < n[axis] - 1 && out[i + st]) a[i] = true;
        }
        out = a;
      }
      return out;
    };
    let near = dilate(full);
    const out = full.map(() => 0);
    for (let k = 0; k < FLOW_SKIP; k++) {
      near.forEach((v, i) => { if (!v) out[i] += 1; });
      near = dilate(near);
    }
    return out;
  }

  function motions(d, v) {
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
    return out;
  }

  // ------------------------------------------------------------ polyrigid field (alignment.py)
  const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
  const dot3 = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
  function logRigid([R, t]) {
    const c = Math.max(-1, Math.min(1, (R[0][0] + R[1][1] + R[2][2] - 1) / 2)), th = Math.acos(c);
    if (th < 1e-9) return [[0, 0, 0], [...t]];
    const s = Math.sin(th), f = th / (2 * s);
    const w = [f * (R[2][1] - R[1][2]), f * (R[0][2] - R[2][0]), f * (R[1][0] - R[0][1])];
    const k = (1 - th * s / (2 * (1 - c))) / (th * th), wt = cross(w, t), wwt = cross(w, wt);
    return [w, [0, 1, 2].map((i) => t[i] - 0.5 * wt[i] + k * wwt[i])];
  }
  function screw(x, w, u) {
    const phi = Math.hypot(...w);
    if (phi < 1e-9) return [x[0] + u[0], x[1] + u[1], x[2] + u[2]];
    const k = w.map((v) => v / phi), c = Math.cos(phi), s = Math.sin(phi);
    const kx = cross(k, x), kd = dot3(k, x), wu = cross(w, u), wwu = cross(w, wu);
    const a = (1 - c) / (phi * phi), b = (phi - s) / (phi * phi * phi);
    return [0, 1, 2].map((i) => x[i] * c + kx[i] * s + k[i] * kd * (1 - c) + u[i] + a * wu[i] + b * wwu[i]);
  }
  function chainTable(d, out) {
    const segs = ["sacrum", ...d.levels], rows = [];
    segs.forEach((s, i) => {
      const [R, t] = out[s];
      let w = [0, 0, 0], u = [0, 0, 0];
      if (i + 1 < segs.length) {
        const Rt = [0, 1, 2].map((a) => [0, 1, 2].map((b) => R[b][a]));
        [w, u] = logRigid(compose([Rt, mv(Rt, t).map((v) => -v)], out[segs[i + 1]]));
      }
      rows.push([R[0][0], R[1][0], R[2][0], t[0]], [R[0][1], R[1][1], R[2][1], t[1]], [R[0][2], R[1][2], R[2][2], t[2]],
        [...w, 0], [...u, 0]);
      for (const side of ["inf", "sup"]) {
        const pl = d.planes[s][side];
        rows.push(pl ? [...pl[1], -dot3(pl[0], pl[1])] : [0, 0, 0, -1]);
      }
    });
    return rows.flat();
  }
  const chainRow = (T, r, k) => T.slice((r * 7 + k) * 4, (r * 7 + k) * 4 + 4);
  function polyRegion(T, x) {
    const n = T.length / 28;
    const side = (r, k) => { const pl = chainRow(T, r, k); return pl[0] * x[0] + pl[1] * x[1] + pl[2] * x[2] + pl[3]; };
    let cnt = 0;
    for (let r = 0; r < n; r++) { if (r > 0 && side(r, 5) >= 0) cnt++; if (side(r, 6) >= 0) cnt++; }
    const j = cnt - 1, r = j < 0 ? 0 : (j % 2 === 1 ? (j + 1) / 2 : j / 2);
    let lam = 0;
    if (j >= 0 && j % 2 === 0 && r < n - 1) {
      const a = side(r, 6), b = side(r + 1, 5);
      lam = Math.max(0, Math.min(1, a / (Math.abs(a - b) > 1e-12 ? a - b : 1)));
    }
    return [r, lam];
  }
  function polyForward(T, x) {
    const [r, lam] = polyRegion(T, x);
    const c = [0, 1, 2].map((k) => chainRow(T, r, k)), w = chainRow(T, r, 3), u = chainRow(T, r, 4);
    const y = screw(x, w.slice(0, 3).map((v) => lam * v), u.slice(0, 3).map((v) => lam * v));
    return [0, 1, 2].map((i) => c[0][i] * y[0] + c[1][i] * y[1] + c[2][i] * y[2] + c[i][3]);
  }
  function gridPoints(g) {
    const [nx, ny, nz] = g.n, out = [];
    for (let k = 0; k < nz; k++) for (let j = 0; j < ny; j++) for (let i = 0; i < nx; i++) {
      out.push([g.lo[0] + g.step * i, g.lo[1] + g.step * j, g.lo[2] + g.step * k]);
    }
    return out;
  }
  function tpsEval(T, p) {
    const q = [0, 1, 2].map((c) => T[4 * c] * p[0] + T[4 * c + 1] * p[1] + T[4 * c + 2] * p[2] + T[4 * c + 3]);
    for (let i = 1; i < T.length / 12; i++) {
      const o = 12 * i, r = Math.hypot(p[0] - T[o], p[1] - T[o + 1], p[2] - T[o + 2]);
      q[0] += T[o + 4] * r; q[1] += T[o + 5] * r; q[2] += T[o + 6] * r;
    }
    return q;
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
    return combos.map((vals) => KINDS[solver.kind](solver.data, vals, true));  // light: motions only
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
