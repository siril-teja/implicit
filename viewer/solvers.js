// CPU computations whose results reach the shader as "computed" uniforms (Python:
// implicit.expr.Solver / Computed). Same arithmetic as the Python side, which the GPU
// check confirms: it compares the shader at moved sliders with NumPy.
"use strict";

const Solvers = (() => {
  // ------------------------------------------------------------ spinopelvic chain
  // Port of implicit/anatomy/alignment.py solve(); see there for the model.
  const LORDOSIS = ["L1", "L2", "L3", "L4", "L5"];
  const KYPHOSIS = ["T4", "T5", "T6", "T7", "T8", "T9", "T10", "T11"];
  const rad = (d) => d * Math.PI / 180, deg = (r) => r * 180 / Math.PI;

  const rot = (c, a) => {
    const ca = Math.cos(a), sa = Math.sin(a);
    return [a, [c[0] - (ca * c[0] - sa * c[1]), c[1] - (sa * c[0] + ca * c[1])]];
  };
  const compose = ([a, [ty, tz]], [b, [uy, uz]]) => {
    const ca = Math.cos(a), sa = Math.sin(a);
    return [a + b, [ca * uy - sa * uz + ty, sa * uy + ca * uz + tz]];
  };
  const apply = ([a, [ty, tz]], p) => {
    const ca = Math.cos(a), sa = Math.sin(a);
    return [ca * p[0] - sa * p[1] + ty, sa * p[0] + ca * p[1] + tz];
  };
  const ptSs = (m, n, H) => [deg(Math.atan2(H[0] - m[0], m[1] - H[1])), deg(Math.atan2(n[0], n[1]))];
  function piAfter(d, beta) {
    const m = apply(rot(d.si, beta), d.s1_point), n = d.s1_normal;
    const cb = Math.cos(beta), sb = Math.sin(beta);
    const [pt, ss] = ptSs(m, [cb * n[0] - sb * n[1], sb * n[0] + cb * n[1]], d.hip);
    return [pt + ss, pt];
  }
  function weights(discs, uppers, sign) {
    const w = discs.map((d) => (uppers.includes(d.upper) ? Math.max(sign * d.angle, 1.0) : 0.0));
    const s = w.reduce((a, b) => a + b, 0);
    return w.map((x) => (s > 0 ? x / s : 0));
  }
  function spinopelvic(d, { PI, PT, LL, TK }) {
    const base = d.base, meas = d.measured;
    PI += meas.PI - base.PI;
    PT += meas.PT - base.PT;
    let lo = -rad(60), hi = rad(60);
    let fLo = piAfter(d, lo)[0] - PI;
    for (let i = 0; i < 60; i++) {
      const mid = 0.5 * (lo + hi), fMid = piAfter(d, mid)[0] - PI;
      if ((fMid > 0) === (fLo > 0)) { lo = mid; fLo = fMid; } else hi = mid;
    }
    let beta = 0.5 * (lo + hi);
    if (Math.abs(PI - meas.PI) < 1e-9) beta = 0;
    const alpha = rad(PT - piAfter(d, beta)[1]);
    const pelvis = rot(d.hip, alpha);
    let T = compose(pelvis, rot(d.si, beta));
    const out = { pelvis, sacrum: T };
    const dLL = rad(LL - ("LL" in base ? base.LL : LL)), dTK = rad(TK - ("TK" in base ? base.TK : TK));
    const wl = weights(d.discs, LORDOSIS, 1), wk = weights(d.discs, KYPHOSIS, -1);
    d.discs.forEach((disc, i) => {
      T = compose(T, rot(disc.center, dLL * wl[i] - dTK * wk[i]));
      out[disc.upper] = T;
    });
    const res = {};
    for (const [k, [a, t]] of Object.entries(out)) res[k] = [Math.cos(a), Math.sin(a), t[0], t[1]];
    res.curve = curve(d.curve.map((c) => [c.point[0], ...apply(out[c.segment], c.point.slice(1))]));
    return res;
  }

  const CURVE_STEPS = 4;
  function curve(P) {  // Catmull-Rom, as alignment.py _curve
    const n = P.length, out = [];
    for (let i = 0; i < n - 1; i++) {
      const p0 = P[Math.max(i - 1, 0)], p1 = P[i], p2 = P[i + 1], p3 = P[Math.min(i + 2, n - 1)];
      for (let k = 0; k < CURVE_STEPS; k++) {
        const t = k / CURVE_STEPS, t2 = t * t, t3 = t * t * t;
        for (let j = 0; j < 3; j++) {
          const a = p0[j], b = p1[j], c = p2[j], d = p3[j];
          out.push(0.5 * (2 * b + (c - a) * t + (2 * a - 5 * b + 4 * c - d) * t2 + (3 * b - a - 3 * c + d) * t3));
        }
      }
    }
    return out.concat(P[n - 1]);
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

  // Results over the input ranges (low, current, high of each; for bounds), as in Python.
  function samples(solver, range) {
    let combos = [{}];
    for (const [k, name] of Object.entries(solver.inputs)) {
      const [lo, cur, hi] = range(name);
      combos = combos.flatMap((c) => [lo, cur, hi].map((v) => ({ ...c, [k]: v })));
    }
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

  return { run, samples, apply: applyScene, KINDS };
})();

if (typeof module !== "undefined") module.exports = Solvers;
