// Disc space heights between neighbouring vertebrae (implicit/anatomy/discs.py, same
// computation on the same corner points): anterior, middle, posterior, left and right
// heights along the disc axis, and the posture rules relative to the rest heights.
"use strict";

const Discs = (() => {
  const VERTEBRAE = [..."1234567"].map((i) => `C${i}`)
    .concat(Array.from({ length: 12 }, (_, i) => `T${i + 1}`), Array.from({ length: 5 }, (_, i) => `L${i + 1}`));
  const KEYS = ["anterior", "centre", "posterior", "left", "right"];
  const MEAN_CHANGE = 0.10, LOCAL_STRAIN = 0.40, MIN_REST = 0.5;
  // (A, M, P) population heights per disc: [mean, SD] mm (sources in discs.py).
  const NORMS = {
    "T1-T2": [[4.5, 0.77], [5.7, 0.66], [4.3, 0.62]], "T2-T3": [[3.4, 0.97], [5.8, 0.79], [3.5, 0.99]],
    "T3-T4": [[3.3, 0.30], [4.4, 0.80], [3.2, 0.49]], "T4-T5": [[3.0, 0.74], [5.0, 0.61], [3.3, 0.45]],
    "T5-T6": [[3.5, 0.37], [4.7, 0.91], [3.6, 0.47]], "T6-T7": [[4.1, 0.35], [5.5, 0.45], [4.1, 0.65]],
    "T7-T8": [[4.2, 0.97], [5.3, 0.58], [3.6, 0.90]], "T8-T9": [[5.6, 1.17], [6.1, 0.89], [5.0, 0.91]],
    "T9-T10": [[5.4, 1.74], [5.5, 0.90], [4.2, 1.10]], "T10-T11": [[7.2, 1.21], [6.7, 1.18], [5.8, 1.00]],
    "T11-T12": [[6.0, 1.14], [6.5, 1.15], [4.8, 1.26]], "T12-L1": [[6.0, 1.14], [6.5, 1.15], [4.8, 1.26]],
    "L1-L2": [[4.89, 1.53], [6.36, 1.24], [2.98, 1.10]], "L2-L3": [[5.83, 1.71], [7.61, 1.48], [3.82, 1.21]],
    "L3-L4": [[7.12, 1.76], [8.53, 1.48], [4.44, 1.38]], "L4-L5": [[8.69, 2.02], [8.99, 1.80], [4.73, 1.49]],
    "L5-S1": [[10.38, 2.64], [8.07, 1.86], [3.77, 1.36]],
  };
  const IDENTITY = [[[1, 0, 0], [0, 1, 0], [0, 0, 1]], [0, 0, 0]];

  function pairs(names) {
    const has = new Set(names), levels = VERTEBRAE.filter((lv) => has.has(lv)), out = [];
    for (let i = 0; i + 1 < levels.length; i++) out.push([levels[i], levels[i + 1]]);
    if (levels.length && has.has("sacrum")) out.push([levels[levels.length - 1], "sacrum"]);
    return out;
  }
  const label = (u, l) => `${u}-${l === "sacrum" ? "S1" : l}`;
  const mv = (R, v) => [0, 1, 2].map((r) => R[r][0] * v[0] + R[r][1] * v[1] + R[r][2] * v[2]);
  const move = ([R, t], p) => mv(R, p).map((x, i) => x + t[i]);

  // [[upper, lower, A, M, P, L, R]]; bodies: { name: { corners, motion: [R rows, t] } }.
  function heights(bodies) {
    const out = [];
    for (const [up, lo] of pairs(Object.keys(bodies).filter((k) => bodies[k].corners))) {
      const cu = bodies[up].corners, cl = bodies[lo].corners;
      if (!cu.inf || !cl.sup) continue;
      const Mu = bodies[up].motion, Ml = bodies[lo].motion;
      const a = mv(Ml[0], cl.sup.normal), b = mv(Mu[0], cu.inf.normal);
      let w = a.map((x, i) => x - b[i]);
      const n = Math.hypot(...w);
      w = w.map((x) => x / n);
      out.push([up, lo, ...KEYS.map((k) => {
        const p = move(Mu, cu.inf[k]), q = move(Ml, cl.sup[k]);
        return (p[0] - q[0]) * w[0] + (p[1] - q[1]) * w[1] + (p[2] - q[2]) * w[2];
      })]);
    }
    return out;
  }
  const restHeights = (bodies) => heights(Object.fromEntries(Object.entries(bodies)
    .map(([k, b]) => [k, { corners: b.corners, motion: IDENTITY }])));

  // { "upper|lower": [reasons] } where the posture deforms a disc beyond the rules.
  function postureFlags(now, rest) {
    const r = Object.fromEntries(rest.map(([u, l, ...h]) => [`${u}|${l}`, h])), out = {};
    for (const [u, l, ...h] of now) {
      const h0 = r[`${u}|${l}`], why = [], pc = (x, x0) => `${(100 * (x / x0 - 1)) >= 0 ? "+" : ""}${(100 * (x / x0 - 1)).toFixed(0)} %`;
      if (h0[1] > MIN_REST && Math.abs(h[1] - h0[1]) > MEAN_CHANGE * h0[1]) why.push(`middle ${pc(h[1], h0[1])}`);
      for (const [name, i] of [["anterior", 0], ["posterior", 2], ["left", 3], ["right", 4]]) {
        if (h0[i] > MIN_REST && Math.abs(h[i] - h0[i]) > LOCAL_STRAIN * h0[i]) why.push(`${name} ${pc(h[i], h0[i])}`);
      }
      if (why.length) out[`${u}|${l}`] = why;
    }
    return out;
  }

  // { "upper|lower": [reasons] } where a rest height is outside the population mean +- 2 SD.
  function normFlags(hs) {
    const out = {};
    for (const [u, l, ...h] of hs) {
      const ref = NORMS[label(u, l)];
      if (!ref) continue;
      const why = [];
      ["A", "M", "P"].forEach((k, i) => {
        const [mu, sd] = ref[i];
        if (Math.abs(h[i] - mu) > 2 * sd) why.push(`${k} ${h[i].toFixed(1)} mm (norm ${mu.toFixed(1)} ± ${(2 * sd).toFixed(1)})`);
      });
      if (why.length) out[`${u}|${l}`] = why;
    }
    return out;
  }

  return { pairs, label, heights, restHeights, postureFlags, normFlags, KEYS, NORMS, MEAN_CHANGE, LOCAL_STRAIN, MIN_REST };
})();

if (typeof module !== "undefined") module.exports = Discs;
