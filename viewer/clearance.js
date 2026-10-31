// Clearance between neighbouring bones (implicit/anatomy/clearance.py, same computation
// on the same surface points and grid values): each bone's grid is read from the CPU
// copy of the atlas and evaluated exactly as implicit.sdf.Grid.eval (trilinear; outside
// the grid the larger of distance to the box + margin and value at the nearest grid
// point - distance).
"use strict";

const Clearance = (() => {
  const VERTEBRAE = [..."1234567"].map((i) => `C${i}`)
    .concat(Array.from({ length: 12 }, (_, i) => `T${i + 1}`), Array.from({ length: 5 }, (_, i) => `L${i + 1}`));
  const QSTEP = 0.1, QZERO = 128;
  const COLLIDE = 0.0;

  // Neighbouring pairs among ``names`` (as clearance.py pairs()).
  function pairs(names) {
    const has = new Set(names), levels = VERTEBRAE.filter((lv) => has.has(lv)), out = [];
    for (let i = 0; i + 1 < levels.length; i++) out.push([levels[i], levels[i + 1]]);
    if (levels.length && has.has("sacrum")) out.push([levels[levels.length - 1], "sacrum"]);
    for (const side of ["left", "right"]) {
      const hip = `hip_${side}`, femur = `femur_${side}`;
      if (!has.has(hip)) continue;
      for (const [a, b] of [["sacrum", hip], ["L5", hip], [hip, femur]]) if (has.has(a) && has.has(b)) out.push([a, b]);
    }
    for (const side of ["left", "right"]) {  // ribs: their vertebrae and the next rib
      for (let n = 1; n <= 12; n++) {
        const rib = `rib${n}_${side}`;
        if (!has.has(rib)) continue;
        for (const v of [`T${n - 1}`, `T${n}`]) if (has.has(v)) out.push([v, rib]);
        if (has.has(`rib${n + 1}_${side}`)) out.push([rib, `rib${n + 1}_${side}`]);
      }
    }
    return out;
  }

  // Grid value at point q (grid frame), as Grid.eval; g: { origin, spacing, n, offset, margin }.
  function gridEval(atlas, size, g, x, y, z) {
    const [W, H] = size, h = g.spacing, n = g.n.map((v) => v - 1);
    const q = [(x - g.origin[0]) / h, (y - g.origin[1]) / h, (z - g.origin[2]) / h];
    const qc = q.map((v, i) => Math.min(Math.max(v, 0), n[i]));
    const i0 = qc.map((v, i) => Math.min(Math.floor(v), Math.max(n[i] - 1, 0)));
    const f = qc.map((v, i) => v - i0[i]);
    const at = (i, j, k) => (atlas[(g.offset[2] + i) * W * H + (g.offset[1] + j) * W + (g.offset[0] + k)] - QZERO) * QSTEP;
    let out = 0;
    for (const dx of [0, 1]) for (const dy of [0, 1]) for (const dz of [0, 1]) {
      const w = (dx ? f[0] : 1 - f[0]) * (dy ? f[1] : 1 - f[1]) * (dz ? f[2] : 1 - f[2]);
      out += w * at(Math.min(i0[0] + dx, n[0]), Math.min(i0[1] + dy, n[1]), Math.min(i0[2] + dz, n[2]));
    }
    const ext = Math.hypot(q[0] - qc[0], q[1] - qc[1], q[2] - qc[2]) * h;
    return ext > 0 ? Math.max(ext + g.margin, out - ext) : out;
  }

  // [[a, b, clearance, point]] for the neighbouring pairs of ``bodies``:
  // { name: { grid, samples (flat, grid frame), motion: [R rows, t] } }.
  function compute(atlas, size, bodies) {
    const out = [];
    for (const [a, b] of pairs(Object.keys(bodies))) {
      let best = [Infinity, null];
      for (const [s, o] of [[a, b], [b, a]]) {
        const S = bodies[s], O = bodies[o], [Rs, ts] = S.motion, [Ro, to] = O.motion, P = S.samples;
        for (let i = 0; i < P.length; i += 3) {
          const w = [0, 1, 2].map((r) => Rs[r][0] * P[i] + Rs[r][1] * P[i + 1] + Rs[r][2] * P[i + 2] + ts[r]);
          const d0 = [w[0] - to[0], w[1] - to[1], w[2] - to[2]];  // into o's frame: R^T (w - t)
          const l = [0, 1, 2].map((c) => Ro[0][c] * d0[0] + Ro[1][c] * d0[1] + Ro[2][c] * d0[2]);
          const d = gridEval(atlas, size, O.grid, l[0], l[1], l[2]);
          if (d < best[0]) best = [d, w];
        }
      }
      out.push([a, b, best[0], best[1]]);
    }
    return out;
  }

  return { pairs, gridEval, compute, COLLIDE };
})();

if (typeof module !== "undefined") module.exports = Clearance;
