"""Ranges of motion and water-filling (implicit.anatomy.motion)."""

import json
import shutil
import subprocess
import unittest
from pathlib import Path

import numpy as np

from implicit.anatomy import motion

ROOT = Path(__file__).resolve().parents[1]


class TestFill(unittest.TestCase):
    def test_fill(self):
        rng = np.random.default_rng(0)
        for _ in range(200):
            n = rng.integers(1, 8)
            w = list(rng.uniform(0, 1, n) * (rng.uniform(0, 1, n) > 0.2))
            caps = list(rng.uniform(0.1, 2.0, n))
            total = float(rng.uniform(-8, 8))
            x, miss = motion.fill(total, w, caps)
            self.assertAlmostEqual(sum(x) + miss, total, places=9)
            act = [j for j in range(n) if w[j] > 0]
            for j in range(n):
                self.assertLessEqual(abs(x[j]), caps[j] + 1e-12)
                if j not in act:
                    self.assertEqual(x[j], 0.0)
            if act and miss != 0:
                self.assertTrue(all(abs(abs(x[j]) - caps[j]) < 1e-12 for j in act))
            free = [j for j in act if abs(x[j]) < caps[j] - 1e-12]
            if len(free) > 1:  # unsaturated levels keep the weights' proportion
                r = [x[j] / w[j] for j in free]
                self.assertAlmostEqual(min(r), max(r), places=9)

    def test_fill_unlimited_is_proportional(self):
        x, miss = motion.fill(3.0, [0.2, 0.3, 0.5], [float("inf")] * 3)
        np.testing.assert_allclose(x, [0.6, 0.9, 1.5])
        self.assertEqual(miss, 0.0)

    def test_fill_continuous(self):
        w, caps = [0.1, 0.4, 0.5], [0.3, 0.5, 2.0]
        T = np.linspace(-4, 4, 2001)
        X = np.array([motion.fill(t, w, caps)[0] for t in T])
        self.assertLess(np.abs(np.diff(X, axis=0)).max(), 2 * (T[1] - T[0]))

    @unittest.skipUnless(shutil.which("node"), "node not installed")
    def test_viewer_fill_matches(self):
        rng = np.random.default_rng(5)
        cases = [(float(rng.uniform(-6, 6)), list(rng.uniform(0, 1, 5)), list(rng.uniform(0.1, 2, 5)))
                 for _ in range(50)]
        js = ("const S = require(process.argv[1]); const c = JSON.parse(require('fs').readFileSync(0));"
              "process.stdout.write(JSON.stringify(c.map(([t, w, k]) => S.fill(t, w, k))));")
        out = subprocess.run(["node", "-e", js, str(ROOT / "viewer" / "solvers.js")], check=True,
                             capture_output=True, input=json.dumps(cases), text=True).stdout
        for (t, w, k), (x, miss) in zip(cases, json.loads(out)):
            rx, rmiss = motion.fill(t, w, k)
            np.testing.assert_allclose(x, rx, atol=1e-12)
            self.assertAlmostEqual(miss, rmiss, places=12)


class TestTables(unittest.TestCase):
    def test_every_level_has_a_range(self):
        levels = [f"T{i}" for i in range(1, 13)] + [f"L{i}" for i in range(1, 6)]
        for up, lo in zip(levels, levels[1:] + ["S1"]):
            fe, lb, ar = motion.ROM[f"{up}-{lo}"]
            self.assertTrue(0 < fe < 15 and 0 < lb < 15 and 0 < ar < 10, up)
        # Lumbar flexion-extension grows caudally (Wilke et al. 2025: L1-L2 6.4 ... L5-S1 10.8).
        self.assertLess(motion.ROM["L1-L2"][0], motion.ROM["L5-S1"][0])
        self.assertAlmostEqual(motion.COUPLING["T6-T7"], 0.397, places=3)
        self.assertEqual(motion.COUPLING["L3-L4"], 0.0)


if __name__ == "__main__":
    unittest.main()
