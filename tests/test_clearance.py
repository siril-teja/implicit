"""Clearance between neighbouring bones (implicit.anatomy.clearance, viewer/clearance.js)."""

import json
import shutil
import subprocess
import unittest
from pathlib import Path

import numpy as np

from implicit.anatomy import clearance as cl
from implicit.graph import load_graph
from implicit.scene import pack_grids
from implicit.sdf import Grid

ROOT = Path(__file__).resolve().parents[1]


def ball(radius=10.0, h=1.0, pad=6.0):
    """Grid of a ball centred at the origin."""
    n = int(np.ceil(2 * (radius + pad) / h)) + 1
    o = -(n - 1) * h / 2
    x = o + h * np.arange(n)
    X, Y, Z = np.meshgrid(x, x, x, indexing="ij")
    return Grid(np.sqrt(X ** 2 + Y ** 2 + Z ** 2) - radius, [o] * 3, h)


def rot_z(deg):
    a = np.radians(deg)
    return np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1.0]])


class TestClearance(unittest.TestCase):
    def test_pairs(self):
        names = ["sacrum", "L5", "L4", "T12", "hip_left", "femur_left", "femur_right"]
        self.assertEqual(cl.pairs(names), [("T12", "L4"), ("L4", "L5"), ("L5", "sacrum"), ("sacrum", "hip_left"),
                                           ("L5", "hip_left"), ("hip_left", "femur_left")])
        ribs = ["T4", "T5", "rib4_left", "rib5_left", "rib5_right"]
        self.assertEqual(cl.pairs(ribs), [("T4", "T5"), ("T4", "rib4_left"), ("rib4_left", "rib5_left"),
                                          ("T4", "rib5_left"), ("T5", "rib5_left"), ("T4", "rib5_right"),
                                          ("T5", "rib5_right")])

    def test_balls(self):
        """Gap and overlap of two balls (radius 10) to within the sample spacing."""
        a, b = ball(), ball()
        samples = {"L4": cl.surface_samples(a), "L5": cl.surface_samples(b)}
        for d, want in ((25.0, 5.0), (16.0, -4.0)):
            R = rot_z(30)
            bodies = {"L4": (a, (np.eye(3), np.zeros(3))), "L5": (b, (R, np.array([d, 0, 0])))}
            (x, y, c, p), = cl.clearance(bodies, samples)
            self.assertEqual((x, y), ("L4", "L5"))
            self.assertAlmostEqual(c, want, delta=0.3)
            self.assertLess(abs(p[1]) + abs(p[2]), 3.0)  # found on the line of centres

    @unittest.skipUnless(shutil.which("node"), "node not installed")
    def test_viewer_matches(self):
        """viewer/clearance.js reads the atlas and finds the same clearance."""
        a, b = ball(10.0), ball(7.0, h=0.8)
        atlas = pack_grids([a, b])
        bodies = {"L4": (a, (rot_z(10), np.array([1.0, -2, 3]))), "L5": (b, (rot_z(-50), np.array([17.5, 3, 4])))}
        samples = {k: cl.surface_samples(g) for k, (g, _) in bodies.items()}
        ref = cl.clearance(bodies, samples)
        js_bodies = {k: {"grid": {"origin": g.origin.tolist(), "spacing": g.spacing, "n": list(g.shape),
                                  "offset": list(g.atlas_offset), "margin": g.margin},
                         "samples": samples[k].ravel().tolist(), "motion": [R.tolist(), t.tolist()]}
                     for k, (g, (R, t)) in bodies.items()}
        js = ("const C = require(process.argv[1]); const d = JSON.parse(require('fs').readFileSync(0));"
              "process.stdout.write(JSON.stringify(C.compute(Uint8Array.from(d.atlas), d.size, d.bodies)));")
        out = subprocess.run(["node", "-e", js, str(ROOT / "viewer" / "clearance.js")], check=True,
                             capture_output=True, text=True,
                             input=json.dumps({"atlas": atlas.ravel().tolist(),
                                               "size": [atlas.shape[2], atlas.shape[1], atlas.shape[0]],
                                               "bodies": js_bodies})).stdout
        got = json.loads(out)
        self.assertEqual(len(got), len(ref))
        for (x, y, c, p), (gx, gy, gc, gp) in zip(ref, got):
            self.assertEqual((x, y), (gx, gy))
            self.assertAlmostEqual(c, gc, places=6)
            np.testing.assert_allclose(p, gp, atol=1e-9)


class TestSpineClearance(unittest.TestCase):
    """The average spine: clean at its measured posture, colliding at LL 80 / TK 60."""

    @classmethod
    def setUpClass(cls):
        cls.scene = load_graph(ROOT / "examples" / "average_spine.json")
        cls.samples = {n: cl.surface_samples(g) for n, (g, _) in cl.scene_bodies(cls.scene).items()}
        cls.defaults = {p.name: p.value() for p in cls.scene.params()}

    def tearDown(self):
        self.scene.set_params(self.defaults)

    def clearances(self, **values):
        self.scene.set_params({**self.defaults, **values})
        return {(a, b): c for a, b, c, _ in cl.report(self.scene, self.samples)}

    def test_baseline_clean(self):
        c = self.clearances()
        self.assertEqual(len(c), 23 + 4 * 12 - 2 + 2 * 11)  # spine, ribs with their vertebrae, rib to rib
        self.assertTrue(all(v > cl.COLLIDE for v in c.values()), {k: v for k, v in c.items() if v <= cl.COLLIDE})

    def test_hyperlordosis_collides(self):
        c = self.clearances(spp_LL=80, spp_TK=60)
        hits = {k for k, v in c.items() if v < cl.COLLIDE}
        self.assertEqual(hits, {("L4", "L5")})
        self.assertLess(c[("L4", "L5")], -0.03)


if __name__ == "__main__":
    unittest.main()
