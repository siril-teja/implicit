import unittest

import numpy as np

from implicit import Box, Scene, Sphere
from implicit.diagnostics import probes
from implicit.sdf import Grid


class Probes(unittest.TestCase):
    def setUp(self):
        sc = Scene("t", ((-20,) * 3, (20,) * 3))
        ax = np.arange(-12, 12.5, 1.0)
        X, Y, Z = np.meshgrid(ax, ax, ax, indexing="ij")
        self.grid = Grid(np.sqrt(X**2 + Y**2 + Z**2) - 8, (-12, -12, -12), 1.0, "ball")
        sc.add("ball", self.grid, bounds=((-12,) * 3, (12,) * 3))
        sc.add("box", Box((3, 4, 5)).translate((10, 0, 0)))
        sc.to_json()  # pack the atlas
        self.scene, self.p = sc, probes(sc)

    def test_one_entry_per_unit_with_reference_values(self):
        self.assertEqual([u["unit"] for u in self.p["units"]], [0, 1])
        for u, body in zip(self.p["units"], self.scene.bodies):
            P = np.array(u["points"])
            np.testing.assert_allclose(P[:, 3], body.sdf.eval(P[:, :3]), atol=1e-4)

    def test_grid_points_cover_inside_and_outside(self):
        d = np.array(self.p["units"][0]["points"])[:, 3]
        self.assertLess(d.min(), -5)      # deep inside the ball
        self.assertGreater(d.max(), 3)    # outside the grid

    def test_ray_targets_are_inside(self):
        for u, body in zip(self.p["units"], self.scene.bodies):
            self.assertLess(body.sdf.eval(np.array([u["target"]]))[0], 0)


if __name__ == "__main__":
    unittest.main()
