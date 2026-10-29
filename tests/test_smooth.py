import unittest

import numpy as np

from implicit.expr import Param
from implicit.export import mesh_report
from implicit.mesh import mesh_sdf
from implicit.sdf import Grid, smooth


def bumpy_sphere(h=0.5, r=10.0, amp=0.6, seed=0):
    """Sphere SDF with high-frequency radial bumps (like CT surface noise)."""
    rng = np.random.default_rng(seed)
    ax = np.arange(-16, 16 + h / 2, h)
    X, Y, Z = np.meshgrid(ax, ax, ax, indexing="ij")
    R = np.sqrt(X**2 + Y**2 + Z**2)
    th, ph = np.arccos(np.clip(Z / np.maximum(R, 1e-9), -1, 1)), np.arctan2(Y, X)
    bumps = amp * np.sin(9 * th + rng.uniform(0, 6)) * np.sin(11 * ph + rng.uniform(0, 6))
    return Grid(R - r - bumps, (-16, -16, -16), h, "bumpy")


def area_and_volume(sdf, h=0.4):
    r = mesh_report(*mesh_sdf(sdf.eval, (-16,) * 3, (16,) * 3, h))
    return r["area_mm2"], r["volume_mm3"], r["closed"]


class Smoothing(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.g = bumpy_sphere()
        cls.sigma = Param("sigma", 0, 0, 4)
        cls.keep = Param("keep", 1, 0, 1)
        cls.s = smooth(cls.g, cls.sigma, cls.keep)
        cls.area0, cls.vol0, _ = area_and_volume(cls.g)

    def setUp(self):
        self.sigma.set(0.0)
        self.keep.set(1.0)

    def test_sigma_zero_is_the_original_field(self):
        p = np.random.default_rng(1).uniform(-15, 15, (500, 3))
        np.testing.assert_allclose(self.s.eval(p), self.g.eval(p), atol=1e-9)

    def test_smoothing_reduces_roughness_and_keeps_volume(self):
        self.sigma.set(1.5)
        area, vol, closed = area_and_volume(self.s)
        self.assertTrue(closed)                         # still a closed, meshable surface
        self.assertLess(area, 0.97 * self.area0)        # bumps flattened
        self.assertAlmostEqual(vol / self.vol0, 1.0, delta=0.02)

    def test_without_preservation_the_body_shrinks(self):
        self.sigma.set(3.0)
        self.keep.set(0.0)
        _, vol, _ = area_and_volume(self.s)
        self.assertLess(vol, 0.99 * self.vol0)

    def test_continuous_across_pyramid_levels(self):
        p = np.random.default_rng(2).uniform(-12, 12, (300, 3))
        prev = None
        for sig in np.linspace(0, 4, 81):
            self.sigma.set(sig)
            d = self.s.eval(p)
            if prev is not None:
                self.assertLess(np.abs(d - prev).max(), 0.25, sig)  # no jumps between levels
            prev = d

    def test_lipschitz(self):
        self.sigma.set(2.2)
        rng = np.random.default_rng(3)
        a = rng.uniform(-14, 14, (2000, 3))
        b = a + rng.normal(0, 0.5, a.shape)
        ratio = np.abs(self.s.eval(a) - self.s.eval(b)) / np.linalg.norm(a - b, axis=1)
        self.assertLess(ratio.max(), 1.05)  # ray marching and meshing stay valid


if __name__ == "__main__":
    unittest.main()
