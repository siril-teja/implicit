import unittest

import numpy as np

from implicit import Box, Sphere
from implicit.export import mesh_report
from implicit.mesh import mesh_sdf


class MarchingTetrahedra(unittest.TestCase):
    def check(self, sdf, lo, hi, h, volume=None, rtol=0.01):
        r = mesh_report(*mesh_sdf(sdf.eval, lo, hi, h))
        self.assertTrue(r["closed"], "every edge must be shared by exactly two triangles")
        self.assertGreater(r["volume_mm3"], 0, "triangles must face outwards")
        if volume is not None:
            self.assertAlmostEqual(r["volume_mm3"] / volume, 1.0, delta=rtol)
        return r

    def test_sphere_volume(self):
        self.check(Sphere(10), (-12,) * 3, (12,) * 3, 0.5, 4 / 3 * np.pi * 1000, 0.005)

    def test_touching_spheres_stay_manifold(self):
        # A single contact point is the classic non-manifold case for surface nets.
        two = Sphere(5).translate((5.0, 0, 0)) | Sphere(5).translate((-5.0, 0, 0))
        self.check(two, (-11, -6, -6), (11, 6, 6), 0.5)

    def test_thin_plate_and_sharp_box(self):
        self.check(Box((8, 8, 0.6)), (-10,) * 3, (10,) * 3, 0.4, 16 * 16 * 1.2, 0.05)
        self.check(Box((5, 4, 3)), (-7,) * 3, (7,) * 3, 0.5, 10 * 8 * 6, 0.03)

    def test_narrow_band_matches_dense_result(self):
        # Coarse blocks far from the surface are skipped; the mesh must not change.
        a = mesh_report(*mesh_sdf(Sphere(6).eval, (-8,) * 3, (8,) * 3, 0.5, block=4))
        b = mesh_report(*mesh_sdf(Sphere(6).eval, (-8,) * 3, (8,) * 3, 0.5, block=32))
        self.assertEqual(a["triangles"], b["triangles"])
        self.assertAlmostEqual(a["volume_mm3"], b["volume_mm3"], places=6)


if __name__ == "__main__":
    unittest.main()
