import math
import unittest
from pathlib import Path

import numpy as np

from implicit import Box, Capsule, Cylinder, HalfSpace, Param, Scene, Sphere
from implicit.mesh import mesh_volume, surface_nets
from implicit.project import load_scene

ROOT = Path(__file__).resolve().parent.parent


def pts(*rows):
    return np.array(rows, dtype=float)


class Primitives(unittest.TestCase):
    def test_sphere(self):
        d = Sphere(10).eval(pts([0, 0, 0], [20, 0, 0], [0, 6, 8]))
        np.testing.assert_allclose(d, [-10, 10, 0])

    def test_box_exact_outside_corner(self):
        d = Box((1, 2, 3)).eval(pts([4, 6, 3], [0, 0, 0]))
        np.testing.assert_allclose(d, [5, -1])

    def test_cylinder(self):
        d = Cylinder(5, 10).eval(pts([8, 0, 0], [0, 0, 13], [8, 0, 14]))
        np.testing.assert_allclose(d, [3, 3, 5])

    def test_capsule(self):
        d = Capsule((0, 0, 0), (0, 0, 10), 2).eval(pts([5, 0, 5], [0, 0, 15]))
        np.testing.assert_allclose(d, [3, 3])

    def test_halfspace_normalises(self):
        d = HalfSpace((0, 0, 1), (0, 0, 7)).eval(pts([0, 0, 4], [3, 3, -1]))
        np.testing.assert_allclose(d, [3, -2])


class Operations(unittest.TestCase):
    def test_booleans(self):
        a, b = Sphere(5), Sphere(5).translate((6, 0, 0))
        p = pts([-4, 0, 0], [3, 0, 0], [10, 0, 0])
        np.testing.assert_allclose((a | b).eval(p), np.minimum(a.eval(p), b.eval(p)))
        np.testing.assert_allclose((a & b).eval(p), np.maximum(a.eval(p), b.eval(p)))
        np.testing.assert_allclose((a - b).eval(p), np.maximum(a.eval(p), -b.eval(p)))

    def test_smooth_union_is_below_min(self):
        a, b = Sphere(5), Sphere(5).translate((8, 0, 0))
        p = pts([4, 0, 0])
        self.assertLess(a.smooth_union(b, 2).eval(p)[0], (a | b).eval(p)[0])

    def test_rotation_right_hand(self):
        # A point marker at +y rotated +90 deg about x must land on +z.
        for axis, src, dst in (("x", (0, 5, 0), (0, 0, 5)),
                               ("y", (0, 0, 5), (5, 0, 0)),
                               ("z", (5, 0, 0), (0, 5, 0))):
            s = Sphere(1).translate(src).rotate(axis, 90)
            self.assertAlmostEqual(s.eval(pts(dst))[0], -1, places=6, msg=axis)

    def test_rotation_about_center(self):
        s = Sphere(1).translate((0, 0, 10)).rotate_x(180, center=(0, 0, 5))
        self.assertAlmostEqual(s.eval(pts([0, 0, 0]))[0], -1, places=6)

    def test_shell_and_offset(self):
        s = Sphere(10)
        np.testing.assert_allclose(s.shell(2).eval(pts([0, 0, 9], [0, 0, 5])), [-1, 3])
        np.testing.assert_allclose(s.offset(-2).eval(pts([0, 0, 8])), [0], atol=1e-12)

    def test_param_drives_geometry(self):
        r = Param("r", 10, 1, 20)
        s = Sphere(r)
        self.assertAlmostEqual(s.eval(pts([0, 0, 12]))[0], 2)
        r.set(12)
        self.assertAlmostEqual(s.eval(pts([0, 0, 12]))[0], 0)


class Glsl(unittest.TestCase):
    def test_scene_glsl_has_uniforms_and_bodies(self):
        r = Param("radius", 10, 1, 20)
        sc = Scene("t", ((-20,) * 3, (20,) * 3))
        sc.add("a", Sphere(r).rotate_x(r) - Box((1, 2, 3)))
        g = sc.glsl()
        self.assertIn("uniform float u_radius;", g)
        self.assertIn("float body0(vec3 p)", g)
        self.assertIn("vec2 sceneMap(vec3 p)", g)
        self.assertEqual(sc.to_json()["params"][0]["name"], "radius")

    def test_duplicate_param_names_rejected(self):
        sc = Scene("t", ((-1,) * 3, (1,) * 3))
        sc.add("a", Sphere(Param("r", 1, 0, 2)) | Sphere(Param("r", 1, 0, 2)))
        with self.assertRaises(ValueError):
            sc.params()


class Meshing(unittest.TestCase):
    def test_sphere_volume_and_orientation(self):
        sc = Scene("t", ((-11,) * 3, (11,) * 3))
        sc.add("s", Sphere(10))
        v, t = surface_nets(*sc.sample("s", 0.4))
        vol = mesh_volume(v, t)
        self.assertAlmostEqual(vol / (4 / 3 * math.pi * 1000), 1.0, delta=0.01)

    def test_closed_manifold_edges(self):
        sc = Scene("t", ((-6,) * 3, (6,) * 3))
        sc.add("s", Box((4, 3, 2), round=0.5))
        v, t = surface_nets(*sc.sample("s", 0.5))
        e = np.sort(np.concatenate([t[:, [0, 1]], t[:, [1, 2]], t[:, [2, 0]]]), axis=1)
        _, counts = np.unique(e, axis=0, return_counts=True)
        self.assertTrue(np.all(counts == 2))


class Examples(unittest.TestCase):
    def test_spine_segment_loads_and_meshes(self):
        sc = load_scene(ROOT / "examples" / "spine_segment.py")
        self.assertGreaterEqual(len(sc.params()), 10)
        sc.set_params({"pso_angle": 20, "pso_close": 1.0, "lordosis": 12})
        v, t = surface_nets(*sc.sample("Cage", 1.0))
        self.assertGreater(mesh_volume(v, t), 0)


if __name__ == "__main__":
    unittest.main()
