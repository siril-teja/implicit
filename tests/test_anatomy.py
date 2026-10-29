import unittest

import numpy as np

try:
    import scipy  # noqa: F401
    HAVE_SCIPY = True
except ImportError:
    HAVE_SCIPY = False

from implicit import Scene
from implicit.sdf import QSTEP, Grid


def synthetic_vertebra(tilt_deg=12.0, shape=(90, 110, 80)):
    """Voxel mask (1 mm, identity affine): elliptic body tilted about x + posterior arch."""
    idx = np.indices(shape).reshape(3, -1).T.astype(float)
    c = np.array([45.0, 60.0, 40.0])
    t = np.radians(tilt_deg)
    R = np.array([[1, 0, 0], [0, np.cos(t), -np.sin(t)], [0, np.sin(t), np.cos(t)]])
    p = (idx - c) @ R  # local coordinates (rotate world -> local)
    body = ((p[:, 0] / 22) ** 2 + (p[:, 1] / 16) ** 2 <= 1) & (np.abs(p[:, 2]) <= 13)

    def capsule(a, b, r):
        a, b = np.array(a, float), np.array(b, float)
        ba = b - a
        h = np.clip((p - a) @ ba / (ba @ ba), 0, 1)
        return np.linalg.norm(p - a - h[:, None] * ba, axis=1) <= r

    arch = (capsule((12, -14, 4), (13, -30, 4), 4) | capsule((-12, -14, 4), (-13, -30, 4), 4)
            | capsule((13, -30, 3), (0, -40, 2), 3.5) | capsule((-13, -30, 3), (0, -40, 2), 3.5)
            | capsule((0, -40, 2), (0, -52, -4), 3.5))
    mask = (body | arch).reshape(shape)
    return mask, c, R


class GridField(unittest.TestCase):
    def test_grid_matches_sphere_and_bounds_outside(self):
        ax = np.arange(-15, 15.5, 1.0)
        X, Y, Z = np.meshgrid(ax, ax, ax, indexing="ij")
        g = Grid(np.sqrt(X**2 + Y**2 + Z**2) - 10, (-15, -15, -15), 1.0)
        p = np.array([[0, 0, 0], [10, 0, 0], [5.5, 3.2, 1.1], [40, 0, 0]], float)
        d = g.eval(p)
        exact = np.linalg.norm(p, axis=1) - 10
        np.testing.assert_allclose(d[:3], exact[:3], atol=0.15)
        self.assertLessEqual(d[3], exact[3])  # exterior is a lower bound
        self.assertGreater(d[3], 0)

    def test_atlas_packing_offsets_and_glsl(self):
        sc = Scene("t", ((-20,) * 3, (20,) * 3))
        a = Grid(np.ones((10, 12, 14)), (0, 0, 0), 1.0, "a")
        b = Grid(np.ones((30, 8, 9)), (0, 0, 0), 1.0, "b")
        sc.add("a", a)
        sc.add("b", b | a)
        js = sc.to_json()
        self.assertIn("atlas", js)
        w, h, d = js["atlas"]["size"]
        for g in (a, b):  # blocks lie inside the atlas, in texture (zyx) order
            ox, oy, oz = g.atlas_offset
            self.assertLessEqual(ox + g.shape[2], w)
            self.assertLessEqual(oy + g.shape[1], h)
            self.assertLessEqual(oz + g.shape[0], d)
        self.assertIn("sampler3D u_atlas", js["glsl"])
        self.assertEqual(len(sc.grids()), 2)  # shared grid packed once

    def test_quantisation_step(self):
        g = Grid(np.full((4, 4, 4), 0.37), (0, 0, 0), 1.0)
        self.assertAlmostEqual(float(g.values[0, 0, 0]), 0.4, delta=QSTEP / 2 + 1e-6)


@unittest.skipUnless(HAVE_SCIPY, "scipy not installed")
class Endplates(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from implicit.anatomy.preprocess import Field, mask_to_grid, vertebra_landmarks
        cls.tilt = 12.0
        mask, c, R = synthetic_vertebra(cls.tilt)
        v, o = mask_to_grid(mask, np.eye(4), 1.0)
        cls.f = Field(v, o, 1.0)
        canal = c + R @ np.array([0, -24.0, 0])
        cls.e = vertebra_landmarks(cls.f, "L3", R[:, 2], canal)
        cls.c, cls.R = c, R

    def test_sdf_is_accurate(self):
        # A point 3 mm above the top endplate centre (outside) and 3 mm below (inside).
        top = self.c + self.R @ np.array([0, 0, 13.0])
        d = self.f(np.array([top + self.R[:, 2] * 3, top - self.R[:, 2] * 3]))
        np.testing.assert_allclose(d, [3, -3], atol=0.6)

    def test_endplate_normals(self):
        z = self.R[:, 2]
        for which, sign in (("sup", 1), ("inf", -1)):
            ang = np.degrees(np.arccos(np.clip(self.e[which]["normal"] @ (sign * z), -1, 1)))
            self.assertLess(ang, 2.0, which)

    def test_endplate_heights_and_body(self):
        for which, h in (("sup", 13.0), ("inf", -13.0)):
            p = self.e[which]["point"]
            self.assertAlmostEqual(float((p - self.c) @ self.R[:, 2]), h, delta=0.8)
        self.assertAlmostEqual(self.e["width"], 44, delta=3)
        self.assertAlmostEqual(self.e["depth"], 32, delta=4)  # arch removed


if __name__ == "__main__":
    unittest.main()


def write_synthetic_subject(d):
    """L3-L5 (27 mm bodies, 10 mm discs, 4 deg lordosis each) + a sacral block, as NIfTI."""
    import nibabel as nib
    shape = (100, 130, 150)
    idx = np.indices(shape).reshape(3, -1).T.astype(float)
    labels = {}
    z, ang = 95.0, 0.0
    for lv in ("L3", "L4", "L5"):
        t = np.radians(ang)
        R = np.array([[1, 0, 0], [0, np.cos(t), -np.sin(t)], [0, np.sin(t), np.cos(t)]])
        p = (idx - np.array([50.0, 70.0, z])) @ R
        body = ((p[:, 0] / 22) ** 2 + (p[:, 1] / 16) ** 2 <= 1) & (np.abs(p[:, 2]) <= 13.5)
        ped = (np.abs(np.abs(p[:, 0]) - 12) < 4) & (p[:, 1] < -12) & (p[:, 1] > -30) & (np.abs(p[:, 2] - 4) < 4)
        lam = (p[:, 1] < -28) & (p[:, 1] > -36) & (np.abs(p[:, 0]) < 16) & (np.abs(p[:, 2]) < 6)
        labels[lv] = (body | ped | lam).reshape(shape)
        z -= 37.0
        ang += 4.0
    t = np.radians(ang + 10)
    R = np.array([[1, 0, 0], [0, np.cos(t), -np.sin(t)], [0, np.sin(t), np.cos(t)]])
    p = (idx - np.array([50.0, 68.0, z + 3])) @ R
    labels["sacrum"] = ((np.abs(p[:, 0]) < 30) & (np.abs(p[:, 1]) < 20) & (p[:, 2] < 13.5)
                        & (p[:, 2] > -20)).reshape(shape)
    for name, m in labels.items():
        fn = f"vertebrae_{name}.nii.gz" if name != "sacrum" else "sacrum.nii.gz"
        nib.save(nib.Nifti1Image(m.astype(np.uint8), np.eye(4)), str(d / fn))


@unittest.skipUnless(HAVE_SCIPY, "scipy not installed")
class SpinePipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import tempfile
        from pathlib import Path
        from implicit.anatomy.preprocess import process
        from implicit.anatomy.spine import build_scene
        cls.tmp = tempfile.TemporaryDirectory()
        d = Path(cls.tmp.name)
        (d / "subj").mkdir()
        write_synthetic_subject(d / "subj")
        process(d / "subj", d / "model", log=lambda *a: None)
        cls.scene, cls.model = build_scene(d / "model")

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_one_implicit_body_per_bone(self):
        self.assertEqual(set(self.model.marks), {"L3", "L4", "L5", "S1"})
        names = [b.name for b in self.scene.bodies]
        for bone in ("L3", "L4", "L5", "sacrum"):
            self.assertIn(bone, names)

    def test_endplate_geometry(self):
        m = self.model.marks
        n_u, n_l = m["L3"]["inf"]["normal"], m["L4"]["sup"]["normal"]
        ang = np.degrees(np.arccos(np.clip(-n_u @ n_l, -1, 1)))
        self.assertAlmostEqual(ang, 4.0, delta=1.5)
        gap = (m["L3"]["inf"]["point"] - m["L4"]["sup"]["point"]) @ n_l
        self.assertAlmostEqual(gap, 10.0, delta=1.0)

    def test_bone_volume(self):
        from implicit.mesh import mesh_volume, surface_nets
        vol = mesh_volume(*surface_nets(*self.scene.sample("L4", 1.0)))
        body = np.pi * 22 * 16 * 27  # synthetic elliptic body; arch adds a little
        self.assertGreater(vol, 0.95 * body)
        self.assertLess(vol, 1.4 * body)
