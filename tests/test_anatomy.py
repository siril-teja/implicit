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
        from implicit.anatomy.convert import bone_sdf
        from implicit.anatomy.landmarks import Field, vertebra_landmarks
        cls.tilt = 12.0
        mask, c, R = synthetic_vertebra(cls.tilt)
        v, o, _ = bone_sdf(mask, np.eye(4), 1.0)
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
    """Masks -> one implicit file per bone -> landmarks -> scene."""

    @classmethod
    def setUpClass(cls):
        import tempfile
        from pathlib import Path
        from implicit.anatomy.convert import convert
        from implicit.anatomy.landmarks import detect
        from implicit.anatomy.spine import build_scene
        cls.tmp = tempfile.TemporaryDirectory()
        d = Path(cls.tmp.name)
        (d / "subj").mkdir()
        write_synthetic_subject(d / "subj")
        cls.meta = convert(d / "subj", d / "bones", ["L3", "L4", "L5", "sacrum"],
                           spacing={"vertebra": 1.0, "sacrum": 1.0}, log=lambda *a: None)
        cls.bone_dir = d / "bones"
        detect(cls.bone_dir, log=lambda *a: None)
        cls.scene, cls.model = build_scene(cls.bone_dir)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_one_file_per_bone(self):
        files = sorted(p.name for p in self.bone_dir.glob("*.npz"))
        self.assertEqual(files, ["L3.npz", "L4.npz", "L5.npz", "sacrum.npz"])
        self.assertTrue((self.bone_dir / "ATTRIBUTION.txt").exists())
        self.assertEqual(self.meta["L4"]["kind"], "vertebra")
        self.assertFalse(self.meta["L4"]["ct_refined"])

    def test_round_trip(self):
        from implicit.anatomy.convert import load_bone
        g, meta = load_bone(self.bone_dir / "L4.npz")
        self.assertEqual(meta["name"], "L4")
        self.assertEqual(meta["format_version"], 1)
        self.assertEqual(g.spacing, 1.0)
        self.assertIs(type(self.scene.body("L4").sdf), type(g))

    def test_landmarks_from_implicit_bones(self):
        m = self.model.marks
        self.assertEqual(set(m), {"L3", "L4", "L5", "S1"})
        n_u, n_l = m["L3"]["inf"]["normal"], m["L4"]["sup"]["normal"]
        ang = np.degrees(np.arccos(np.clip(-n_u @ n_l, -1, 1)))
        self.assertAlmostEqual(ang, 4.0, delta=1.5)
        gap = (m["L3"]["inf"]["point"] - m["L4"]["sup"]["point"]) @ n_l
        self.assertAlmostEqual(gap, 10.0, delta=1.0)

    def test_bone_volume(self):
        body = np.pi * 22 * 16 * 27  # synthetic elliptic body; the arch adds a little
        vol = self.meta["L4"]["volume_mm3"]
        self.assertGreater(vol, 0.95 * body)
        self.assertLess(vol, 1.4 * body)


@unittest.skipUnless(HAVE_SCIPY, "scipy not installed")
class CTRefinement(unittest.TestCase):
    """Sphere of known radius in a 1.5 mm partial-volume CT, labelled with a mask."""

    def surface_errors(self, label_offset):
        import tempfile
        from pathlib import Path
        import nibabel as nib
        from implicit.anatomy.convert import convert, load_bone
        h, r_true, n = 1.5, 14.3, 40
        idx = np.indices((n, n, n)).reshape(3, -1).T * h
        c = np.array([30.2, 29.7, 30.6])
        dist = np.linalg.norm(idx - c, axis=1)
        ct = 50 + 750 * np.clip((r_true - dist) / h + 0.5, 0, 1)  # soft tissue 50, bone 800 HU
        mask = dist <= r_true + label_offset
        aff = np.diag([h, h, h, 1.0])
        dirs = np.random.default_rng(0).normal(size=(400, 3))
        dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
        errs = {}
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            nib.save(nib.Nifti1Image(mask.reshape(n, n, n).astype(np.uint8), aff), str(d / "vertebrae_L3.nii.gz"))
            nib.save(nib.Nifti1Image(ct.reshape(n, n, n).astype(np.int16), aff), str(d / "ct.nii.gz"))
            for use_ct in (False, True):
                out = d / f"out_{use_ct}"
                convert(d, out, ["L3"], use_ct=use_ct, spacing={"vertebra": 0.5}, log=lambda *a: None)
                g, _ = load_bone(out / "L3.npz")
                errs[use_ct] = float(np.abs(g.eval(c + r_true * dirs)).mean())
        return errs

    def test_label_off_by_1mm_is_corrected_by_ct(self):
        for offset in (-1.0, 1.0):  # under- and over-segmented label
            errs = self.surface_errors(offset)
            self.assertGreater(errs[False], 0.7, offset)
            self.assertLess(errs[True], 0.3, offset)

    def test_exact_label_stays_accurate(self):
        errs = self.surface_errors(0.0)
        self.assertLess(errs[False], 0.3)
        self.assertLess(errs[True], 0.3)
