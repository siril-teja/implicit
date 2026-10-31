"""Template disc space heights (implicit.anatomy.template) on VIVA+ 50F."""

import json
import unittest
from pathlib import Path

import numpy as np

from implicit.anatomy import alignment as al
from implicit.anatomy import template
from implicit.anatomy.convert import load_bone, load_bones

ROOT = Path(__file__).resolve().parents[1]
SUBJECT = ROOT / "assets" / "bones" / "vivaplus-50F"


class TestTemplate(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        d = json.loads((SUBJECT / "landmarks.json").read_text())
        cls.lm, cls.pelvis = d["landmarks"], d["pelvis"]
        cls.record = json.loads((SUBJECT / "template.json").read_text())

    def test_applied_once(self):
        """A second run finds nothing to do: no shift, no joint to carve."""
        grids = {n: g for n, (g, _) in load_bones(SUBJECT).items()}
        for u, l, d, *_ in template.disc_shifts(self.lm, grids):
            self.assertLess(abs(d), template.TOL, f"{u}-{l}")
        self.assertEqual(template.carve_joints(dict(grids), log=lambda *a: None), {})

    def test_joint_spaces(self):
        """Every pair of neighbouring bones at least its joint space apart (2.0 mm lumbar,
        1.3 mm thoracic): on the surface of each, the two distances add up to at least
        the space (to the 0.1 mm the values are stored to)."""
        grids = {n: g for n, (g, _) in load_bones(SUBJECT).items()}
        self.assertEqual(set(self.record["joints"]), {"T3-T4", "T12-L1", "L1-L2", "L2-L3", "L3-L4", "L4-L5", "L5-sacrum"})
        for k in self.record["joints"]:
            u, l = k.split("-")
            for a, b in ((u, l), (l, u)):
                f = grids[a].values
                g = grids[b].eval(template._points(grids[a])).reshape(f.shape)
                near = np.abs(f) < 0.5 * grids[a].spacing
                self.assertGreaterEqual(float((f + g)[near].min()), template.joint_space(u, l) - 0.1, k)

    def test_translations_only(self):
        """Orientations untouched: the subject's alignment is the unmodified VIVA+ 50F's
        measured before the correction; the endplate planes are fitted again on the
        shifted grids, which moves the angles by up to 0.05 deg)."""
        m = al.chain(self.lm, self.pelvis)["measured"]
        for k, v in {"PI": 58.701, "PT": 17.216, "SS": 41.485, "LL": 51.925, "TK": 30.262, "MT": 0.044,
                     "TL": -0.26}.items():
            self.assertAlmostEqual(m[k], v, delta=0.1, msg=k)

    def test_bone_files_record_their_shift(self):
        for lv, t in self.record["vertebrae_mm"].items():
            meta = load_bone(SUBJECT / f"{lv}.npz")[1]
            if np.linalg.norm(t) >= template.TOL:
                np.testing.assert_allclose(meta["template_shift_mm"], t, atol=1e-3)
        self.assertNotIn("template_shift_mm", load_bone(SUBJECT / "sacrum.npz")[1])

    def test_rib_cage_follows(self):
        """Ribs carry their vertebra's total template shift; the sternum and cartilages are
        carried by the rib ends; the costovertebral joints are open."""
        total = template.recorded_shifts(SUBJECT)
        for n in (1, 6, 12):
            for side in ("left", "right"):
                meta = load_bone(SUBJECT / f"rib{n}_{side}.npz")[1]
                np.testing.assert_allclose(meta["template_shift_mm"], total[f"T{n}"], atol=1e-3)
        for part in ("sternum", "costal_cartilage_left", "costal_cartilage_right"):
            self.assertIn("template_carried", load_bone(SUBJECT / f"{part}.npz")[1])
        grids = {n: g for n, (g, _) in load_bones(SUBJECT).items()}
        for v, rib in template.rib_joints(set(grids)):
            f = grids[rib].values
            g = grids[v].eval(template._points(grids[rib])).reshape(f.shape)
            near = np.abs(f) < 0.5 * grids[rib].spacing
            self.assertGreaterEqual(float((f + g)[near].min()), template.joint_space(v, rib) - 0.1, f"{v}-{rib}")

    def test_vertebra_shifts_add_up(self):
        ax = np.array([0.0, 0.0, 1.0])
        shifts = [("L5", "S1", -1.0, ax, None, ""), ("L4", "L5", 2.0, ax, None, ""), ("L3", "L4", 0.5, ax, None, "")]
        out = template.vertebra_shifts(shifts, {"L3", "L4", "L5", "L2"})
        np.testing.assert_allclose([out[k][2] for k in ("L5", "L4", "L3", "L2")], [-1.0, 1.0, 1.5, 1.5])


if __name__ == "__main__":
    unittest.main()
