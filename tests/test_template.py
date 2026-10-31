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
        """A second run finds nothing to do."""
        grids = {n: g for n, (g, _) in load_bones(SUBJECT).items()}
        shifts, _ = template.disc_shifts(self.lm, grids, self.record["clearance_floor_mm"])
        for u, l, d, *_ in shifts:
            self.assertLess(abs(d), template.TOL, f"{u}-{l}")

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

    def test_vertebra_shifts_add_up(self):
        ax = np.array([0.0, 0.0, 1.0])
        shifts = [("L5", "S1", -1.0, ax, None, ""), ("L4", "L5", 2.0, ax, None, ""), ("L3", "L4", 0.5, ax, None, "")]
        out = template.vertebra_shifts(shifts, {"L3", "L4", "L5", "L2"})
        np.testing.assert_allclose([out[k][2] for k in ("L5", "L4", "L3", "L2")], [-1.0, 1.0, 1.5, 1.5])


if __name__ == "__main__":
    unittest.main()
