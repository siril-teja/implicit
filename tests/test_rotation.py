"""Physiological rotation centres of the discs (implicit.anatomy.rotation)."""

import json
import unittest
from pathlib import Path

import numpy as np

from implicit.anatomy import alignment as al
from implicit.anatomy import rotation as rc

ROOT = Path(__file__).resolve().parents[1]
SUBJECT = ROOT / "assets" / "bones" / "vivaplus-50F"


class TestRotationCentres(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        d = json.loads((SUBJECT / "landmarks.json").read_text())
        cls.lm, cls.pelvis = d["landmarks"], d["pelvis"]

    def local(self, upper, lower):
        """Stored centres in the lower vertebra's endplate frame: (anterior, cranial) offsets."""
        lo = self.lm[lower]
        p, n = np.array(lo["sup"]["point"]), np.array(lo["sup"]["normal"])
        a = np.array(lo["R"])[:, 1]
        a = (a - a.dot(n) * n) / np.linalg.norm(a - a.dot(n) * n)
        r = self.lm[upper]["rotation"]
        return {k: ((np.array(r[k]) - p) @ a, (np.array(r[k]) - p) @ n) for k in ("sagittal", "coronal")}

    def test_stored_for_every_disc(self):
        levels = [lv for lv in al.VERTEBRAE if lv in self.lm]
        self.assertTrue(all("rotation" in self.lm[lv] for lv in levels))

    def test_lumbar_sagittal(self):
        """Pearcy & Bogduk: just below the lower superior endplate, posterior third."""
        for up, lo in (("L1", "L2"), ("L3", "L4"), ("L5", "S1")):
            ant, cra = self.local(up, lo)["sagittal"]
            depth = self.lm["L5" if lo == "S1" else lo]["depth"]
            self.assertAlmostEqual(cra, -rc.BELOW_ENDPLATE, places=3)
            self.assertAlmostEqual(ant, -depth / 6, places=3)

    def test_thoracic(self):
        """Qiu et al.: below the lower superior endplate on the antero-posterior centre."""
        for up, lo in (("T4", "T5"), ("T10", "T11"), ("T12", "L1")):
            ant, cra = self.local(up, lo)["sagittal"]
            self.assertAlmostEqual(ant, 0.0, places=3)
            self.assertAlmostEqual(cra, -self.lm[lo]["height"] / 4, delta=0.01)
            self.assertLess(self.local(up, lo)["coronal"][1], 0.0)  # below the endplate, above the body centre

    def test_targets_still_met(self):
        """Cobb angles depend only on the turns: the moved centres keep every target."""
        data = al.chain(self.lm, self.pelvis)
        b = data["base"]
        t = {**{k: b.get(k, 0.0) for k in ("PI", "PT", "LL", "TK", "MT", "TL", "CB")}, "AR": 0.0, "PO": 0.0}
        t.update(LL=70.0, TK=45.0, PT=25.0)  # sagittal only: coronal turns couple at second order
        m = al.solve(data, **t, light=True)
        R = lambda lv: np.array(m[lv][:9]).reshape(3, 3).T
        tilt = lambda v: np.degrees(np.arctan2(v[1], v[2]))
        up = lambda lv, ep: (1 if ep == "sup" else -1) * np.array(self.lm[lv][ep]["normal"])
        ll = tilt(R("sacrum") @ self.lm["S1"]["sup"]["normal"]) - tilt(R("L1") @ up("L1", "sup"))
        tk = tilt(R("T4") @ up("T4", "sup")) - tilt(R("T12") @ up("T12", "inf"))
        off = {k: data["measured"][k] - b[k] for k in ("LL", "TK")}
        self.assertAlmostEqual(ll - off["LL"], 70.0, places=6)
        self.assertAlmostEqual(tk - off["TK"], 45.0, places=6)


if __name__ == "__main__":
    unittest.main()
