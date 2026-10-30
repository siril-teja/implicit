"""Spinopelvic chain model: identity at the measured values, targets met exactly."""

import json
import shutil
import subprocess
import unittest
from pathlib import Path

import numpy as np

from implicit.anatomy import alignment as al

ROOT = Path(__file__).resolve().parents[1]
SUBJECT = ROOT / "assets" / "bones" / "vivaplus-50F"


def moved_landmarks(lm, motions):
    """Landmarks with each segment's endplates moved by its solved motion."""
    out = {}
    for lv, e in lm.items():
        c, s, ty, tz = motions["sacrum" if lv == "S1" else lv]
        f = {}
        for ep in ("sup", "inf"):
            if ep in e:
                p, n = np.array(e[ep]["point"]), np.array(e[ep]["normal"])
                f[ep] = {"point": [p[0], c * p[1] - s * p[2] + ty, s * p[1] + c * p[2] + tz],
                         "normal": [n[0], c * n[1] - s * n[2], s * n[1] + c * n[2]]}
        out[lv] = f
    return out


class TestAlignment(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        d = json.loads((SUBJECT / "landmarks.json").read_text())
        cls.lm, cls.pelvis = d["landmarks"], d["pelvis"]
        cls.data = al.chain(cls.lm, cls.pelvis)

    def test_baseline_is_plausible(self):
        b = self.data["measured"]
        self.assertAlmostEqual(b["PI"], b["PT"] + b["SS"], places=9)
        for k, (lo, hi) in {"PI": (40, 75), "PT": (5, 30), "SS": (25, 55), "LL": (35, 70), "TK": (20, 55)}.items():
            self.assertTrue(lo < b[k] < hi, f"{k} = {b[k]:.1f}")

    def test_identity_at_baseline(self):
        b = self.data["base"]
        res = al.solve(self.data, b["PI"], b["PT"], b["LL"], b["TK"])
        for seg, m in res.items():
            if seg != "curve":
                np.testing.assert_allclose(m, [1, 0, 0, 0], atol=1e-9, err_msg=seg)
        # The curve passes through the S1 endplate centre and the body centres.
        knots = np.reshape(res["curve"], (-1, 3))[::al.CURVE_STEPS]
        np.testing.assert_allclose(knots, [c["point"] for c in self.data["curve"]], atol=1e-9)

    def test_targets_met(self):
        targets = {"PI": 66.0, "PT": 25.0, "LL": 62.0, "TK": 45.0}
        motions = al.solve(self.data, **targets)
        lm = moved_landmarks(self.lm, motions)
        hips = {k: np.array(v) for k, v in self.pelvis.items()}
        c, s, ty, tz = motions["pelvis"]  # hip centres turn with the pelvis about their midpoint: fixed
        H = 0.5 * (hips["hip_left"] + hips["hip_right"])
        np.testing.assert_allclose([c * H[1] - s * H[2] + ty, s * H[1] + c * H[2] + tz], H[1:], atol=1e-9)
        got = al.measure_chain({**self.data, "s1_point": lm["S1"]["sup"]["point"][1:],
                                "s1_normal": lm["S1"]["sup"]["normal"][1:]}, lm)
        off = {k: self.data["measured"][k] - self.data["base"][k] for k in self.data["base"]}
        for k, v in targets.items():  # relative to the rounded starting values
            self.assertAlmostEqual(got[k], v + off[k], places=6, msg=k)
        self.assertAlmostEqual(got["SS"], got["PI"] - got["PT"], places=9)

    def test_untouched_levels_keep_their_disc_angles(self):
        """Discs outside the LL and TK ranges keep their angle (T12-L1, T1-T4)."""
        motions = al.solve(self.data, 60.0, 20.0, 40.0, 40.0)
        lm = moved_landmarks(self.lm, motions)
        for upper, lower in (("T12", "L1"), ("T2", "T3"), ("T3", "T4")):
            before = al.tilt(self.lm[lower]["sup"]["normal"]) - al.tilt(-np.array(self.lm[upper]["inf"]["normal"]))
            after = al.tilt(lm[lower]["sup"]["normal"]) - al.tilt(-np.array(lm[upper]["inf"]["normal"]))
            self.assertAlmostEqual(before, after, places=9)


    @unittest.skipUnless(shutil.which("node"), "node not installed")
    def test_viewer_solver_matches(self):
        """viewer/solvers.js computes the same motions as the Python model."""
        targets = {"PI": 66.0, "PT": 25.0, "LL": 62.0, "TK": 45.0}
        js = ("const S = require(process.argv[1]); const d = JSON.parse(require('fs').readFileSync(0));"
              "process.stdout.write(JSON.stringify(S.KINDS.spinopelvic(d.data, d.targets)));")
        out = subprocess.run(["node", "-e", js, str(ROOT / "viewer" / "solvers.js")], check=True, capture_output=True,
                             input=json.dumps({"data": self.data, "targets": targets}), text=True).stdout
        got, ref = json.loads(out), al.solve(self.data, **targets)
        self.assertEqual(set(got), set(ref))
        for k in ref:
            np.testing.assert_allclose(got[k], ref[k], atol=1e-9, err_msg=k)


if __name__ == "__main__":
    unittest.main()
