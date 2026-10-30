"""Spine alignment chain model: identity at the measured values, targets met, TPS."""

import json
import shutil
import subprocess
import unittest
from pathlib import Path

import numpy as np

from implicit.anatomy import alignment as al

ROOT = Path(__file__).resolve().parents[1]
SUBJECT = ROOT / "assets" / "bones" / "vivaplus-50F"


def motion(m):
    """(R, t) from a solver motion [R columns (9), t (3)]."""
    m = np.asarray(m, float)
    return m[:9].reshape(3, 3).T, m[9:]


def moved_landmarks(lm, motions):
    """Landmarks with each segment's endplates and body centre moved by its motion."""
    out = {}
    for lv, e in lm.items():
        R, t = motion(motions["sacrum" if lv == "S1" else lv])
        f = {"body_center": (R @ e["body_center"] + t).tolist()}
        for ep in ("sup", "inf"):
            if ep in e:
                f[ep] = {"point": (R @ e[ep]["point"] + t).tolist(), "normal": (R @ e[ep]["normal"]).tolist()}
        out[lv] = f
    return out


class TestAlignment(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        d = json.loads((SUBJECT / "landmarks.json").read_text())
        cls.lm, cls.pelvis = d["landmarks"], d["pelvis"]
        cls.data = al.chain(cls.lm, cls.pelvis)
        cls.b = cls.data["base"]

    def targets(self, **kw):
        return {**{k: self.b.get(k, 0.0) for k in ("PI", "PT", "LL", "TK", "MT", "TL", "CB")}, "AR": 0.0, "PO": 0.0,
                **kw}

    def measured(self, motions):
        lm = moved_landmarks(self.lm, motions)
        data = {**self.data, "s1_point": lm["S1"]["sup"]["point"], "s1_normal": lm["S1"]["sup"]["normal"]}
        got = al.measure_chain(data, lm)
        off = {k: self.data["measured"][k] - self.b[k] for k in self.b}
        return {k: v - off.get(k, 0.0) for k, v in got.items()}  # relative to the rounded starting values

    def test_baseline_is_plausible(self):
        b = self.data["measured"]
        self.assertAlmostEqual(b["PI"], b["PT"] + b["SS"], places=9)
        for k, (lo, hi) in {"PI": (40, 75), "PT": (5, 30), "SS": (25, 55), "LL": (35, 70), "TK": (20, 55),
                            "MT": (-5, 5), "TL": (-5, 5), "CB": (-5, 5)}.items():
            self.assertTrue(lo < b[k] < hi, f"{k} = {b[k]:.1f}")

    def test_identity_at_baseline(self):
        res = al.solve(self.data, **self.targets(AR=0.4))
        for seg, m in res.items():
            if seg != "tps":
                np.testing.assert_allclose(m, [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0], atol=1e-9, err_msg=seg)
        P = np.random.default_rng(0).uniform(-150, 150, (200, 3))
        np.testing.assert_allclose(al.tps_eval(res["tps"], P), P, atol=1e-6)

    def test_sagittal_targets_met(self):
        t = self.targets(PI=66.0, PT=25.0, LL=62.0, TK=45.0)
        got = self.measured(al.solve(self.data, **t))
        for k in ("PI", "PT", "LL", "TK"):
            self.assertAlmostEqual(got[k], t[k], places=6, msg=k)
        self.assertAlmostEqual(got["SS"], got["PI"] - got["PT"], places=9)

    def test_coronal_targets_met(self):
        """Cobb angles and balance hit exactly with the sagittal values unchanged."""
        t = self.targets(MT=40.0, TL=-30.0, CB=12.0, AR=0.3)
        motions = al.solve(self.data, **t)
        got = self.measured(motions)
        for k in ("MT", "TL"):
            self.assertAlmostEqual(got[k], t[k], delta=0.3, msg=k)  # axial turns tilt the endplates slightly
        self.assertAlmostEqual(got["CB"], t["CB"], places=6)
        for k in ("PI", "PT", "LL", "TK"):  # coronal tilt steepens sagittal projections (second order)
            self.assertAlmostEqual(got[k], t[k], delta=2.0, msg=k)
        # Without axial rotation the Cobb angles are exact.
        got = self.measured(al.solve(self.data, **{**t, "AR": 0.0}))
        for k in ("MT", "TL"):
            self.assertAlmostEqual(got[k], t[k], places=6, msg=k)

    def test_apex_is_most_lateral_and_turned_toward_convexity(self):
        motions = al.solve(self.data, **self.targets(MT=40.0, TL=-40.0, AR=0.3))
        lm = moved_landmarks(self.lm, motions)
        c = {lv: np.array(lm[lv]["body_center"])[[0, 2]] for lv in self.data["levels"]}  # coronal (x, z)
        def apex(u, lo):  # farthest from the chord between the end vertebrae (signed: + right)
            d = (c[u] - c[lo]) / np.linalg.norm(c[u] - c[lo])
            levels = self.data["levels"][self.data["levels"].index(lo):self.data["levels"].index(u) + 1]
            dev = {lv: d[1] * (c[lv] - c[lo])[0] - d[0] * (c[lv] - c[lo])[1] for lv in levels}
            return max(dev, key=lambda lv: abs(dev[lv])), dev
        lv, dev = apex("T5", "T12")
        self.assertEqual(lv, "T8")
        self.assertGreater(dev[lv], 0)  # convex right
        lv, dev = apex("T12", "L4")  # one disc above the apex, three below: within a level
        self.assertIn(lv, ("T12", "L1", "L2"))
        self.assertLess(dev[lv], 0)
        R, _ = motion(motions["T8"])
        self.assertAlmostEqual(R[0, 1], np.sin(np.radians(0.3 * 40)), delta=0.01)  # anterior turns right (convex side)

    def test_pelvic_obliquity_turns_legs_and_pelvis(self):
        motions = al.solve(self.data, **self.targets(PO=10.0))
        R, _ = motion(motions["pelvis"])
        self.assertAlmostEqual(np.degrees(np.arctan2(R[2, 0], R[0, 0])), 10.0, places=6)  # right side up
        np.testing.assert_allclose(motions["legs"], motions["pelvis"], atol=1e-12)

    def test_tps_moves_control_points_with_their_segments(self):
        motions = al.solve(self.data, **self.targets(PT=25.0, LL=62.0, MT=35.0, TL=-25.0, AR=0.3, PO=5.0))
        for q in self.data["tps_points"][::7]:
            R, t = motion(motions[q["segment"]])
            y = R @ q["point"] + t
            np.testing.assert_allclose(al.tps_eval(motions["tps"], y[None])[0], q["point"], atol=1e-6)

    @unittest.skipUnless(shutil.which("node"), "node not installed")
    def test_viewer_solver_matches(self):
        """viewer/solvers.js computes the same motions and TPS as the Python model."""
        targets = self.targets(PI=66.0, PT=25.0, LL=62.0, TK=45.0, MT=35.0, TL=-25.0, AR=0.3, PO=5.0, CB=8.0)
        js = ("const S = require(process.argv[1]); const d = JSON.parse(require('fs').readFileSync(0));"
              "process.stdout.write(JSON.stringify(S.KINDS.spinopelvic(d.data, d.targets)));")
        out = subprocess.run(["node", "-e", js, str(ROOT / "viewer" / "solvers.js")], check=True, capture_output=True,
                             input=json.dumps({"data": self.data, "targets": targets}), text=True).stdout
        got, ref = json.loads(out), al.solve(self.data, **targets)
        self.assertEqual(set(got), set(ref))
        for k in ref:
            np.testing.assert_allclose(got[k], ref[k], atol=1e-7, err_msg=k)


if __name__ == "__main__":
    unittest.main()
