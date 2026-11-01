"""Global balance (roadmap 1.8): SV, TPA, compensation modes, Roussouly presets."""

import json
import math
import unittest
from pathlib import Path

import numpy as np

from implicit.anatomy import alignment as al

ROOT = Path(__file__).resolve().parents[1]
SUBJECT = ROOT / "assets" / "bones" / "vivaplus-50F"


def motion(m):
    m = np.asarray(m, float)
    return m[:9].reshape(3, 3).T, m[9:]


class TestBalance(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        d = json.loads((SUBJECT / "landmarks.json").read_text())
        cls.lm = d["landmarks"]
        cls.data = al.chain(cls.lm, d["pelvis"], ribcage=d.get("ribcage"))
        cls.b = cls.data["base"]

    def targets(self, **kw):
        return {**{k: self.b.get(k, 0.0) for k in ("PI", "PT", "LL", "TK", "MT", "TL", "CB")}, **kw}

    def solve(self, **kw):
        return al.solve(self.data, **self.targets(**kw), light=True)

    def measured(self, res):
        """SV and TPA from the moved landmarks, independently of the solver's report."""
        R, t = motion(res["T1"])
        t1 = R @ self.data["frames"]["T1"]["center"] + t
        R, t = motion(res["sacrum"])
        post = R @ self.data["s1_posterior"] + t
        mid = R @ self.data["s1_point"] + t
        H = self.data["hip"]
        tpa = math.degrees(math.atan2(t1[1] - H[1], t1[2] - H[2]) - math.atan2(mid[1] - H[1], mid[2] - H[2]))
        return t1[1] - post[1], tpa

    def test_rest_values(self):
        """VIVA+ 50F stands with T1 16 mm anterior of the S1 posterior corner and a TPA of
        14 deg (asymptomatic adults: TPA mostly below 14-20 deg, Protopsaltis 2014)."""
        self.assertAlmostEqual(self.b["SV"], 15.9, delta=0.1)
        self.assertTrue(5.0 < self.b["TPA"] < 20.0)
        res = self.solve()
        sv, tpa = self.measured(res)
        self.assertAlmostEqual(res["limits"]["balance"]["SV"], sv, places=6)
        self.assertAlmostEqual(res["limits"]["balance"]["TPA"], tpa, places=6)

    def test_compensation_meets_the_target(self):
        """With a compensation mode T1 is held at the SV target within 1 mm; pelvic
        retroversion leaves TPA as it is (it measures the deformity independently of
        pelvic retroversion, Protopsaltis 2014), thoracic flattening lowers it."""
        free = self.solve(LL=35.0)
        sv0, tpa0 = self.measured(free)
        self.assertGreater(sv0, self.b["SV"] + 50.0)  # lordosis lost: the trunk falls forward
        for cm, var in ((1.0, "PT"), (2.0, "TK")):
            res = self.solve(LL=35.0, CM=cm, SV=self.b["SV"])
            sv, tpa = self.measured(res)
            self.assertAlmostEqual(sv, self.b["SV"], delta=1.0, msg=var)
            self.assertIn(var, res["limits"]["compensation"])
            if var == "PT":
                self.assertGreater(res["limits"]["compensation"]["PT"], self.b["PT"] + 10.0)
                self.assertAlmostEqual(tpa, tpa0, delta=0.5)
            else:
                self.assertLess(res["limits"]["compensation"]["TK"], self.b["TK"] - 10.0)
                self.assertLess(tpa, tpa0)
        res = self.solve(SV=self.b["SV"] + 30.0, CM=1.0)
        self.assertAlmostEqual(self.measured(res)[0], self.b["SV"] + 30.0, delta=1.0)

    def test_compensation_within_the_limits(self):
        """With the limits on thoracic flattening cannot reach the target (TK can change
        about 7 deg): the shortfall is reported."""
        res = self.solve(LL=35.0, CM=2.0, SV=self.b["SV"], ML=1.0)
        short = res["limits"]["short"]["SV"]
        self.assertLess(short, -20.0)
        self.assertAlmostEqual(self.measured(res)[0], self.b["SV"] - short, delta=1e-6)

    def test_presets(self):
        """Roussouly presets: PI and PT of each type (Laouissat et al. 2017), LL from PI
        (Le Huec & Hasegawa 2016), limits off; each met exactly with the limits off."""
        p = al.ROUSSOULY
        self.assertEqual([p[k]["PI"] for k in sorted(p)], [39.0, 41.0, 53.0, 48.0, 62.0])
        self.assertEqual(p["Roussouly type 4"]["LL"], 61.0)  # 0.54 * 62 + 27.6 = 61.1
        for name, v in p.items():
            res = self.solve(**v)
            self.assertTrue(all(abs(x) < 1e-6 for x in res["limits"]["short"].values()), name)
        self.assertEqual(self.data["presets"], p)


if __name__ == "__main__":
    unittest.main()
