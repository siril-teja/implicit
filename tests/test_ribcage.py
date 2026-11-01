"""Connected rib cage (implicit.anatomy.ribcage): sternum fit, junctions in motion."""

import json
import shutil
import subprocess
import unittest
from pathlib import Path

import numpy as np

from implicit.anatomy import alignment as al
from implicit.anatomy import ribcage
from implicit.anatomy.convert import load_bone

ROOT = Path(__file__).resolve().parents[1]
SUBJECT = ROOT / "assets" / "bones" / "vivaplus-50F"


def motion(m):
    m = np.asarray(m, float)
    return m[:9].reshape(3, 3).T, m[9:]


def surface(g):
    return g.origin + g.spacing * np.argwhere(np.abs(g.values) < 0.5 * g.spacing)


class TestHorn(unittest.TestCase):
    def test_recovers_a_rigid_motion(self):
        rng = np.random.default_rng(0)
        for _ in range(20):
            P = rng.uniform(-100, 100, (14, 3))
            w = rng.normal(size=3)
            w *= rng.uniform(0, 1.5) / np.linalg.norm(w)
            K = np.array([[0, -w[2], w[1]], [w[2], 0, -w[0]], [-w[1], w[0], 0]])
            a = np.linalg.norm(w)
            R = np.eye(3) + np.sin(a) / a * K + (1 - np.cos(a)) / a ** 2 * K @ K
            t = rng.uniform(-50, 50, 3)
            Rg, tg = ribcage.horn(P.tolist(), (P @ R.T + t).tolist())
            np.testing.assert_allclose(Rg, R, atol=1e-9)
            np.testing.assert_allclose(tg, t, atol=1e-7)

    def test_least_squares_matches_kabsch(self):
        rng = np.random.default_rng(1)
        P = rng.uniform(-100, 100, (14, 3))
        Q = P + rng.normal(scale=5, size=P.shape)
        cp, cq = P.mean(0), Q.mean(0)
        U, _, Vt = np.linalg.svd((P - cp).T @ (Q - cq))
        D = np.diag([1, 1, np.sign(np.linalg.det(Vt.T @ U.T))])
        R = Vt.T @ D @ U.T
        Rg, tg = ribcage.horn(P.tolist(), Q.tolist())
        np.testing.assert_allclose(Rg, R, atol=1e-9)
        np.testing.assert_allclose(tg, cq - R @ cp, atol=1e-7)

    @unittest.skipUnless(shutil.which("node"), "node not installed")
    def test_viewer_matches(self):
        rng = np.random.default_rng(2)
        P = rng.uniform(-100, 100, (14, 3))
        Q = P @ np.array([[0.9, -0.3, 0.1], [0.3, 0.95, 0.0], [-0.1, 0.05, 1.0]]).T + 3
        js = ("const S = require(process.argv[1]); const c = JSON.parse(require('fs').readFileSync(0));"
              "process.stdout.write(JSON.stringify(S.horn(c[0], c[1])));")
        out = subprocess.run(["node", "-e", js, str(ROOT / "viewer" / "solvers.js")], check=True,
                             capture_output=True, input=json.dumps([P.tolist(), Q.tolist()]), text=True).stdout
        R, t = json.loads(out)
        Rp, tp = ribcage.horn(P.tolist(), Q.tolist())
        np.testing.assert_allclose(R, Rp, atol=1e-12)
        np.testing.assert_allclose(t, tp, atol=1e-9)


class TestConnected(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        d = json.loads((SUBJECT / "landmarks.json").read_text())
        cls.rc = d["ribcage"]
        cls.data = al.chain(d["landmarks"], d["pelvis"], ribcage=cls.rc)
        cls.b = cls.data["base"]
        cls.cc = {s: load_bone(SUBJECT / f"costal_cartilage_{s}.npz")[0] for s in ("left", "right")}
        st = load_bone(SUBJECT / "sternum.npz")[0]
        cls.junction = {}
        for s in ("left", "right"):
            for n in ribcage.RIBS:
                P = surface(load_bone(SUBJECT / f"rib{n}_{s}.npz")[0])
                cls.junction[(n, s)] = P[np.abs(cls.cc[s].eval(P)) < ribcage.CONTACT]
        cls.sternal = {s: surface(cls.cc[s])[np.abs(st.eval(surface(cls.cc[s]))) < ribcage.CONTACT]
                       for s in ("left", "right")}

    def targets(self, **kw):
        return {**{k: self.b.get(k, 0.0) for k in ("PI", "PT", "LL", "TK", "MT", "TL", "CB")}, "AR": 0.3, "PO": 0.0,
                "ML": 1.0, "CP": 1.0, **kw}

    def junction_change(self, res):
        """Largest change (mm) of the cartilage field at the junction samples of the ribs
        and the sternum, each moved with its bone, through the rib cage field."""
        tab, worst = res["tps_ribcage"], 0.0
        for (n, s), P in self.junction.items():
            R, t = motion(res[f"T{n}"])
            f = self.cc[s].eval(al.tps_eval(tab, P @ R.T + t))
            worst = max(worst, float(np.abs(f - self.cc[s].eval(P)).max()))
        R, t = motion(res["sternum"])
        for s, P in self.sternal.items():
            f = self.cc[s].eval(al.tps_eval(tab, P @ R.T + t))
            worst = max(worst, float(np.abs(f - self.cc[s].eval(P)).max()))
        return worst

    def test_every_junction_found(self):
        self.assertEqual(set(self.rc["junctions"]), {f"rib{n}_{s}" for n in ribcage.RIBS for s in ("left", "right")})
        self.assertEqual(len(self.data["ribcage"]["tips"]), 2 * ribcage.STERNAL_RIBS)

    def test_identity_at_rest(self):
        res = al.solve(self.data, **self.targets())
        np.testing.assert_allclose(res["sternum"], [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0], atol=1e-9)
        P = np.array([q["point"] for q in self.data["ribcage"]["points"]])
        np.testing.assert_allclose(al.tps_eval(res["tps_ribcage"], P), P, atol=1e-6)
        self.assertLess(self.junction_change(res), 1e-6)

    def test_junctions_stay_joined(self):
        """Within the motion limits, on a grid of slider settings, no junction opens or
        overlaps by more than 0.5 mm."""
        grid = [dict(TK=t, LL=l) for t in (25.0, 37.0) for l in (30.0, 72.0)]
        grid += [dict(MT=m, TL=tl, CB=cb) for m in (-10.0, 10.0) for tl in (-15.0, 15.0) for cb in (-20.0, 20.0)]
        grid += [dict(PT=30.0, PI=70.0, PO=10.0, TK=35.0, MT=8.0)]
        for kw in grid:
            res = al.solve(self.data, **self.targets(**kw))
            self.assertLess(self.junction_change(res), 0.5, kw)

    def test_sternum_follows_the_ribs(self):
        """Flexing the thoracic spine moves the sternum with the true rib ends (a close
        rigid fit); the report gives its motion and the cartilage length changes."""
        res = al.solve(self.data, **self.targets(TK=37.0), light=True)
        rep = res["limits"]["ribcage"]
        tips = np.array([q["point"] for q in self.data["ribcage"]["tips"]])
        R, t = motion(res["sternum"])
        moved = []
        for q in self.data["ribcage"]["tips"]:
            Rr, tr = motion(res[q["segment"]])
            moved.append(Rr @ q["point"] + tr)
        fit = np.linalg.norm(tips @ R.T + t - np.array(moved), axis=1)
        self.assertLess(fit.mean(), 2.0)  # the true rib ends move nearly as one body here
        self.assertGreater(rep["sternum_mm"], 1.0)
        self.assertEqual(set(rep["cartilage_mm"]), {f"rib{n}_{s}" for n in range(1, 8) for s in ("left", "right")})


if __name__ == "__main__":
    unittest.main()
