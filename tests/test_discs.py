"""Disc space heights (implicit.anatomy.discs, viewer/discs.js)."""

import json
import shutil
import subprocess
import unittest
from pathlib import Path

import numpy as np

from implicit.anatomy import discs
from implicit.anatomy.convert import load_bone
from implicit.anatomy.landmarks import Field
from implicit.graph import load_graph

ROOT = Path(__file__).resolve().parents[1]
SUBJECT = ROOT / "assets" / "bones" / "vivaplus-50F"


def gap(upper, lower, lm, v=0.0):
    """Bone-to-bone gap along the lower endplate normal, v mm anterior of its centre."""
    e = lm[lower]
    p, n = np.array(e["sup"]["point"]), np.array(e["sup"]["normal"])
    a = np.array(e["R"])[:, 1]
    a = (a - a.dot(n) * n) / np.linalg.norm(a - a.dot(n) * n)
    s = np.arange(-6, 25, 0.02)
    P = p + v * a + s[:, None] * n
    f = {k: Field(*(lambda g: (g.values, g.origin, g.spacing))(load_bone(SUBJECT / f"{k}.npz")[0]))
         for k in (upper, lower)}
    return s[np.flatnonzero(f[upper](P) < 0).min()] - s[np.flatnonzero(f[lower](P) < 0).max()]


class TestDiscs(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.lm = json.loads((SUBJECT / "landmarks.json").read_text())["landmarks"]
        cls.scene = load_graph(ROOT / "examples" / "average_spine.json")
        cls.defaults = {p.name: p.value() for p in cls.scene.params()}

    def tearDown(self):
        self.scene.set_params(self.defaults)

    def heights(self, **values):
        self.scene.set_params({**self.defaults, **values})
        now, rest = discs.report(self.scene)
        return {(u, l): h for u, l, *h in now}, rest

    def test_corners_stored(self):
        """Every endplate has corners, except T1's (plane fit rms above RMS_MAX)."""
        for lv, e in self.lm.items():
            for ep in ("sup", "inf"):
                if ep in e:
                    self.assertEqual("corners" in e[ep], e[ep]["rms"] <= discs.RMS_MAX, f"{lv} {ep}")

    def test_middle_height_is_the_bone_gap(self):
        h, _ = self.heights()
        for up, lo in (("L1", "L2"), ("L3", "L4"), ("T10", "T11")):
            self.assertAlmostEqual(h[(up, lo)][1], gap(up, lo, self.lm), delta=1.0)

    def test_rest_is_the_default_posture(self):
        h, rest = self.heights()
        for u, l, *h0 in rest:
            np.testing.assert_allclose(h[(u, l)], h0, atol=1e-6)
        self.assertEqual(discs.posture_flags([(u, l, *x) for (u, l), x in h.items()], rest), {})

    def test_hyperkyphosis_compresses_upper_thoracic_discs(self):
        h, rest = self.heights(spp_TK=60)
        flags = discs.posture_flags([(u, l, *x) for (u, l), x in h.items()], rest)
        self.assertEqual(set(flags), {("T4", "T5"), ("T5", "T6")})
        self.assertTrue(all(w.startswith("anterior -") for v in flags.values() for w in v))

    def test_template_upper_lumbar_discs_outside_norms(self):
        """VIVA+ 50F: L1-L2 and L2-L3 are about twice the population middle height."""
        _, rest = self.heights()
        norms = discs.norm_flags(rest)
        for k in (("L1", "L2"), ("L2", "L3")):
            self.assertTrue(any(w.startswith("M ") for w in norms[k]), norms.get(k))

    @unittest.skipUnless(shutil.which("node"), "node not installed")
    def test_viewer_matches(self):
        """viewer/discs.js: same heights, flags and constants."""
        self.scene.set_params({**self.defaults, "spp_LL": 75, "spp_TK": 55, "spp_MT": 25})
        bodies = discs.scene_bodies(self.scene)
        js_bodies = {k: {"corners": {ep: {c: np.asarray(v).tolist() for c, v in e.items()} for ep, e in c.items()} if c else None,
                         "motion": [np.asarray(R).tolist(), np.asarray(t).tolist()]} for k, (c, (R, t)) in bodies.items()}
        js = ("const D = require(process.argv[1]); const b = JSON.parse(require('fs').readFileSync(0));"
              "const now = D.heights(b), rest = D.restHeights(b);"
              "process.stdout.write(JSON.stringify({now, rest, flags: D.postureFlags(now, rest), norms: D.normFlags(rest),"
              "NORMS: D.NORMS, c: [D.MEAN_CHANGE, D.LOCAL_STRAIN, D.MIN_REST]}));")
        out = json.loads(subprocess.run(["node", "-e", js, str(ROOT / "viewer" / "discs.js")], check=True,
                                        capture_output=True, text=True, input=json.dumps(js_bodies)).stdout)
        now, rest = discs.heights(bodies), discs.rest_heights(bodies)
        for ref, got in ((now, out["now"]), (rest, out["rest"])):
            self.assertEqual([list(r[:2]) for r in ref], [g[:2] for g in got])
            np.testing.assert_allclose([r[2:] for r in ref], [g[2:] for g in got], atol=1e-9)
        self.assertEqual(set(out["flags"]), {f"{u}|{l}" for u, l in discs.posture_flags(now, rest)})
        self.assertEqual(set(out["norms"]), {f"{u}|{l}" for u, l in discs.norm_flags(rest)})
        self.assertEqual({k: [list(x) for x in v] for k, v in discs.NORMS.items()}, out["NORMS"])
        self.assertEqual([discs.MEAN_CHANGE, discs.LOCAL_STRAIN, discs.MIN_REST], out["c"])


if __name__ == "__main__":
    unittest.main()
