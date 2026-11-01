"""Constrained distribution (roadmap 1.7): feasible ranges (implicit.anatomy.ranges) and
postures with no collision and no flagged disc."""

import json
import unittest
from pathlib import Path

import numpy as np

from implicit.anatomy import alignment as al
from implicit.anatomy import clearance as cl
from implicit.anatomy import discs, ranges
from implicit.graph import load_graph

ROOT = Path(__file__).resolve().parents[1]
SUBJECT = ROOT / "assets" / "bones" / "vivaplus-50F"


class TestRanges(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.doc = json.loads((SUBJECT / "landmarks.json").read_text())

    def test_stored_for_every_disc(self):
        data = al.chain(self.doc["landmarks"], self.doc["pelvis"])
        for d in data["discs"]:
            r = d["range"]
            for p, plane in enumerate(ranges.PLANES):
                lo, hi = r[plane]
                half = d["rom"][p] / 2
                self.assertTrue(-half - 1e-3 <= lo <= 0 <= hi <= half + 1e-3, f"{d['label']} {plane} {lo} {hi}")
            self.assertEqual(set(r["why"]), {f"{p}{s}" for p in ranges.PLANES for s in "+-"})
        lo, hi = data["pi_range"]
        self.assertTrue(-ranges.PI_SEARCH <= lo < 0 < hi <= ranges.PI_SEARCH)

    def test_known_bounds(self):
        """The facets bound lumbar extension-side turns and axial rotation; the thickness
        floor bounds T5-T6 and T8-T9 (opened to their floor by the template)."""
        lm = self.doc["landmarks"]
        self.assertTrue(lm["L4"]["range"]["why"]["FE+"].startswith("clearance L4-L5"))
        self.assertTrue(lm["L5"]["range"]["why"]["AR+"].startswith("clearance L5-sacrum"))
        for lv in ("T5", "T8"):
            self.assertTrue(lm[lv]["range"]["why"]["FE-"].startswith("disc thinnest"), lv)

    def test_pi_limited_by_the_sacroiliac_joints(self):
        data = al.chain(self.doc["landmarks"], self.doc["pelvis"])
        res = al.solve(data, **{**{k: data["base"].get(k, 0.0) for k in ("PI", "PT", "LL", "TK", "MT", "TL", "CB")},
                                "PI": 85.0, "ML": 1.0}, light=True)
        self.assertGreater(res["limits"]["short"]["PI"], 5.0)
        res = al.solve(data, **{**{k: data["base"].get(k, 0.0) for k in ("PI", "PT", "LL", "TK", "MT", "TL", "CB")},
                                "PI": 85.0, "ML": 0.0}, light=True)
        self.assertAlmostEqual(res["limits"]["short"]["PI"], 0.0, places=6)


class TestFeasiblePostures(unittest.TestCase):
    """The average spine with the limits on: no pair of neighbouring bones closer than
    min(1 mm, its rest gap) - 0.05 mm, and no disc flagged (posture rules, thickness
    floors), on a grid of slider settings and at random settings of every slider."""

    @classmethod
    def setUpClass(cls):
        cls.scene = load_graph(ROOT / "examples" / "average_spine.json")
        cls.samples = {n: cl.surface_samples(g) for n, (g, _) in cl.scene_bodies(cls.scene).items()}
        cls.defaults = {p.name: p.value() for p in cls.scene.params()}
        cls.rest = {(a, b): c for a, b, c, _ in cl.report(cls.scene, cls.samples)}

    def tearDown(self):
        self.scene.set_params(self.defaults)

    def problems(self, **values):
        self.scene.set_params({**self.defaults, "spp_ML": 1.0, **values})
        hits = {(a, b): c for a, b, c, _ in cl.report(self.scene, self.samples)
                if c < min(ranges.CLEARANCE, self.rest[(a, b)]) - 0.05}
        now, rest = discs.report(self.scene)
        flags = discs.posture_flags(now, rest, discs.report_thickness(self.scene)[0])
        return hits, flags

    def test_grid(self):
        grid = [dict(spp_LL=ll, spp_TK=tk) for ll in (30.0, 80.0) for tk in (10.0, 60.0)]
        grid += [dict(spp_MT=mt, spp_TL=tl, spp_CB=cb) for mt in (-40.0, 40.0) for tl in (-40.0, 40.0)
                 for cb in (-30.0, 30.0)]
        grid += [dict(spp_PT=pt, spp_PI=pi, spp_PO=10.0, spp_LL=70.0, spp_TK=50.0, spp_MT=20.0)
                 for pt in (5.0, 30.0) for pi in (45.0, 70.0)]
        for kw in grid:
            hits, flags = self.problems(**kw)
            self.assertEqual((hits, flags), ({}, {}), kw)

    def test_random(self):
        rng = np.random.default_rng(7)
        for _ in range(12):
            kw = {f"spp_{k}": float(rng.uniform(lo, hi)) for k, (lo, hi) in al.KNOBS.items()
                  if f"spp_{k}" in self.defaults and k != "ML"}
            hits, flags = self.problems(**kw)
            self.assertEqual((hits, flags), ({}, {}), kw)


if __name__ == "__main__":
    unittest.main()
