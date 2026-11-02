"""Intervertebral discs (roadmap 2.1): built between the endplates, conforming to both
in every posture, nucleus fraction, rim areas, wedge angle."""

import json
import shutil
import subprocess
import unittest
from pathlib import Path

import numpy as np

from implicit.anatomy import discs
from implicit.anatomy import intervertebral as iv
from implicit.anatomy.convert import load_bone
from implicit.graph import load_graph

ROOT = Path(__file__).resolve().parents[1]
SUBJECT = ROOT / "assets" / "bones" / "vivaplus-50F"
LABELS = [f"T{i}-T{i + 1}" for i in range(1, 12)] + ["T12-L1"] + [f"L{i}-L{i + 1}" for i in range(1, 5)] + ["L5-S1"]


class TestDiscFiles(unittest.TestCase):
    def test_every_disc_built(self):
        """A disc file for every level, T1-T2 to L5-S1 (no endplate corners needed)."""
        for lab in LABELS:
            g, meta = load_bone(SUBJECT / f"disc_{lab}.npz")
            self.assertEqual(meta["kind"], "disc")
            self.assertGreater(meta["disc_volume_mm3"], 500.0, lab)
            self.assertTrue(all(v >= iv.EXT_MIN for v in meta["extension_mm"].values()), lab)

    def test_lumbar_rim_areas(self):
        """Lumbar endplate rim areas within the women's mean +- 2 SD of Louie et al. 2018;
        areas grow caudally from the upper thoracic to the lumbar spine."""
        mu, sd = iv.LUMBAR_AREA
        areas = {}
        for lab in LABELS:
            a = load_bone(SUBJECT / f"disc_{lab}.npz")[1]["rim_area_mm2"]
            areas[lab] = np.mean(list(a.values()))
            if lab.startswith("L"):
                for k, v in a.items():
                    self.assertTrue(mu - 2 * sd <= v <= mu + 2 * sd, f"{lab} {k} {v}")
        self.assertLess(areas["T3-T4"], areas["T11-T12"])
        self.assertLess(areas["T11-T12"], areas["L4-L5"])


class TestFibres(unittest.TestCase):
    def test_fibre_field(self):
        """Annulus points get a lamella (0 at the wall, 19 at the nucleus) and a unit fibre
        direction normal to the outline's radial direction, inclined to the transverse
        plane by 28-45 deg (outer to inner), alternately up and down; nucleus points -1."""
        g, meta = load_bone(SUBJECT / "disc_L4-L5.npz")
        c, w = np.array(meta["centre"]), np.array(meta["axis"])
        a = np.array(meta["anterior"])
        P = c + np.outer(np.linspace(0, 25, 120), a)  # centre to the anterior wall, mid-height
        P = P[g.eval(P) < 0]
        lam, f = iv.fibres(g, c, w, P, nucleus=0.22)
        self.assertEqual(lam[0], -1)  # the centre: nucleus
        ann = lam >= 0
        np.testing.assert_allclose(np.linalg.norm(f[ann], axis=1), 1.0, atol=1e-9)
        incl = np.degrees(np.arcsin(np.abs(f[ann] @ w)))
        self.assertTrue(np.all((incl > 27.9) & (incl < 45.1)))
        self.assertEqual(lam[ann].max(), iv.LAMELLAE - 1)
        self.assertEqual(lam[ann].min(), 0)
        self.assertGreater(incl[lam[ann].argmax()], incl[lam[ann].argmin()])  # steeper inside
        sign = np.sign(f[ann] @ w)
        self.assertTrue(np.all(sign[lam[ann] % 2 == 0] > 0) and np.all(sign[lam[ann] % 2 == 1] < 0))
        np.testing.assert_allclose(f[ann] @ a, 0.0, atol=0.15)  # circumferential: across the radius


class TestDiscBodies(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.scene = load_graph(ROOT / "examples" / "average_spine.json")
        cls.defaults = {p.name: p.value() for p in cls.scene.params()}

    def tearDown(self):
        self.scene.set_params(self.defaults)

    def interface(self, lab):
        """(disc field, bone field) at each endplate's surface samples inside the
        extended disc (by 0.5 mm), moved with their vertebra, away from the line where the
        nucleus-annulus boundary meets the endplate: the disc is its nucleus and annulus."""
        up, lo = lab.split("-")
        lo = "sacrum" if lo == "S1" else lo
        bodies = discs.scene_bodies(self.scene, grids=True)
        an, nu = self.scene.body(f"{lab} annulus").sdf, self.scene.body(f"{lab} nucleus").sdf
        meta = load_bone(SUBJECT / f"disc_{lab}.npz")[1]
        rest = load_bone(SUBJECT / f"disc_{lab}.npz")[0]  # the extended disc at rest (file frame)
        off = self.scene.frame_offset
        disc, bone = [], []
        for b in (up, lo):
            g, S, _, (R, t) = bodies[b]
            S = np.asarray(S)
            # Endplate side: within the disc's extension, well inside its rim (rest frame).
            sel = rest.eval(S + off) < -0.5
            W = S[sel] @ np.asarray(R).T + np.asarray(t)
            fa, fn = an.eval(W), nu.eval(W)
            clear = np.maximum(fa, fn) > 0.5  # not on the nucleus-annulus boundary (an inner interface)
            disc.append(np.minimum(fa, fn)[clear])
            bone.append(g.eval(S[sel][clear]))
        return np.concatenate(disc), np.concatenate(bone)

    def test_shares_the_bone_surface_in_every_posture(self):
        """The disc is the extended disc cut by the posed bones: at the endplate (inside
        the rim) its field is exactly minus the bone's, so the two share one surface (no
        gap, no overlap) at rest, within the motion limits and without them."""
        postures = [{}, {"spp_LL": 60.0, "spp_TK": 37.0, "spp_MT": 8.0, "spp_PT": 25.0},
                    {"spp_ML": 0.0, "spp_LL": 85.0, "spp_TK": 70.0, "spp_MT": 40.0, "spp_TL": -35.0, "spp_CB": 30.0}]
        for kw in postures:
            self.scene.set_params({**self.defaults, **kw})
            for lab in ("T1-T2", "T8-T9", "L4-L5", "L5-S1"):
                d, b = self.interface(lab)
                self.assertGreater(len(d), 50, lab)  # endplate samples (2 mm apart) under the disc
                np.testing.assert_allclose(d, -b, atol=1e-6, err_msg=f"{lab} {kw}")

    def test_nucleus_fraction(self):
        """The nucleus covers the knob's fraction of the disc's cross-section at mid-height."""
        lab = "L4-L5"
        meta = load_bone(SUBJECT / f"disc_{lab}.npz")[1]
        c, w = np.array(meta["centre"]), np.array(meta["axis"])
        offset = self.scene.frame_offset
        a = np.array(meta["anterior"])
        r = np.cross(a, w)
        s = np.arange(-35, 35, 0.4)
        U, V = np.meshgrid(s, s, indexing="ij")
        X = c - offset + U.reshape(-1, 1) * a + V.reshape(-1, 1) * r
        for frac in (0.22, 0.45):
            self.scene.set_params({**self.defaults, "disc_L4_L5_split_nucleus": frac})
            an = self.scene.body(f"{lab} annulus").sdf.eval(X) < 0
            nu = self.scene.body(f"{lab} nucleus").sdf.eval(X) < 0
            self.assertAlmostEqual(nu.sum() / (an.sum() + nu.sum()), frac, delta=0.02)

    def test_wedge_matches_heights(self):
        """Wedge angle and heights agree: lordotic lumbar discs open anteriorly; the angle
        is close to atan((A - P) / depth)."""
        lm = json.loads((SUBJECT / "landmarks.json").read_text())["landmarks"]
        b = {("sacrum" if k == "S1" else k): ({ep: e[ep]["corners"] for ep in ("sup", "inf")
                                                if ep in e and "corners" in e[ep]}, discs.IDENTITY)
             for k, e in lm.items()}
        w = discs.wedge(b)
        h = {(u, l): x for u, l, *x in discs.heights(b)}
        for k in (("L4", "L5"), ("L2", "L3"), ("T8", "T9")):
            A, _, P = h[k][:3]
            depth = np.linalg.norm(np.subtract(lm[k[1] if k[1] != "sacrum" else "S1"]["sup"]["corners"]["anterior"],
                                               lm[k[1] if k[1] != "sacrum" else "S1"]["sup"]["corners"]["posterior"]))
            self.assertAlmostEqual(w[k], np.degrees(np.arctan((A - P) / depth)), delta=2.0, msg=str(k))
        self.assertGreater(w[("L4", "L5")], 8.0)

    @unittest.skipUnless(shutil.which("node"), "node not installed")
    def test_viewer_wedge_matches(self):
        self.scene.set_params({**self.defaults, "spp_LL": 60.0, "spp_MT": 10.0})
        bodies = discs.scene_bodies(self.scene)
        ref = {f"{u}|{l}": v for (u, l), v in discs.wedge(bodies).items()}
        js_bodies = {k: {"corners": {ep: {cn: np.asarray(v).tolist() for cn, v in e.items()} for ep, e in c.items()}
                         if c else None, "motion": [np.asarray(R).tolist(), np.asarray(t).tolist()]}
                     for k, (c, (R, t)) in bodies.items()}
        js = ("const D = require(process.argv[1]); const b = JSON.parse(require('fs').readFileSync(0));"
              "process.stdout.write(JSON.stringify(D.wedge(b)));")
        got = json.loads(subprocess.run(["node", "-e", js, str(ROOT / "viewer" / "discs.js")], check=True,
                                        capture_output=True, input=json.dumps(js_bodies), text=True).stdout)
        self.assertEqual(set(got), set(ref))
        for k, v in ref.items():
            self.assertAlmostEqual(got[k], v, places=6, msg=k)


if __name__ == "__main__":
    unittest.main()
