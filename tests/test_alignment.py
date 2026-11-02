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
        cls.data = al.chain(cls.lm, cls.pelvis, ribcage=d["ribcage"])
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
            if seg not in ("tps", "tps_ribcage", "chain", "grid", "flow_cells", "flow_segs", "limits"):
                np.testing.assert_allclose(m, [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0], atol=1e-9, err_msg=seg)
        P = np.random.default_rng(0).uniform(-150, 150, (200, 3))
        np.testing.assert_allclose(al.tps_eval(res["tps"], P), P, atol=1e-6)
        np.testing.assert_allclose(al.poly_inverse(res["chain"], P), P, atol=1e-9)
        self.assertLess(max(res["grid"][3::4]), 1e-6)
        self.assertEqual(max(res["flow_cells"][1::4]), 0)  # nothing moves: no flow lines

    def test_polyrigid_rigid_in_bodies_and_invertible_near_the_axis(self):
        data = {**self.data, "field": "polyrigid"}
        res = al.solve(data, **self.targets(PT=25.0, LL=62.0, TK=45.0, MT=35.0, TL=-25.0, AR=0.3, PO=5.0, CB=8.0))
        # Points inside a vertebral body (between its endplate planes) move with it.
        for lv in ("L3", "T8"):
            c = np.array(self.data["frames"][lv]["center"])
            P = c + np.random.default_rng(1).uniform(-4, 4, (20, 3))
            R, t = motion(res[lv])
            np.testing.assert_allclose(al.poly_forward(res["chain"], P), P @ R.T + t, atol=1e-9, err_msg=lv)
        # Disc spaces: continuous between the two vertebrae; the inverse finds every point
        # within 15 mm of the spine curve.
        C = np.reshape(self.data["curve"], (-1, 3))
        rng = np.random.default_rng(2)
        P0 = C[rng.integers(0, len(C), 800)] + rng.uniform(-10, 10, (800, 3))
        Pw = al.poly_forward(res["chain"], P0)
        np.testing.assert_allclose(al.poly_inverse(res["chain"], Pw), P0, atol=1e-3)
        # Its display grid is the forward displacement.
        G = al.grid_points(data["grid"])
        D = np.asarray(res["grid"]).reshape(-1, 4)
        np.testing.assert_allclose(D[::97, :3], al.poly_forward(res["chain"], G[::97]) - G[::97], atol=1e-9)

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

    def _disc_turns(self, motions):
        """Angle (deg) of each disc's relative rotation from rest: lower^T upper."""
        out = {}
        below = "sacrum"
        for d in self.data["discs"]:
            Rl, Ru = motion(motions[below])[0], motion(motions[d["upper"]])[0]
            c = np.clip((np.trace(Rl.T @ Ru) - 1) / 2, -1, 1)
            out[d["label"]] = np.degrees(np.arccos(c))
            below = d["upper"]
        return out

    def test_motion_limits_respected(self):
        """With the limits on no disc turns beyond its range (half the total range of
        Wilke et al. 2025 each way, or less: its feasible range); sagittal-only and
        coronal-only turns stay within the plane's own half range, any setting within
        the combined one."""
        rom = {d["label"]: np.array(d["rom"]) / 2 for d in self.data["discs"]}
        for kw, plane in ((dict(LL=90.0, TK=0.0), 0), (dict(LL=0.0, TK=80.0), 0),
                          (dict(MT=80.0, TL=-80.0, CB=40.0, CP=0.0), 1)):
            turns = self._disc_turns(al.solve(self.data, **self.targets(**kw, ML=1.0), light=True))
            for lab, a in turns.items():
                self.assertLessEqual(a, rom[lab][plane] + 1e-6, f"{kw} {lab}")
        rng = np.random.default_rng(3)
        for _ in range(20):
            kw = {k: rng.uniform(lo, hi) for k, (lo, hi) in al.KNOBS.items()}
            kw["ML"] = 1.0
            turns = self._disc_turns(al.solve(self.data, **kw, light=True))
            for lab, a in turns.items():
                self.assertLessEqual(a, np.linalg.norm(rom[lab]) + 1e-6, f"{lab}")

    def test_shortfall_reported(self):
        """A target beyond the levels' ranges is met up to the reported shortfall; the
        levels it needs are all at their limit (half the range of motion, or the
        feasible range where that is smaller: implicit.anatomy.ranges)."""
        t = self.targets(LL=80.0, TK=50.0, ML=1.0)
        res = al.solve(self.data, **t, light=True)
        got, short = self.measured(res), res["limits"]["short"]
        D = self.data["discs"]
        pos, neg = al._caps(D, 0, True, 1), al._caps(D, 0, True, -1)  # rad
        cap_ll = np.degrees(sum(c for d, c in zip(D, pos) if d["upper"] in al.LORDOSIS))
        cap_tk = np.degrees(sum(c for d, c in zip(D, neg) if d["upper"] in al.KYPHOSIS))  # TK up: turns -
        self.assertAlmostEqual(short["LL"], 80.0 - self.b["LL"] - cap_ll, places=6)
        self.assertAlmostEqual(short["TK"], 50.0 - self.b["TK"] - cap_tk, places=6)
        for k in ("LL", "TK"):
            self.assertAlmostEqual(got[k], t[k] - short[k], places=6, msg=k)
        at = {lab for lab, p, _ in res["limits"]["at_limit"] if p == "FE"}
        self.assertEqual(at, {d["label"] for d in self.data["discs"] if d["upper"] in al.LORDOSIS + al.KYPHOSIS})
        # Within the ranges: met exactly, nothing short.
        res = al.solve(self.data, **self.targets(LL=56.0, TK=33.0, ML=1.0), light=True)
        self.assertTrue(all(abs(v) < 1e-9 for v in res["limits"]["short"].values()))
        got = self.measured(res)
        self.assertAlmostEqual(got["LL"], 56.0, places=6)
        self.assertAlmostEqual(got["TK"], 33.0, places=6)

    def test_feasible_ranges_respected(self):
        """Every disc turn within its feasible range (implicit.anatomy.ranges) in each
        plane alone; the reason at a limit is the range's."""
        rng = {d["label"]: d["range"] for d in self.data["discs"] if "range" in d}
        self.assertEqual(len(rng), len(self.data["discs"]))
        for kw in (dict(LL=90.0), dict(LL=0.0), dict(TK=80.0), dict(TK=0.0)):
            res = al.solve(self.data, **self.targets(**kw, ML=1.0), light=True)
            below = "sacrum"
            for d in self.data["discs"]:
                Rl, Ru = motion(res[below])[0], motion(res[d["upper"]])[0]
                a = np.degrees(np.arctan2((Rl.T @ Ru)[2, 1], (Rl.T @ Ru)[2, 2]))  # turn about x
                lo, hi = rng[d["label"]]["FE"]
                self.assertTrue(lo - 1e-6 <= a <= hi + 1e-6, f"{kw} {d['label']} {a:.3f} not in [{lo}, {hi}]")
                below = d["upper"]
        res = al.solve(self.data, **self.targets(LL=90.0, ML=1.0), light=True)
        why = {(lab, p): w for lab, p, w in res["limits"]["at_limit"]}
        self.assertEqual(why[("L4-L5", "FE")], rng["L4-L5"]["why"]["FE+"])  # clearance L4-L5

    def test_coupled_rotation_follows_the_bend(self):
        """Thoracic lateral bending carries ipsilateral axial rotation, 0.40 deg per deg
        (Fujimori et al. 2014); none at CP 0."""
        kw = self.targets(MT=-20.0)  # convex left: the thoracic discs bend the spine above to the right
        R0 = motion(al.solve(self.data, **kw, CP=0.0, light=True)["T5"])[0]
        R1 = motion(al.solve(self.data, **kw, CP=1.0, light=True)["T5"])[0]
        # At T5 (upper end vertebra) the bend of all MT discs (20 deg) has accumulated.
        D = R1 @ R0.T  # the extra turn, about T5's own axis
        turn = np.degrees(np.arccos(np.clip((np.trace(D) - 1) / 2, -1, 1))) * np.sign(D[0, 1])  # + : anterior turns right
        self.assertAlmostEqual(turn, 20.0 * 6.2 / 15.6, delta=0.3)

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

    def test_flow_lines_follow_the_seeds(self):
        """Each line starts at its seed and ends where the field takes it; the cell
        lookup finds the nearest line near the lines and a lower bound elsewhere."""
        data = {**self.data, "field": "polyrigid"}
        t = self.targets(PT=25.0, LL=62.0, MT=35.0, TL=-25.0, AR=0.3)
        res = al.solve(data, **t)
        f = data["flow"]
        S = np.asarray(f["seeds"])
        end = al._forward(data, al._motions(data, **t))(S)
        moved = np.linalg.norm(end - S, axis=1) >= 0.5
        r = 0.4
        d = al.flow_eval(res["flow_cells"], res["flow_segs"], f, np.concatenate([S[moved], end[moved]]), r)
        np.testing.assert_allclose(d[:, 0], -r, atol=1e-9)  # on the lines
        np.testing.assert_allclose(d[len(S[moved]):, 1], np.linalg.norm(end - S, axis=1)[moved], atol=1e-9)
        # Elsewhere never more than the true distance to the nearest line.
        E = np.asarray(res["flow_segs"], float).reshape(-1, 2, 4)
        E = E[np.any(E[:, :, :3] != 0, axis=(1, 2))]
        P = S[::5] + np.random.default_rng(3).uniform(-20, 20, (len(S[::5]), 3))
        a, b = E[:, 0, :3], E[:, 1, :3]
        true = np.array([np.min(np.linalg.norm(p - a - (b - a) * np.clip(np.sum((p - a) * (b - a), 1)
                                                                           / np.maximum(np.sum((b - a) ** 2, 1), 1e-12), 0, 1)[:, None], axis=1))
                         for p in P]) - r
        got = al.flow_eval(res["flow_cells"], res["flow_segs"], f, P, r)[:, 0]
        self.assertTrue(np.all(got <= true + 1e-9))
        np.testing.assert_allclose(got[got < al.FLOW_MARGIN], true[got < al.FLOW_MARGIN], atol=1e-9)

    @unittest.skipUnless(shutil.which("node"), "node not installed")
    def test_viewer_solver_matches(self):
        """viewer/solvers.js computes the same motions and TPS as the Python model."""
        targets = self.targets(PI=66.0, PT=25.0, LL=62.0, TK=45.0, MT=35.0, TL=-25.0, AR=0.3, PO=5.0, CB=8.0)
        for field in ("tps", "polyrigid"):
            self._compare_js({**self.data, "field": field}, targets)
        # Extreme curves: the coronal balance saturates (soft limit, branch ends).
        self._compare_js(self.data, self.targets(MT=71.0, TL=77.0, CB=46.0, TK=14.0, AR=0.3))
        self._compare_js(self.data, self.targets(MT=-85.0, TL=-75.0, CB=-9.0))
        # Motion limits and coupling: saturated levels, shortfalls.
        self._compare_js(self.data, {**targets, "ML": 1.0, "CP": 1.0})
        self._compare_js(self.data, self.targets(LL=85.0, TK=10.0, MT=-20.0, TL=30.0, CB=-30.0, AR=0.6, ML=1.0, CP=0.5))
        # Global balance: compensation by the pelvis and by the thoracic spine.
        self._compare_js(self.data, self.targets(LL=35.0, CM=1.0, SV=20.0, ML=1.0, CP=1.0))
        self._compare_js(self.data, self.targets(LL=35.0, CM=2.0, SV=20.0))

    def _positions(self, targets):
        m = al.solve(self.data, **targets, light=True)
        pts = {lv: self.data["frames"][lv]["center"] for lv in self.data["levels"]}
        pts["sacrum"] = self.data["s1_point"]
        out = []
        for k, p in pts.items():
            R, t = motion(m[k])
            out.append(R @ np.asarray(p) + t)
        return np.array(out)

    def _assert_smooth(self, path, label):
        P = np.array([self._positions(v) for v in path])
        step = np.linalg.norm(np.diff(P, axis=0), axis=2).max(1)  # largest vertebra move per step
        moving = step[step > 1e-6]  # (limits: nothing moves while a target is beyond the ranges)
        ref = np.median(moving) if len(moving) else 0.0
        if step.max() < 4.0 * max(ref, 1e-3) + 1e-6:
            return
        # A large step can be a steeper stretch (e.g. once a limit stops the balance
        # from compensating): continuous if it splits into proportionally smaller steps.
        i = int(step.argmax())
        a, b = path[i], path[i + 1]
        sub = [{k: a[k] + s * (b[k] - a[k]) for k in a} for s in np.linspace(0, 1, 17)]
        fine = np.linalg.norm(np.diff(np.array([self._positions(v) for v in sub]), axis=0), axis=2).max(1)
        self.assertLess(fine.max(), 4.0 * step.max() / 16 + 1e-6, f"{label}: jump of {fine.max():.1f} mm")

    def test_sliders_smooth(self):
        """Every knob over its whole range, and straight paths through random settings of
        all knobs, with the motion limits off and on: no vertebra jumps (the largest move
        per step stays within 4x the median of the steps that move, or splits into
        proportionally smaller steps; a jump was 400 mm in a 0.45 deg step of TL before the balance fix)."""
        for ml in (0.0, 1.0):
            for k, (lo, hi) in al.KNOBS.items():
                if k in ("AR", "ML", "CM"):
                    continue  # AR moves nothing without curves; ML and CM are switches
                self._assert_smooth([self.targets(**{"ML": ml, "CP": 1.0, k: x}) for x in np.linspace(lo, hi, 81)],
                                    f"{k} (ML {ml:.0f})")
        rng = np.random.default_rng(1)
        for i in range(8):
            a, b = ({k: rng.uniform(lo, hi) for k, (lo, hi) in al.KNOBS.items()} for _ in range(2))
            a["ML"] = b["ML"] = float(i % 2)  # limits off and on (held along the path)
            a["CM"] = b["CM"] = float(i % 3)  # each compensation mode (held)
            self._assert_smooth([{k: a[k] + s * (b[k] - a[k]) for k in a} for s in np.linspace(0, 1, 41)], f"path {i}")

    def _compare_js(self, data, targets):
        js = ("const S = require(process.argv[1]); const d = JSON.parse(require('fs').readFileSync(0));"
              "process.stdout.write(JSON.stringify(S.KINDS.spinopelvic(d.data, d.targets)));")
        out = subprocess.run(["node", "-e", js, str(ROOT / "viewer" / "solvers.js")], check=True, capture_output=True,
                             input=json.dumps({"data": data, "targets": targets}), text=True).stdout
        got, ref = json.loads(out), al.solve(data, **targets)
        self.assertEqual(set(got), set(ref))
        lim, ref_lim = got.pop("limits"), ref.pop("limits")
        self.assertEqual(lim["at_limit"], ref_lim["at_limit"])
        self.assertEqual(set(lim["short"]), set(ref_lim["short"]))
        for k, v in ref_lim["short"].items():
            self.assertAlmostEqual(lim["short"][k], v, delta=1e-6, msg=k)
        for k in ("SV", "TPA", "CB"):
            self.assertAlmostEqual(lim["balance"][k], ref_lim["balance"][k], delta=1e-6, msg=k)
        self.assertEqual(set(lim.get("compensation", {})), set(ref_lim.get("compensation", {})))
        for k, v in ref_lim.get("compensation", {}).items():
            self.assertAlmostEqual(lim["compensation"][k], v, delta=1e-6, msg=k)
        if "ribcage" in ref_lim:
            rc, ref_rc = lim["ribcage"], ref_lim["ribcage"]
            for k in ("sternum_mm", "sternum_deg"):
                self.assertAlmostEqual(rc[k], ref_rc[k], delta=1e-6, msg=k)
            self.assertEqual(set(rc["cartilage_mm"]), set(ref_rc["cartilage_mm"]))
            for k, v in ref_rc["cartilage_mm"].items():
                self.assertAlmostEqual(rc["cartilage_mm"][k], v, delta=1e-6, msg=k)
        for k in ref:
            np.testing.assert_allclose(got[k], ref[k], atol=1e-6, err_msg=f"{data['field']} {k}")


if __name__ == "__main__":
    unittest.main()
