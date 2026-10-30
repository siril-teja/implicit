import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from implicit.graph import BLOCKS, GraphError, build_graph, load_graph, manifest, read_graph, save_graph
from implicit.sdf import Box, Sphere

ROOT = Path(__file__).resolve().parent.parent
BONES = ROOT / "assets" / "bones" / "s0751"


def graph(nodes, **kw):
    return {"format": "implicit.graph", "version": 1, "name": "t", "nodes": nodes, **kw}


def out(src, name="body"):
    return {"block": "output", "in": {"body": src, "name": name}}


class TestBlocks(unittest.TestCase):
    def test_manifest(self):
        m = json.loads(json.dumps(manifest()))
        names = [b["name"] for b in m["blocks"]]
        self.assertEqual(len(names), len(set(names)))
        types = {"body", "bone", "scalar", "vec3", "number", "text", "color", "bool", "choice", "path"}
        for b in m["blocks"]:
            self.assertTrue(b["doc"], b["name"])
            for p in b["inputs"] + b["outputs"]:
                self.assertIn(p["type"], types, (b["name"], p["name"]))
            self.assertTrue(callable(BLOCKS[b["name"]].build))

    def test_shapes_match_direct_construction(self):
        sc = build_graph(graph({
            "c": {"block": "vector", "in": {"x": 5, "y": 0, "z": 0}},
            "s": {"block": "sphere", "in": {"radius": 8, "center": "@c"}},
            "b": {"block": "box", "in": {"size": [10, 20, 30]}},
            "u": {"block": "union", "in": {"a": "@s", "b": "@b"}},
            "o": out("@u"),
        }))
        p = np.random.default_rng(0).uniform(-25, 25, (2000, 3))
        ref = np.minimum(Sphere(8).translate((5, 0, 0)).eval(p), Box((5, 10, 15)).eval(p))
        np.testing.assert_allclose(sc.body("body").sdf.eval(p), ref, atol=1e-9)
        names = {q.name for q in sc.params()}
        self.assertTrue({"c_x", "s_radius", "b_size_y", "u_blend"} <= names)
        self.assertIn("vec3", sc.glsl())

    def test_literals_are_live_parameters(self):
        sc = build_graph(graph({"s": {"block": "sphere", "in": {"radius": 5}}, "o": out("@s")}))
        p = np.array([[9.0, 0, 0]])
        self.assertAlmostEqual(sc.body("body").sdf.eval(p)[0], 4.0)
        sc.set_params({"s_radius": 7})
        self.assertAlmostEqual(sc.body("body").sdf.eval(p)[0], 2.0)

    def test_number_drives_several_inputs(self):
        sc = build_graph(graph({
            "r": {"block": "number", "in": {"value": 4, "min": 1, "max": 10}},
            "a": {"block": "sphere", "in": {"radius": "@r"}},
            "b": {"block": "sphere", "in": {"radius": "@r", "center": [20, 0, 0]}},
            "oa": out("@a", "a"), "ob": out("@b", "b"),
        }))
        sc.set_params({"r": 6})
        self.assertAlmostEqual(sc.body("a").sdf.eval(np.array([[10.0, 0, 0]]))[0], 4.0)
        self.assertAlmostEqual(sc.body("b").sdf.eval(np.array([[10.0, 0, 0]]))[0], 4.0)
        self.assertNotIn("a_radius", {q.name for q in sc.params()})

    def test_bounds_hold_over_slider_ranges(self):
        """Body bounds must contain the body for every slider value (the shader skips outside)."""
        sc = build_graph(graph({
            "b": {"block": "box", "in": {"size": [10, 4, 20], "center": [3, 0, 0]}},
            "m": {"block": "translate", "in": {"body": "@b", "offset": [0, 0, 5]}},
            "r": {"block": "rotate", "in": {"body": "@m", "axis": "y", "angle": 10, "center": [0, 0, 2]}},
            "g": {"block": "offset", "in": {"body": "@r", "distance": 1}},
            "o": out("@g"),
        }))
        body = sc.body("body")
        lo, hi = (np.array(x) for x in body.bounds)
        rng = np.random.default_rng(1)
        params = sc.params()
        for trial in range(30):
            sc.set_params({q.name: rng.uniform(q.lo, q.hi) for q in params})
            pts = rng.uniform(lo - 150, hi + 150, (20000, 3))
            inside = pts[body.sdf.eval(pts) < 0]
            self.assertTrue(np.all(inside >= lo - 1e-6) and np.all(inside <= hi + 1e-6), trial)

    def test_errors(self):
        cases = {
            "cycle": {"a": {"block": "offset", "in": {"body": "@b"}},
                      "b": {"block": "offset", "in": {"body": "@a"}}, "o": out("@a")},
            "unknown block": {"a": {"block": "nope"}, "o": out("@a")},
            "unknown node": {"o": out("@ghost")},
            "type": {"s": {"block": "sphere"}, "m": {"block": "smooth", "in": {"bone": "@s"}}, "o": out("@m")},
            "missing wire": {"o": {"block": "output", "in": {"name": "x"}}},
            "choice": {"s": {"block": "sphere"}, "r": {"block": "rotate", "in": {"body": "@s", "axis": "w"}},
                       "o": out("@r")},
            "bad port": {"s": {"block": "sphere", "in": {"radius": 1, "colour": 2}}, "o": out("@s")},
            "no output": {"s": {"block": "sphere"}},
            "bad id": {"not-an-id": {"block": "sphere"}, "o": out("@s")},
        }
        for what, nodes in cases.items():
            with self.subTest(what), self.assertRaises(GraphError):
                build_graph(graph(nodes))

    def test_save_and_read_round_trip(self):
        g = graph({"s": {"block": "sphere", "in": {"radius": 3}, "pos": [10, 20]}, "o": out("@s")})
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "g.json"
            save_graph(g, path)
            self.assertEqual(read_graph(path), g)
            self.assertEqual(len(path.read_text().splitlines()), 4)  # header, one line per node, close


class TestBrowserBuild(unittest.TestCase):
    def test_templates_for_every_body_block(self):
        from implicit.graph import BROWSER_BUILT, browser_manifest
        m = browser_manifest()
        self.assertIn("gridSDF", m["glsl_lib"])
        for b in m["blocks"]:
            if b["name"] in BROWSER_BUILT:
                continue
            self.assertIn("glsl", b, b["name"])
            n_choice = [len(p["choices"]) for p in b["inputs"] if p["type"] == "choice"]
            self.assertEqual(len(b["glsl"]), int(np.prod(n_choice)) if n_choice else 1, b["name"])
            for tmpl in b["glsl"].values():
                self.assertTrue(tmpl["out"].startswith("$"))
                for op in tmpl["ops"]:
                    self.assertIn(op[0], ("var", "sub"))
        self.assertEqual(set(next(b for b in m["blocks"] if b["name"] == "rotate")["glsl"]),
                         {"axis=x", "axis=y", "axis=z"})

    def test_positions_slide_around_their_value(self):
        sc = build_graph(graph({"s": {"block": "sphere", "in": {"center": [120, 0, -7]}}, "o": out("@s")}))
        p = {q.name: q for q in sc.params()}
        self.assertEqual((p["s_center_x"].lo, p["s_center_x"].hi), (70.0, 170.0))
        self.assertEqual((p["s_center_z"].lo, p["s_center_z"].hi), (-57.0, 43.0))

    def test_current_bounds_are_tight_and_hold(self):
        sc = load_graph(ROOT / "tests" / "graphs" / "all_blocks.json")
        rng = np.random.default_rng(3)
        for b in sc.bodies:
            lo, hi = (np.array(x) for x in b.bounds_now())
            rlo, rhi = (np.array(x) for x in b.bounds)
            self.assertTrue(np.all(rlo <= lo) and np.all(hi <= rhi))
            self.assertLess(np.max(hi - lo), 120)
            pts = rng.uniform(rlo.clip(-200), rhi.clip(max=200), (200000, 3))
            inside = pts[b.sdf.eval(pts) < 0]
            self.assertGreater(len(inside), 0, b.name)
            self.assertTrue(np.all(inside >= lo - 1e-6) and np.all(inside <= hi + 1e-6), b.name)

    def test_library_shares_one_atlas(self):
        from implicit.graph import Library
        with tempfile.TemporaryDirectory() as d:
            sub = Path(d) / "bones"
            sub.mkdir()
            for n in ("L3", "L4", "spinal_cord"):
                (sub / f"{n}.npz").write_bytes((BONES / f"{n}.npz").read_bytes())
            g = graph({"b": {"block": "bone", "in": {"file": "bones/L3.npz"}},
                       "s": {"block": "smooth", "in": {"bone": "@b", "sigma": 1}}, "o": out("@s")})
            lib = Library(g, d)
            j = lib.to_json("x.atlas.gz")
            self.assertEqual(set(j["bones"]), {"bones/L3.npz", "bones/L4.npz", "bones/spinal_cord.npz"})
            for b in j["bones"].values():
                self.assertEqual(b["smooth"]["ops"][1], ["sub", "bone", "$p"])
            offsets = {k: lib.ctx.grid(k).atlas_offset for k in j["bones"]}
            sc = build_graph(g, d, library=lib)
            sc.glsl()
            self.assertIs(sc.packed_atlas, lib.atlas)
            self.assertEqual({k: lib.ctx.grid(k).atlas_offset for k in j["bones"]}, offsets)  # not repacked
            self.assertEqual(sc.to_json("x.atlas.gz")["atlas"]["size"], j["atlas"]["size"])


class TestAnatomyGraph(unittest.TestCase):
    def test_example_graph_matches_bone_files(self):
        """examples/anatomy.json: one bone -> smooth -> output chain per bone file, all files present."""
        g = read_graph(ROOT / "examples" / "anatomy.json")
        files = {n["in"]["file"] for n in g["nodes"].values() if n["block"] == "bone"}
        self.assertEqual({Path(f).stem for f in files}, {p.stem for p in BONES.glob("*.npz")})
        for f in files:
            self.assertTrue((ROOT / "examples" / f).exists(), f)

    def test_bone_smoothing_equals_direct_construction(self):
        from implicit.anatomy.convert import load_bone
        from implicit.sdf import smooth
        from implicit.anatomy.spine import anatomy_graph

        with tempfile.TemporaryDirectory() as d:
            sub = Path(d) / "bones"
            sub.mkdir()
            for n in ("L3", "spinal_cord"):
                (sub / f"{n}.npz").write_bytes((BONES / f"{n}.npz").read_bytes())
            g = anatomy_graph(sub, relative_to=d)
            path = Path(d) / "g.json"
            save_graph(g, path)
            sc = load_graph(path)
        self.assertEqual([b.name for b in sc.bodies], ["L3", "spinal_cord"])
        self.assertFalse(sc.body("spinal_cord").visible)
        # Frame: centred on the bones that are not references (here L3 only).
        g3, _ = load_bone(BONES / "L3.npz")
        lo, hi = g3.bounds()
        np.testing.assert_allclose(sc.frame_offset, np.round((lo - 0.5 + hi + 0.5) / 2))
        g3.origin = g3.origin - sc.frame_offset
        rng = np.random.default_rng(2)
        p = rng.uniform(lo - sc.frame_offset - 3, hi - sc.frame_offset + 3, (4000, 3))
        for sigma, keep in ((0.0, 1), (1.5, 1), (3.0, 0)):
            sc.set_params({"L3_smooth_sigma": sigma, "smooth_preserve": keep})
            np.testing.assert_allclose(sc.body("L3").sdf.eval(p), smooth(g3, sigma, keep).eval(p), atol=1e-6)


if __name__ == "__main__":
    unittest.main()
