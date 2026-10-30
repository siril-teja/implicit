"""VIVA+ keyword reading and inside/outside test, on a small synthetic model."""

import tempfile
import unittest
from pathlib import Path

import numpy as np

from implicit.anatomy import vivaplus as vp

CORNERS = [(0, 0, 0), (10, 0, 0), (10, 10, 0), (0, 10, 0), (0, 0, 10), (10, 0, 10), (10, 10, 10), (0, 10, 10)]


def write_model(d):
    """One hex element (a 10 mm cube, part 7) and one shell triangle (part 8)."""
    (d / "common").mkdir()
    (d / "50F-standing").mkdir()
    i8 = lambda *v: "".join(f"{x:8d}" for x in v)
    (d / "common" / "parts.k").write_text("\n".join([
        "*KEYWORD", "*PART", "$ title", "XX-Test-Cube-Cortical", f"{7:10d}{1:10d}{1:10d}",
        "*PART", "XX-Test-Plate", f"{8:10d}{1:10d}{1:10d}",
        "*ELEMENT_SOLID", "$ eid pid n1..n8", i8(1, 7, *range(1, 9)),
        "*ELEMENT_SHELL_THICKNESS", i8(2, 8, 1, 2, 3, 3), "      1.0     1.0     1.0     1.0",
        "*END", ""]))
    (d / "50F-standing" / "nodes.k").write_text("\n".join(
        ["*NODE"] + [f"{i + 1:8d}" + "".join(f"{c:16.4f}" for c in p) for i, p in enumerate(CORNERS)]
        + ["*END", ""]))


class TestVivaplus(unittest.TestCase):
    def test_read_and_mask(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            write_model(d)
            self.assertEqual(vp.read_parts(d), {7: "XX-Test-Cube-Cortical", 8: "XX-Test-Plate"})
            shells, solids = vp.read_elements(d, {7, 8})
            self.assertEqual(solids, {7: [list(range(1, 9))]})
            self.assertEqual(shells, {8: [[1, 2, 3, 3]]})  # thickness card skipped
            nodes = vp.read_nodes(d, "50F-standing")
            self.assertEqual(nodes[7], (10.0, 10.0, 10.0))
        tris = vp.surface({}, solids)
        self.assertEqual(len(tris), 12)  # boundary of one hex
        V = np.array([[nodes[i] for i in t] for t in tris])
        mask, lo = vp.inside_mask(V, 0.5)
        self.assertAlmostEqual(mask.sum() * 0.5 ** 3, 1000.0, delta=1.0)
        centre = lo + 0.5 * np.argwhere(mask).mean(0)
        np.testing.assert_allclose(centre, [5, 5, 5], atol=0.3)

    def test_frame(self):
        # VIVA+: x forward, y left, z up -> x right, y anterior, z up.
        np.testing.assert_allclose(vp.to_ras([[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
                                   [[0, 1, 0], [-1, 0, 0], [0, 0, 1]])


if __name__ == "__main__":
    unittest.main()
