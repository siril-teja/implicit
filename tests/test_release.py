import importlib.util
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("release", ROOT / "ci" / "release.py")
release = importlib.util.module_from_spec(spec)
spec.loader.exec_module(release)

CHANGELOG = """# Changelog

## [0.2.0] - 2026-11-05

### Added
- thing two

## [0.1.0] - 2026-10-29

- thing one
"""


class Release(unittest.TestCase):
    def test_version_matches_package(self):
        import implicit
        self.assertEqual(release.version(), implicit.__version__)

    def test_notes_extracts_one_section(self):
        self.assertEqual(release.notes("0.2.0", CHANGELOG), "### Added\n- thing two")
        self.assertEqual(release.notes("0.1.0", CHANGELOG), "- thing one")

    def test_notes_require_dated_section(self):
        with self.assertRaises(SystemExit):
            release.notes("0.3.0", CHANGELOG)
        with self.assertRaises(SystemExit):
            release.notes("0.3.0", "## [0.3.0] - unreleased\n- x\n")

    def test_current_version_has_release_notes(self):
        self.assertTrue(release.notes(release.version()))

    def test_semver_ordering(self):
        self.assertLess(release.parse("0.9.0"), release.parse("0.10.0"))
        self.assertLess(release.parse("0.1.9"), release.parse("0.2.0"))
