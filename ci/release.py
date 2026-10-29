"""Release helpers used by the CI workflows (stdlib only).

    python ci/release.py version              print the package version
    python ci/release.py notes 0.2.0          print that version's CHANGELOG section
    python ci/release.py check origin/main    fail unless the version is bumped vs the
                                              base ref and has a dated CHANGELOG entry
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
VERSION_FILE = "implicit/__init__.py"
VERSION_RE = re.compile(r'^__version__ = "(\d+\.\d+\.\d+)"', re.M)
SECTION_RE = r"^## \[{v}\] - (\d{{4}}-\d{{2}}-\d{{2}})\s*$"


def parse(v):
    return tuple(int(x) for x in v.split("."))


def version(text=None):
    text = text if text is not None else (ROOT / VERSION_FILE).read_text()
    m = VERSION_RE.search(text)
    if not m:
        raise SystemExit(f"no __version__ = \"X.Y.Z\" in {VERSION_FILE}")
    return m.group(1)


def notes(v, changelog=None):
    """Body of the ``## [v] - YYYY-MM-DD`` section of CHANGELOG.md."""
    text = changelog if changelog is not None else (ROOT / "CHANGELOG.md").read_text()
    m = re.search(SECTION_RE.format(v=re.escape(v)), text, re.M)
    if not m:
        raise SystemExit(f"CHANGELOG.md has no dated section '## [{v}] - YYYY-MM-DD'")
    rest = text[m.end():]
    nxt = re.search(r"^## \[", rest, re.M)
    body = (rest[:nxt.start()] if nxt else rest).strip()
    if not body:
        raise SystemExit(f"CHANGELOG.md section for {v} is empty")
    return body


def check(base_ref):
    v = version()
    base = subprocess.run(["git", "show", f"{base_ref}:{VERSION_FILE}"],
                          capture_output=True, text=True, cwd=ROOT)
    if base.returncode == 0:
        try:
            old = version(base.stdout)
        except SystemExit:
            old = "0.0.0"
    else:  # file did not exist on the base branch yet
        old = "0.0.0"
    if parse(v) <= parse(old):
        raise SystemExit(f"version {v} must be greater than {old} on {base_ref}: "
                         f"bump __version__ in {VERSION_FILE} (or label the PR 'no-release')")
    notes(v)
    print(f"ok: {old} -> {v}, CHANGELOG entry present")


def main(argv):
    if not argv:
        raise SystemExit(__doc__)
    cmd, args = argv[0], argv[1:]
    if cmd == "version":
        print(version())
    elif cmd == "notes":
        print(notes(args[0]))
    elif cmd == "check":
        check(args[0])
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main(sys.argv[1:])
