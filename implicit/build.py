"""Build the static viewer site (for GitHub Pages or any static host).

    python -m implicit.build examples/*.py --out site
    python -m implicit.build examples/spine_segment.py --single-file site/spine.html
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from .project import VIEWER_DIR, load_scene, write_scenes


def build_site(scripts, out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    for f in ("index.html", "viewer.js"):
        shutil.copy(VIEWER_DIR / f, out / f)
    (out / ".nojekyll").touch()
    return write_scenes(scripts, out)


def build_single_file(scripts, path):
    """One self-contained HTML file with the scenes and viewer inlined."""
    scenes = {Path(s).stem: load_scene(s).to_json() for s in scripts}
    html = (VIEWER_DIR / "index.html").read_text()
    js = (VIEWER_DIR / "viewer.js").read_text()
    data = json.dumps(scenes).replace("</", "<\\/")
    html = html.replace("<!--SCENES-->", f"<script>window.SCENES = {data};</script>")
    html = html.replace('<script src="viewer.js"></script>', f"<script>\n{js}\n</script>")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(html)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("scripts", nargs="+")
    ap.add_argument("--out", default="site")
    ap.add_argument("--single-file", metavar="HTML")
    a = ap.parse_args(argv)
    if a.single_file:
        build_single_file(a.scripts, a.single_file)
        print(f"wrote {a.single_file}")
    else:
        keys = build_site(a.scripts, a.out)
        print(f"wrote {a.out}/ with scenes: {', '.join(keys)}")


if __name__ == "__main__":
    main()
