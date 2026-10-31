"""Build the static viewer site (for GitHub Pages or any static host).

    python -m implicit.build examples/*.json examples/*.py --out site
    python -m implicit.build examples/spine_segment.py --single-file site/spine.html

``--commit`` and ``--repo`` (set by the release workflow) stamp the build so the
viewer's About section can link to its release notes, licences and source.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from . import __version__
from .project import VIEWER_DIR, load_scene, write_scenes


def app_info(commit=None, repo=None):
    return {"version": __version__, "commit": commit, "repo": repo}


def _stamp(html, info):
    return html.replace("<!--APP-->", f"<script>window.APP = {json.dumps(info)};</script>")


def build_site(scripts, out, commit=None, repo=None):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    html = (VIEWER_DIR / "index.html").read_text()
    (out / "index.html").write_text(_stamp(html, app_info(commit, repo)))
    for f in ("gpu.js", "solvers.js", "graph.js", "clearance.js", "discs.js", "editor.js", "viewer.js", "diagnostics.html", "diagnostics.js"):
        shutil.copy(VIEWER_DIR / f, out / f)
    (out / ".nojekyll").touch()
    return write_scenes(scripts, out)


def build_single_file(scripts, path, commit=None, repo=None):
    """One self-contained HTML file with the scenes and viewer inlined."""
    scenes = {Path(s).stem: load_scene(s).to_json() for s in scripts}  # graphs: view only
    html = _stamp((VIEWER_DIR / "index.html").read_text(), app_info(commit, repo))
    js = "\n".join((VIEWER_DIR / f).read_text() for f in ("gpu.js", "solvers.js", "graph.js", "clearance.js", "discs.js", "editor.js", "viewer.js"))
    data = json.dumps(scenes).replace("</", "<\\/")
    html = html.replace("<!--SCENES-->", f"<script>window.SCENES = {data};</script>")
    for f in ("gpu.js", "solvers.js", "graph.js", "clearance.js", "discs.js", "editor.js"):
        html = html.replace(f'<script src="{f}"></script>\n', "")
    html = html.replace('<script src="viewer.js"></script>', f"<script>\n{js}\n</script>")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(html)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("scripts", nargs="+")
    ap.add_argument("--out", default="site")
    ap.add_argument("--single-file", metavar="HTML")
    ap.add_argument("--commit", help="git commit of this build (shown in About)")
    ap.add_argument("--repo", help="GitHub owner/repo, for release-note and licence links")
    a = ap.parse_args(argv)
    if a.single_file:
        build_single_file(a.scripts, a.single_file, a.commit, a.repo)
        print(f"wrote {a.single_file}")
    else:
        keys = build_site(a.scripts, a.out, a.commit, a.repo)
        print(f"wrote {a.out}/ with scenes: {', '.join(keys)}")


if __name__ == "__main__":
    main()
