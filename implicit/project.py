"""Load scene scripts and write viewer-ready JSON."""

from __future__ import annotations

import json
import runpy
from pathlib import Path

from .diagnostics import probes
from .scene import Scene

VIEWER_DIR = Path(__file__).resolve().parent.parent / "viewer"


def load_scene(script) -> Scene:
    """Run a Python scene script and return its module-level ``scene``."""
    ns = runpy.run_path(str(script), run_name="__scene__")
    scene = ns.get("scene")
    if not isinstance(scene, Scene):
        raise RuntimeError(f"{script} must define a module-level `scene = Scene(...)`")
    return scene


def write_scenes(scripts, out_dir) -> list[str]:
    """Write scenes/<stem>.json and scenes/index.json; return the scene keys."""
    out = Path(out_dir) / "scenes"
    out.mkdir(parents=True, exist_ok=True)
    keys = []
    for s in scripts:
        key = Path(s).stem
        scene = load_scene(s)
        scene.write_json(out / f"{key}.json", atlas_url=f"{key}.atlas.gz")
        if scene.atlas_gz():
            (out / f"{key}.atlas.gz").write_bytes(scene.atlas_gz())
        (out / f"{key}.probes.json").write_text(json.dumps(probes(scene)))
        keys.append(key)
    (out / "index.json").write_text(json.dumps(keys))
    return keys
