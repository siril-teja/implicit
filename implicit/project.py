"""Load scene scripts and write viewer-ready JSON."""

from __future__ import annotations

import json
import runpy
from pathlib import Path

from .diagnostics import probes
from .scene import Scene

VIEWER_DIR = Path(__file__).resolve().parent.parent / "viewer"


def load_scene(script) -> Scene:
    """Scene from a block graph (``.json``) or a Python script's module-level ``scene``."""
    if Path(script).suffix == ".json":
        from .graph import load_graph
        return load_graph(script)
    ns = runpy.run_path(str(script), run_name="__scene__")
    scene = ns.get("scene")
    if not isinstance(scene, Scene):
        raise RuntimeError(f"{script} must define a module-level `scene = Scene(...)`")
    return scene


def load_graph_scene(path):
    """(scene, graph dict, library) for a block graph, bones in the shared library atlas."""
    from .graph import Library, build_graph, read_graph
    g = read_graph(path)
    lib = Library(g, Path(path).parent)
    return build_graph(g, Path(path).parent, library=lib), g, lib


def write_scenes(scripts, out_dir) -> list[str]:
    """Write scenes/<stem>.json (+ atlas, probes) and scenes/index.json; return the scene keys.

    Block graphs also get <stem>.graph.json and <stem>.library.json (bones and
    their GLSL), and the site gets blocks.json, so the viewer can edit them.
    """
    out = Path(out_dir) / "scenes"
    out.mkdir(parents=True, exist_ok=True)
    keys, graphs = [], False
    for s in scripts:
        key = Path(s).stem
        if Path(s).suffix == ".json":
            scene, g, lib = load_graph_scene(s)
            (out / f"{key}.graph.json").write_text(json.dumps(g))
            (out / f"{key}.library.json").write_text(json.dumps(lib.to_json(f"{key}.atlas.gz")))
            graphs = True
        else:
            scene = load_scene(s)
        scene.write_json(out / f"{key}.json", atlas_url=f"{key}.atlas.gz")
        if scene.atlas_gz():
            (out / f"{key}.atlas.gz").write_bytes(scene.atlas_gz())
        (out / f"{key}.probes.json").write_text(json.dumps(probes(scene)))
        keys.append(key)
    if graphs:
        from .graph import browser_manifest
        (out / "blocks.json").write_text(json.dumps(browser_manifest()))
    (out / "index.json").write_text(json.dumps(keys))
    return keys
