"""Local dev server: viewer + live reload of scene scripts + STL export.

    python -m implicit.serve examples/spine_segment.py [more.py ...] [--port 8000]

Edit a scene script and save: the browser recompiles the shader within ~1 s.
Moving sliders never touches Python; "Export STL" meshes on the Python side
with the slider values currently shown in the browser.
"""

from __future__ import annotations

import argparse
import json
import re
import time
import traceback
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from .mesh import surface_nets, write_stl
from .project import VIEWER_DIR, load_scene


class Project:
    def __init__(self, scripts, out_dir, voxel):
        self.scripts = {Path(s).stem: Path(s).resolve() for s in scripts}
        self.cache = {}  # key -> (mtime, scene, json_bytes)
        self.out_dir = Path(out_dir)
        self.voxel = voxel

    def get(self, key):
        path = self.scripts[key]
        mtime = path.stat().st_mtime
        hit = self.cache.get(key)
        if hit and hit[0] == mtime:
            return hit[1], hit[2]
        scene = load_scene(path)
        data = json.dumps(scene.to_json()).encode()
        self.cache[key] = (mtime, scene, data)
        print(f"[implicit] compiled {key} ({len(scene.bodies)} bodies, {len(scene.params())} params)")
        return scene, data

    def export(self, key, params, visible):
        scene, _ = self.get(key)
        scene.set_params(params)
        out = self.out_dir / key
        out.mkdir(parents=True, exist_ok=True)
        files = []
        for i, b in enumerate(scene.bodies):
            if visible and not visible[i]:
                continue
            t0 = time.time()
            f, origin, h = scene.sample(b.name, self.voxel)
            v, t = surface_nets(f, origin, h)
            if len(t) == 0:
                continue
            fn = out / (re.sub(r"[^A-Za-z0-9_-]+", "_", b.name) + ".stl")
            write_stl(fn, v, t, b.name)
            files.append(f"{fn}  ({len(t)} tris, {time.time() - t0:.1f} s)")
        (out / "params.json").write_text(json.dumps(params, indent=1))
        return files


class Handler(SimpleHTTPRequestHandler):
    project: Project = None

    def log_message(self, fmt, *args):
        pass

    def _json(self, code, obj=None, raw=None):
        body = raw if raw is not None else json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = self.path.split("?")[0]
        if path == "/api/ping":
            return self._json(200, {"ok": True})
        if path == "/scenes/index.json":
            return self._json(200, list(self.project.scripts))
        m = re.fullmatch(r"/scenes/(\w+)\.json", path)
        if m and m.group(1) in self.project.scripts:
            try:
                _, data = self.project.get(m.group(1))
                return self._json(200, raw=data)
            except Exception:
                traceback.print_exc()
                return self._json(500, {"error": traceback.format_exc(limit=3)})
        return super().do_GET()

    def do_POST(self):
        if self.path != "/api/export":
            return self._json(404, {"error": "not found"})
        try:
            req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            key = req.get("scene") or next(iter(self.project.scripts))
            files = self.project.export(key, req.get("params", {}), req.get("visible"))
            return self._json(200, {"files": files})
        except Exception as e:
            traceback.print_exc()
            return self._json(500, {"error": str(e)})


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("scripts", nargs="+")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--out", default="out", help="STL export directory")
    ap.add_argument("--voxel", type=float, default=0.5, help="mesh grid spacing (scene units)")
    a = ap.parse_args(argv)
    Handler.project = Project(a.scripts, a.out, a.voxel)
    for k in Handler.project.scripts:
        Handler.project.get(k)
    srv = ThreadingHTTPServer(("127.0.0.1", a.port), partial(Handler, directory=str(VIEWER_DIR)))
    print(f"[implicit] http://127.0.0.1:{a.port}/  (Ctrl+C to stop)")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
