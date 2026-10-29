# implicit

nTop-style implicit (signed-distance-field) modeling for parametric spine FE
models. Geometry is written as Python scripts, rendered live in the browser,
and exported for meshing.

- **Python** builds a graph of field operations (primitives, booleans, smooth
  blends, offsets/shells, cuts, rigid transforms) with named `Param`s.
- **Browser** (WebGL2) ray-marches the same graph compiled to GLSL. Params are
  shader uniforms, so sliders update the whole assembly in real time with no
  rebuild.
- **Export:** the graph is evaluated in NumPy and meshed with surface nets to
  STL. FE mesh export (Marc/Abaqus) is next.

Read [docs/primer.md](docs/primer.md) for the nTop scouting, the math, and the
roadmap. Read [docs/LICENSING.md](docs/LICENSING.md) for the dependency policy
(commercial-use-safe only).

## Patient anatomy (full spine + pelvis)

`examples/full_spine.py` loads `assets/models/s0751`: C1–L5, sacrum, both hips and
proximal femurs from a TotalSegmentator v2 CT (CC BY 4.0). It detects the endplates
and builds all 23 discs (C2–C3 … L5–S1) implicitly, as annulus + nucleus, with live
parameters. To process another subject:

```bash
python -m implicit.anatomy.totalseg s0751 --out data/totalseg        # ~4 MB, no full download
python -m implicit.anatomy.preprocess data/totalseg/s0751 --out assets/models/s0751
IMPLICIT_MODEL=assets/models/s0751 python -m implicit.serve examples/full_spine.py
```

## Quick start

```bash
pip install -r requirements.txt          # numpy (+ scipy, nibabel for preprocessing)
python -m implicit.serve examples/spine_segment.py
# open http://127.0.0.1:8000
```

Edit `examples/spine_segment.py` and save, and the viewer reloads. "Export STL"
meshes the visible bodies at the current slider values into `out/`.

```bash
python -m unittest discover -s tests     # tests
python -m implicit.build examples/*.py --out site            # static site
python -m implicit.build examples/spine_segment.py --single-file spine.html
```

## Writing a scene

```python
from implicit import Param, Scene, Box, HalfSpace, radians, sin, cos

angle = Param("angle", 8, 0, 20, 0.5, "Wedge angle", "deg")
t = radians(angle)
cage = Box((15, 9, 50), round=1.5) & HalfSpace((0, 0, -5), (0, 0, -1)) \
       & HalfSpace((0, 0, 5), (0, -sin(t), cos(t)))

scene = Scene("cage", bounds=((-20, -15, -15), (20, 15, 15)))
scene.add("Cage", cage, color=(0.3, 0.8, 0.4))
```

## GitHub Pages

`.github/workflows/pages.yml` runs the tests on every push and deploys the
viewer from `main`. Turn it on once in **Settings → Pages → Source: GitHub
Actions**. The site will be `https://siril-teja.github.io/implicit/`.

## Status

Prototype, not validated. The idealised L4–L5 example is not patient-specific.
The full-spine model is one CT subject at 1.5 mm resolution. Exported STL is a
surface for inspection only, and FE volume meshing is not implemented yet.
