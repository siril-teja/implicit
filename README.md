# implicit

Implicit (signed-distance-field) modeling for parametric spine FE
models. Geometry is written as Python scripts, rendered live in the browser,
and exported for meshing.

- **Python** builds a graph of field operations (primitives, booleans, smooth
  blends, offsets/shells, cuts, rigid transforms) with named `Param`s.
- **Browser** (WebGL2) ray-marches the same graph compiled to GLSL. Params are
  shader uniforms, so sliders update the whole assembly in real time with no
  rebuild.
- **Export:** the graph is evaluated in NumPy and meshed with narrow-band
  marching tetrahedra to closed, manifold STL. FE mesh export (Marc/Abaqus) is next.

Read [docs/primer.md](docs/primer.md) for the background, the math, and the
roadmap. Read [docs/LICENSING.md](docs/LICENSING.md) for the dependency policy
(commercial-use-safe only).

## Patient anatomy: implicit bones

TotalSegmentator provides voxel label masks (NIfTI), not CAD. They are converted
**once**, bone by bone, into implicit bodies: one signed-distance file per bone
(`assets/bones/<subject>/<bone>.npz`, format in `implicit/anatomy/convert.py`).
With the CT available, each surface is placed on the CT iso-surface (half-maximum
threshold) and the label only guards it to ±2 mm. From then on all modelling
works on these fields; masks are never read again.

`examples/anatomy.py` shows `assets/bones/s0751`: L1–L5, sacrum, hips and proximal
femurs from a TotalSegmentator v2 CT (CC BY 4.0).

```bash
python -m implicit.anatomy.totalseg s0751 --out data/totalseg --ct      # masks + CT, ~40 MB
python -m implicit.anatomy.convert data/totalseg/s0751 --out assets/bones/s0751 --select lumbar
python -m implicit.serve examples/anatomy.py
```

Each bone has a **smoothing** slider (σ in mm, Gaussian blur of the implicit
field) that keeps the bone volume by default. To mesh
bodies at chosen slider values (closed, manifold STL):

```bash
python -m implicit.export examples/anatomy.py --body L3 --set smooth_L3=1.0 --voxel 0.4 --out out/
```

`--select` takes `lumbar`, `all` or a list (`L3,L4,sacrum`). Optional:
`python -m implicit.anatomy.landmarks assets/bones/s0751` fits vertebral frames
and endplate planes from the implicit bones (`landmarks.json`; L5 is not yet reliable).

## Quick start

```bash
pip install -r requirements.txt          # numpy (+ scipy, nibabel for the conversion)
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

## Releases and live viewer

`main` only changes through pull requests that pass CI. Each merged PR with a
new version is tagged and released automatically, and the viewer at
https://siril.me/implicit/ is updated from it. Release notes are in
[CHANGELOG.md](CHANGELOG.md), and licences and data attribution in [NOTICE.md](NOTICE.md).
The workflow and one-time repository settings are in [CONTRIBUTING.md](CONTRIBUTING.md).

## Status

Prototype, not validated. The idealised L4–L5 example is not patient-specific.
The anatomy is one CT subject (1.5 mm CT, bones resampled to 0.6–1.5 mm). Exported STL is a
surface for inspection only, and FE volume meshing is not implemented yet.

## Licence

The code is released under the [MIT License](LICENSE). The bundled anatomy in
`assets/bones/` is derived from TotalSegmentator (CC BY 4.0); see
[NOTICE.md](NOTICE.md) for all third-party licences and attribution.
