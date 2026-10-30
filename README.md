# implicit

Implicit (signed-distance-field) modeling for parametric spine FE
models. A model is a graph of blocks (bones, smoothing, booleans, transforms,
offsets, cuts) edited on a block canvas in the browser or written in Python,
rendered live, and exported for meshing.

- **Block canvas** (browser): wire blocks together and the 3D view rebuilds as
  you connect them; click a bone in 3D to see and edit every block that builds it.
- **Python** builds the same kind of graph of field operations with named `Param`s.
- **Browser** (WebGL2) ray-marches the same graph compiled to GLSL. Params are
  shader uniforms, so sliders update the whole assembly in real time with no
  rebuild.
- **Export:** the graph is evaluated in NumPy and meshed with narrow-band
  marching tetrahedra to closed, manifold STL. Next: anatomy blocks (endplates,
  discs, ligaments, osteotomies, cages), then a lightweight FE solver in the browser.

Read [docs/primer.md](docs/primer.md) for the background, the math, and the
roadmap. Read [docs/LICENSING.md](docs/LICENSING.md) for the dependency policy
(commercial-use-safe only).

## Average spine (VIVA+)

`examples/average_spine.json`, the default scene, is an idealised spine:
T1–T12, L1–L5, sacrum, hip bones and proximal femurs of the
[VIVA+](https://zenodo.org/records/18338876) average-female human body model (50F,
standing; LGPL-3.0-or-later, see [NOTICE](NOTICE.md)). Its bone surfaces are
converted to the same implicit bone files as the patient data:

```bash
python -m implicit.anatomy.vivaplus --model 50F-standing --out assets/bones/vivaplus-50F   # downloads VIVA+ once
python -m implicit.anatomy.landmarks assets/bones/vivaplus-50F
```

The fitted lumbar endplate planes agree with VIVA+'s own endplate shells to
0.7–2.4° and 0.3 mm (normal offset).

## Spinopelvic alignment

Four sliders in the viewer's Scene tab set the sagittal alignment of a subject:
pelvic incidence (PI), pelvic tilt (PT), lumbar lordosis L1-S1 (LL) and thoracic
kyphosis T4-T12 (TK); sacral slope follows as SS = PI - PT. They start at the
subject's own values (VIVA+ 50F: PI 58.7, PT 17.2, SS 41.5, LL 51.9, TK 30.3
deg), where nothing moves. Changing them turns the pelvis about the hip axis (PT),
the sacrum about the sacroiliac joints (PI) and the discs L1-S1 (LL) and T4-T12
(TK), each in proportion to its own angle; the targets are met exactly. A tube
along the spine curve (through the vertebral body centres) follows. Blocks:
**Spinopelvic alignment** (sliders in, one alignment wire out), **Align** (one per
segment: pelvis, sacrum, L5 ... T1) and **Spine curve**. Model and definitions:
`implicit/anatomy/alignment.py`.

```bash
python -m implicit.anatomy.alignment assets/bones/vivaplus-50F     # print the measured parameters
```

The femoral head centres (sphere fits) and sacroiliac joint centres it needs are
found by `python -m implicit.anatomy.landmarks` and stored in `landmarks.json`.

## Patient anatomy: implicit bones

TotalSegmentator provides voxel label masks (NIfTI), not CAD. They are converted
**once**, bone by bone, into implicit bodies: one signed-distance file per bone
(`assets/bones/<subject>/<bone>.npz`, format in `implicit/anatomy/convert.py`).
With the CT available, each surface is placed on the CT iso-surface (half-maximum
threshold) and the label only guards it to ±2 mm. From then on all modelling
works on these fields; masks are never read again.

`examples/patient.json` (a block graph) shows `assets/bones/s0751`: L1–L5,
sacrum, hips and proximal femurs from a TotalSegmentator v2 CT (CC BY 4.0).

```bash
python -m implicit.anatomy.totalseg s0751 --out data/totalseg --ct      # masks + CT, ~40 MB
python -m implicit.anatomy.convert data/totalseg/s0751 --out assets/bones/s0751 --select lumbar
python -m implicit.serve examples/patient.json
```

Each bone feeds a **Smooth** block (σ in mm, Gaussian blur of the implicit
field) that keeps the bone volume by default. To mesh
bodies at chosen values (closed, manifold STL):

```bash
python -m implicit.export examples/patient.json --body L3 --set L3_smooth_sigma=1.0 --voxel 0.4 --out out/
```

`--select` takes `lumbar`, `all` or a list (`L3,L4,sacrum`). Then
`python -m implicit.anatomy.landmarks assets/bones/s0751` fits vertebral frames
and endplate planes from the implicit bones (`landmarks.json`), which the
**Landmarks** block outputs as a frame and two planes per vertebra.

## Block canvas and block graphs

A block graph is a JSON file of blocks (bone, landmarks, pose, smooth, booleans,
transforms, offsets, cuts, outputs) joined by typed wires (body, bone, scalar,
vector, plane, frame, vertebra). In
the viewer, the panel at the bottom holds the **block canvas** and the settings:

- Double-click the canvas (or **+ Block**) to add a block; drag from an output
  socket to an input socket to connect; drag a wire off an input to disconnect.
- Click a block to edit its values; values update the 3D view instantly,
  rewiring rebuilds the shader (in the browser, so it works on the live site).
- Click a body in the 3D view to highlight it and list every block that builds it.
- **Tab** hides or shows the panel (full-window 3D); drag its top edge to resize.
- Edits are kept in the browser; **Download** saves the graph as a file,
  **Open…** loads one, **Reset** returns to the published graph. With the dev
  server, **Save** writes the graph back to its file.

Literal values in a graph are live parameters named `<node>_<input>`.

```bash
python -m implicit.graph check examples/average_spine.json
python -m implicit.graph blocks          # every block, its inputs and outputs
```

Format: `implicit/graph.py`. The browser builds each block's shader from code
recorded from the Python implementation, and the GPU check compares the
browser-built shader with NumPy too.

## Quick start

```bash
pip install -r requirements.txt          # numpy (+ scipy, nibabel for the conversion)
python -m implicit.serve examples/average_spine.json examples/spine_segment.py
# open http://127.0.0.1:8000
```

Edit `examples/spine_segment.py` and save, and the viewer reloads. "Export STL"
meshes the visible bodies at the current slider values into `out/`.

```bash
python -m unittest discover -s tests     # tests
python -m implicit.build examples/*.json examples/*.py --out site            # static site
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
