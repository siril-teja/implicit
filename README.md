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
  marching tetrahedra to closed, manifold STL.
- **Virtual patients** (in progress): an average spine with pelvis and rib cage
  (VIVA+ 50F) posed by clinical alignment sliders within physiological ranges of
  motion. Next steps (roadmap): a connected rib cage, collision-free distribution,
  discs, ligaments, then meshing and FE.

Read [docs/primer.md](docs/primer.md) for the background and the math, and
[docs/ROADMAP.md](docs/ROADMAP.md) for the roadmap: one objective and its acceptance
criteria per step, each step one pull request, and
[docs/LITERATURE.md](docs/LITERATURE.md) for every source used, what we took from it
and where. Read [docs/LICENSING.md](docs/LICENSING.md) for the dependency policy
(commercial-use-safe only).

## Average spine (VIVA+)

`examples/average_spine.json`, the default scene, is an idealised spine:
T1–T12, L1–L5, sacrum, hip bones, proximal femurs, ribs, sternum and costal
cartilages of the
[VIVA+](https://zenodo.org/records/18338876) average-female human body model (50F,
standing; LGPL-3.0-or-later, see [NOTICE](NOTICE.md)). Its bone surfaces are
converted to the same implicit bone files as the patient data:

```bash
python -m implicit.anatomy.vivaplus --model 50F-standing --select full --out assets/bones/vivaplus-50F   # downloads VIVA+ once
python -m implicit.anatomy.landmarks assets/bones/vivaplus-50F
```

The fitted lumbar endplate planes agree with VIVA+'s own endplate shells to
0.7–2.4° and 0.3 mm (normal offset).

## Spine alignment

Sliders in the viewer's Scene tab reshape the average spine (VIVA+ 50F):

| Plane | Slider | Meaning |
|---|---|---|
| sagittal | PI, PT | pelvic incidence, pelvic tilt (SS = PI - PT) |
| sagittal | LL, TK | lumbar lordosis L1-S1, thoracic kyphosis T4-T12 |
| coronal | MT, TL | Cobb angles of the main thoracic (T5-T8-T12) and thoracolumbar/lumbar (T12-L1-L4) curves, + convex right |
| coronal | AR | apical axial rotation per degree of Cobb (toward the convexity) |
| coronal | PO, CB | pelvic obliquity (+ right side up), coronal balance (T1 right of S1, mm) |
| limits | ML, CP | motion limits per level (1 = on), coupled axial rotation of lateral bending (1 = physiological) |

They start at the subject's own values (VIVA+ 50F: PI 58.7, PT 17.2, SS 41.5,
LL 51.9, TK 30.3, MT 0, TL -0.3 deg, CB 0 mm), where nothing moves. The end
vertebrae and apex of each curve are settings of the **Spine alignment** block.
Every bone moves rigidly: the pelvis about the hip axis (PT) and with the femurs
about the antero-posterior axis (PO), the sacrum about the sacroiliac joints (PI),
and each disc acts as a hinge (LL, TK, MT, TL; balance on the lumbosacral discs
below the curves). Soft structures follow a soft-tissue field (**Field warp** block;
the spine curve through the vertebral body centres is the first), chosen on the
Spine alignment block:

- **TPS**: a 3-D thin-plate spline through control points on every bone;
- **polyrigid**: rigid inside each vertebral body, and in each disc space the screw
  motion between its two vertebrae (log-Euclidean blend by position between the
  endplate planes); exactly rigid at the bones, but more costly to evaluate.

Each disc turns about its physiological rotation centres (lumbar: Pearcy & Bogduk
1988; thoracic: Qiu et al. 2003; definitions and references in
`implicit/anatomy/rotation.py`), stored with the landmarks.

**Motion limits** (ML = 1, the default): no disc turns from its rest orientation by
more than half its in vitro range of motion in each plane (flexion-extension,
lateral bending, axial rotation; medians of Wilke et al. 2025,
[doi:10.1002/jsp2.70052](https://doi.org/10.1002/jsp2.70052), thoracic levels with
the rib cage). A target is spread over its discs as before until a disc reaches
its limit, then over the others; what no disc can take is listed under **Motion
limits** in the Scene tab (target minus reached; the discs at a limit in the
tooltip). Lateral bending of the thoracic discs carries 0.40 deg of axial rotation
in the direction of the bend per degree (CP = 1; Fujimori et al. 2014,
[doi:10.1016/j.spinee.2013.11.054](https://doi.org/10.1016/j.spinee.2013.11.054)).
The ranges are small: T4-T12 together allow about +-7 deg of kyphosis change and
L1-S1 about +-22 deg of lordosis change from the template, so differences in
posture between people beyond that come from the shape of vertebrae and discs
(roadmap 3.2, 3.3), not motion. ML = 0 lifts the limits. Tables and assumptions:
`implicit/anatomy/motion.py`.

The **Field view** block shows the field: flow lines (the paths tissue near the
spine travels while the sliders go from the subject's values to the current ones,
coloured by the distance travelled), a lattice it carries, arrows of the
displacement on a 15 mm grid, or a slice coloured by it, with a colour scale. The
average spine scene has all four as hidden groups in the Scene tab. Blocks: Spine
alignment, Align (one per bone), Spine curve, Field warp, Field view. Model and
definitions: `implicit/anatomy/alignment.py`.

```bash
python -m implicit.anatomy.alignment assets/bones/vivaplus-50F     # print the measured parameters
```

The femoral head centres (sphere fits) and sacroiliac joint centres it needs are
found by `python -m implicit.anatomy.landmarks` and stored in `landmarks.json`.
The patient scene (lumbar only) has the sagittal sliders and pelvic obliquity, with
the motion limits on (the block's default).

**Clearance.** The viewer measures the smallest gap between every pair of
neighbouring bones (consecutive vertebrae, L5 and sacrum, sacrum and L5 with each
hip bone, hip bone and femur) at the current sliders: a table under Clearance,
smallest first, and colliding bones tinted red. A bone's field is evaluated at the
other bone's surface points (2 mm apart), both ways; negative is the depth of the
overlap. The motion limits (ML) keep most postures clear (LL 80 with TK 60 collides
at L4-L5 only with ML = 0); the sliders do not yet avoid collisions directly (roadmap
step 1.7).

```bash
python -m implicit.anatomy.clearance examples/average_spine.json --set spp_LL=80 --set spp_TK=60 --set spp_ML=0
python -m implicit.anatomy.clearance examples/average_spine.json --sweep    # collisions over a grid of sliders
```

**Disc space heights.** Each endplate has its anterior, posterior, left and right
corners and its centre (stored with the landmarks); the disc space between two
vertebrae has anterior, middle, posterior, left and right heights, measured along
the disc axis, in the viewer's Discs table (A / M / P; left and right in the
tooltip). A disc is red when the posture deforms it beyond what healthy discs do in
vivo (middle height +-10 %, anterior/posterior/left/right +-40 % of rest; Xi et al.
2025), and marked * when its rest height is outside the population mean +- 2 SD
(lumbar: Shin et al. 2024, CT, women; thoracic: Kunkel et al. 2011). Definitions and
references: `implicit/anatomy/discs.py`. The VIVA+ 50F template's vertebrae are translated
so that its rest disc spaces have the population middle heights, and its facet
joints are carved to the population joint space (`implicit/anatomy/template.py`;
orientations and alignment unchanged). The Discs table also shows each disc's
thinnest point over the whole endplates, red below its floor (at least 1.5 mm).

```bash
python -m implicit.anatomy.discs examples/average_spine.json --set spp_TK=60 --set spp_ML=0
python -m implicit.anatomy.discs examples/average_spine.json --sweep    # flagged discs over a grid of sliders
```

## Rib cage

The average spine has the VIVA+ 50F ribs (1-12, both sides), sternum and costal
cartilages. Each rib follows its vertebra with the alignment sliders (Align block,
segment T1 ... T12); the sternum and costal cartilages are carried by the
soft-tissue field (Field warp), like the spine curve. That field only follows the
spine, so with large slider changes the sternum and cartilages drift from the rib ends
(roadmap 1.6: a connected rib cage). The clearance table lists every
rib against its vertebra and the one above (costovertebral joints, carved to 1.3 mm
in the template) and against the next rib on its side; e.g. MT 40 crowds the ribs of
the concave side into each other (with ML = 0), which rib motion (roadmap 2.7) and
the constrained distribution (1.7) are to resolve.

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
pip install -r requirements.txt          # numpy (+ scipy, nibabel for the conversion); exact versions
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
The anatomy is one CT subject (1.5 mm CT, bones resampled to 0.6–1.5 mm) and the
VIVA+ 50F average female template (T1 to femurs with rib cage, corrected to
population disc heights and joint spaces). Postures are rigid-body kinematics within
in vitro ranges of motion; discs, ligaments and muscles are not modelled yet.
Exported STL is a surface for inspection only, and FE volume meshing is not
implemented yet.

## Licence

The code is released under the [MIT License](LICENSE). The bundled anatomy in
`assets/bones/` is derived from TotalSegmentator (`s0751`, CC BY 4.0) and VIVA+
(`vivaplus-50F`, LGPL-3.0-or-later); see
[NOTICE.md](NOTICE.md) for all third-party licences and attribution.
