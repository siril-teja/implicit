# Implicit modeling for parametric spine FE models: primer and roadmap

## 1. Why implicit modeling

- **Representation.** A solid is the sub-level set of a scalar field rather than
  a boundary representation (B-rep). Function representation (FRep) and its
  set operations are described by
  [Pasko et al. 1995](https://doi.org/10.1007/BF02464333); sampled distance
  fields for shapes from scans by
  [Frisken et al. 2000](https://doi.org/10.1145/344779.344899).
- **Robustness.** A model is a composition of functions, so changing a
  parameter re-evaluates the composition. There is no topology to rebuild, so
  none of the regeneration failures of feature-based B-rep CAD.
- **Fields everywhere.** Geometry, material properties and simulation results
  are all fields on the same domain, so one can drive another (e.g. CT density
  driving modulus, or a sizing field driving the FE mesh).

**Takeaway:** a model is a dataflow graph of field operations. A parameter
change re-evaluates the graph. Rendering evaluates the field on the GPU per
pixel. Meshing samples the field at whatever resolution FE needs. Nothing is
"rebuilt".

## 2. The math

### 2.1 Implicit solids and signed distance

A solid is Ω = { x ∈ ℝ³ : f(x) ≤ 0 } for a scalar field f. A **signed distance
field** (SDF) also satisfies |∇f| = 1 (the Eikonal equation), so f(x) is the
distance to ∂Ω with a sign. Most operations keep f **1-Lipschitz**
(|f(x) − f(y)| ≤ |x − y|) without keeping it exact. Lipschitz is enough for
ray marching and iso-surfacing.

| Operation | Field | Exact SDF? |
|---|---|---|
| Union / intersection / difference | min(a,b), max(a,b), max(a,−b) | Bound only (exact outside for union) |
| Smooth union (blend radius k) | polynomial smooth-min | Bound |
| Offset by d | f − d | Exact if f is exact |
| Shell, thickness t (e.g. cortex) | max(f, −(f + t)) | Exact if f is exact |
| Rigid transform T | f(T⁻¹x) | Preserves exactness |
| Uniform scale s | s·f(x/s) | Preserves exactness |
| Non-uniform scale / warp | f(W(x)) / L | Needs a Lipschitz factor L |
| Half-space (a cut plane) | n·(x − x₀) | Exact |

This is why osteotomies, cages and cortical shells are so cheap here. A cut is
one `max`, a wedge is two half-spaces, a cortical shell is one line, and a
press-fit clearance is an `offset`.

### 2.2 Parameters as uniforms (why it updates instantly)

The Python graph is compiled once into straight-line GLSL. Every `Param` becomes
a shader `uniform`, so moving a slider rewrites one float on the GPU and redraws.
The shader recompiles (about 20–40 ms here) only when the graph's *structure*
changes, i.e. when you edit the Python script.

### 2.3 Rendering: sphere tracing

For a ray x(t) = o + t·d, a 1-Lipschitz f allows a safe step of t ← t + f(x(t))
([Hart 1996](https://graphics.stanford.edu/courses/cs348b-07/sphere-tracing.pdf)).
We step by 0.85·f to allow for the bound-only operations. Normals come from ∇f
by finite differences. Section views are one more `max` with a plane, applied
in the shader.

### 2.4 Assemblies and conformity

The disc space is the column footprint ∩ {above L5's endplate} ∩ {below L4's
endplate}. Its planes are functions of the `lordosis` and `disc_h` params. The
cage is clipped by the *same* planes, so its wedge angle and height always match
the disc space. The disc is `disc − offset(cage, 0.2)`. Interfaces are
consistent by construction, which is the property you need for contact or tied
FE interfaces.

### 2.5 Osteotomy as a domain map

A PSO resects a wedge with angle α between two planes through an anterior hinge.
Closing it is a rigid rotation of the upper fragment about the hinge:
f_closed = min(f ∩ below, (f ∩ above)∘R(−α·c)), with closure c ∈ [0, 1].
Varying c animates the correction. The same pattern, a piecewise rigid map,
works for SPO/Ponte osteotomies and multilevel alignment changes.

### 2.6 From fields to meshes

- **Surface:** marching cubes, dual contouring, surface nets, or **marching
  tetrahedra** (implemented in `implicit/mesh.py`, sampled only in a narrow band
  around the surface; closed, manifold and outward-oriented by construction,
  which the tests check).
- **Volume, voxel-hex:** a cell is in a body if f < 0 at its centre. This is
  trivially robust and common in bone micro-FE. Material ID comes from the body.
  The boundary is stair-stepped unless smoothed.
- **Volume, conforming tet:** feed the surfaces to a tet mesher, or mesh the field
  directly, e.g. isosurface stuffing
  ([Labelle & Shewchuk 2007](https://people.eecs.berkeley.edu/~jrs/papers/stuffing.pdf)).
- **Sets** (endplate–cage contact, loading surfaces) come from fields too: tag
  faces where |f_cage| < ε and |f_bone| < ε.

### 2.7 Patient geometry (TotalSegmentator)

Conversion happens once per bone (`implicit/anatomy/convert.py`):

1. Mask M (largest component) → f_M = EDT(¬M) − EDT(M), half-voxel corrected.
2. With the CT: occupancy = {f_M < −2 mm} ∪ {|f_M| < 2 mm ∧ nearest label is
   this bone ∧ CT > T}. T is the half-maximum edge between the soft-tissue
   level (3–5 mm outside the label) and the cortical peak (near the label
   surface). Holes are filled and the result is re-distanced on the target grid.
   On a synthetic sphere in a 1.5 mm partial-volume CT, labels off by ±1.5 mm
   give a mean surface error of 0.8–1.45 mm from the mask alone, versus
   ≤ 0.12 mm with the CT.
3. Stored as one file per bone (uint8 SDF, 0.1 mm per level, scanner frame).

Each file becomes a **grid primitive**: trilinear lookup in NumPy, and a 3D
texture in WebGL2 (hardware trilinear). Every downstream operation (offset,
cut, shell, transform, boolean) then works the same on patient bones as on
analytic shapes.

### 2.8 Endplates on patient anatomy

1. **Body isolation** (for the local frame only): remove points behind the
   posterior wall (marching anteriorly from the spinal-cord centroid), then
   morphologically open (radius 4.5–5.5 mm) and keep the largest component.
2. **Frame.** The axial axis w starts as the spine tangent through the
   neighbouring vertebrae (one-sided at the ends). a is anterior and l = a × w.
3. **Endplates.** From the body centre, march ±w along central columns to the
   SDF zero crossing (sub-voxel), then fit w = αu + βv + γ by iteratively
   trimmed least squares. w is then replaced by the bisector of the two
   endplate normals and steps 2–3 repeat until w settles (< 0.5°); the origin
   is midway between the endplate centres. S1 is found from L5 and C2 from C3.

## 3. Architecture

```
examples/*.py   Python scene scripts (source of truth; version-controlled)
      │  builds a graph of SDF nodes + Params
implicit/       expr.py (scalar exprs, Params) · sdf.py (nodes: GLSL + NumPy)
      │          scene.py (bodies → GLSL/JSON) · mesh.py (marching tets, STL)
      ├──► scene JSON ──► viewer/ (WebGL2 sphere tracer, sliders, section)
      │                    static → GitHub Pages; local → live reload + export
      └──► NumPy sampling ──► STL now; FE meshes (Marc/Abaqus) next
```

Every node implements the same math twice, in GLSL (display) and in NumPy
(export). GLSL↔NumPy parity tests are on the roadmap.

## 4. Roadmap

| Phase | Deliverable | Notes |
|---|---|---|
| **0, done** | SDF kernel, Params, GLSL codegen, browser viewer, section view, STL export, Pages deploy | Idealised L4–L5 with wedge cage and PSO |
| **1, done** | **Conversion**: TotalSegmentator masks + CT → one implicit bone file per bone (CT half-maximum surface, label as ±2 mm guard), scanner frame | `implicit/anatomy/convert.py` |
| 1b | Landmarks on implicit bones: body frames, endplate planes (L5 not yet reliable) | `implicit/anatomy/landmarks.py` |
| 1e | Pose params per vertebra (alignment), osteotomies on patient anatomy, cortical shell | |
| 2, on hold | **FE meshing** (feeds phase 7): voxel-hex (C3D8 / Marc hex8) with body material IDs, plus Marc `.dat` and Abaqus `.inp` writers, and auto node/face sets for contact and loading | Pure NumPy, no mesher dependency |
| 2b | Conforming tet mesh with a permissive-licence mesher, sizing field from |f| | See LICENSING.md for candidates |
| 3 | Speed: narrow-band / octree sampling and GPU sampling for export, plus GLSL↔NumPy parity tests | Export is ~20 s per body at 0.5 mm today |
| 4 | Field-driven FE: CT HU → density → modulus as a field, DOE batch runs driving Marc headless | Parameter studies without manual rebuilds |
| **5, done** | Block graphs: JSON graph format and block registry, browser block canvas with live rebuild, selection between canvas and 3D view | `implicit/graph.py`, `viewer/editor.js` |
| **6, in progress** | Anatomy blocks: vertebral frames and endplates (done), pose (done), discs (annulus, nucleus, fibres), cages, osteotomies, ligaments | Each a block that wires into the others |
| 7 | Lightweight FE solver in the browser (open source) | After the model is complete |
