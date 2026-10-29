# Implicit modeling for parametric spine FE models: primer and roadmap

## 1. What nTop does (scouted)

Verified from nTop's own material (vendor claims are marked as such):

- **Representation.** nTop's engine represents solids as distance fields, not
  boundary representations (B-rep). Simulation results are stored as fields too,
  so they can drive geometry directly ("field-driven design"). Sources:
  [How does nTop work?](https://learn.ntop.com/lessons/how-does-ntop-work),
  [Implicit modeling for mechanical design](https://www.ntop.com/resources/blog/implicit-modeling-for-mechanical-design/),
  [Implicit modeling whitepaper](https://www.ntop.com/resources/whitepaper-implicit-modeling-technology).
- **Robustness.** nTop says parameter changes can't cause the feature failures
  that B-rep CAD regeneration can. That is a vendor claim, but the reason is
  structural: a model is a composition of functions, so there is no topology to rebuild.
- **FE output.** The *FE Volume Mesh from Implicit* block meshes an implicit
  body directly into linear or quadratic tetrahedra, with a target edge length
  that can vary with a field. It does not make hexahedra. Exports are Abaqus `.inp`
  (node, element, face and edge sets) and ANSYS `.cdb` (node and element sets).
  [Block docs](https://docs.ntop.com/Product-Documentation/ntop/block-documentation/blocks/design-analysis/mesh-generation/fe-volume-mesh-from-implicit),
  [Simulation data import/export](https://docs.ntop.com/lessons/intro-to-structural-analysis/importing-and-exporting-simulation-data).
- **Workflow (from memory, not verified here).** The interface is a notebook of
  blocks (a dataflow graph), and nTop Automate runs notebooks headless with JSON
  inputs and Python wrappers.

**Takeaway for us:** a model is a dataflow graph of field operations. A
parameter change re-evaluates the graph. Rendering evaluates the field on the
GPU per pixel. Meshing samples the field at whatever resolution FE needs.
Nothing is "rebuilt".

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

- **Surface:** marching cubes, dual contouring, or **surface nets** (implemented
  in `implicit/mesh.py`; it produces a watertight, outward-oriented mesh, which
  the tests check).
- **Volume, voxel-hex:** a cell is in a body if f < 0 at its centre. This is
  trivially robust and common in bone micro-FE. Material ID comes from the body.
  The boundary is stair-stepped unless smoothed.
- **Volume, conforming tet:** feed the surfaces to a tet mesher, or mesh the field
  directly, e.g. isosurface stuffing
  ([Labelle & Shewchuk 2007](https://people.eecs.berkeley.edu/~jrs/papers/stuffing.pdf)).
  nTop takes the direct route.
- **Sets** (endplate–cage contact, loading surfaces) come from fields too: tag
  faces where |f_cage| < ε and |f_bone| < ε.

### 2.7 Patient geometry (VerSe)

A segmentation mask M becomes an SDF via Euclidean distance transforms,
f = EDT(¬M) − EDT(M), sampled on the CT grid. It is used as a **grid primitive**:
trilinear lookup in NumPy, and a 3D texture in WebGL2 (hardware trilinear).
Every downstream operation (offset, cut, shell, cage clip) then works the same
on patient vertebrae as on analytic shapes. nTop's mesh→implicit conversion is
equivalent.

## 3. Architecture

```
examples/*.py   Python scene scripts (source of truth; version-controlled)
      │  builds a graph of SDF nodes + Params
implicit/       expr.py (scalar exprs, Params) · sdf.py (nodes: GLSL + NumPy)
      │          scene.py (bodies → GLSL/JSON) · mesh.py (surface nets, STL)
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
| **1** | **VerSe import**: NIfTI → per-vertebra SDF grid → `Grid` primitive (3D texture). Fit endplate planes and a local frame per vertebra. | nibabel (MIT), SciPy EDT (BSD). Data is CC BY-SA (see LICENSING.md) |
| 1b | Pose params per vertebra (sagittal alignment), disc generated between fitted endplates, cortical shell by offset | Same pattern as the idealised example |
| 1c | Implant library: TLIF/ALIF/LLIF cages, footprint from fitted endplates, TPMS-lattice infill option | Gyroid infill is one more field |
| **2** | **FE export**: voxel-hex (C3D8 / Marc hex8) with body material IDs, plus Marc `.dat` and Abaqus `.inp` writers, and auto node/face sets for contact and loading | Pure NumPy, no mesher dependency |
| 2b | Conforming tet mesh with a permissive-licence mesher, sizing field from |f| | See LICENSING.md for candidates |
| 3 | Speed: narrow-band / octree sampling and GPU sampling for export, plus GLSL↔NumPy parity tests | Export is ~20 s per body at 0.5 mm today |
| 4 | Field-driven FE: CT HU → density → modulus as a field, DOE batch runs driving Marc headless | Parameter studies without manual rebuilds |
| 5 | Browser node-graph editor (optional). Python stays the source of truth. | |
