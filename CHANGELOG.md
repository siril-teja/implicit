# Changelog

All notable changes to this project are listed here, newest first.
Format: one `## [X.Y.Z] - YYYY-MM-DD` section per release; the release workflow
publishes that section as the GitHub release notes. Versions follow
[Semantic Versioning](https://semver.org/) (0.x while the API is still moving).

## [0.7.0] - 2026-10-30

### Added
- **Pose block** (Anatomy): moves a body in a vertebra's own frame. Flexion (+) /
  extension (−), lateral bending (+ to the right) and axial rotation (+ front
  turns left) about the vertebral body centre, plus a shift along the
  vertebra's right/anterior/cranial axes. The vertebra's frame and endplates
  move with it, so blocks downstream follow.
- **Vertebra wire type**: a vertebra's frame and both endplate planes on one
  wire. Landmarks outputs it; **Vertebra parts** splits it; **Show vertebra**
  draws its axes and endplates (and follows Pose).
- `examples/anatomy.json`: every vertebra is bone → smooth → pose → output,
  with its landmarks driving the pose. Clicking a vertebra lists its smoothing
  and pose sliders together. Its markers follow the pose.
- Blocks can output moved planes, frames and vertebrae (not only bodies); the
  browser evaluates them from expressions recorded from the Python blocks.

### Fixed
- Landmark axes are re-orthonormalised when loaded (`landmarks.json` stores 4
  decimals), so a pose with zero angles is exactly the identity.

## [0.6.0] - 2026-10-30

### Added
- **Frame-rate readout** at the top right of the 3D view: frames per second while
  the view moves (amber below 30, red below 15) and the preview resolution, then
  the time the sharp image took, measured when the GPU has finished it.
- **Quality** setting (Scene tab): pixel density of the sharp image on high-DPI
  screens (high 2×, balanced 1.5× by default, fast 1×).

### Changed
- Faster rendering, measured on the bundled scenes (software renderer, full frame):
  - surface normals from 3 scene evaluations instead of 4, ambient occlusion
    from 3 samples instead of 5: −17 % (anatomy), −13 % (spine segment);
  - while the view moves, no ambient occlusion and a looser hit tolerance:
    −28 % / −23 %; the sharp image keeps full quality;
  - smoothed bones only sample the blur levels with non-zero weight;
  - the preview resolution adapts to keep moving at about 30 fps.

## [0.5.0] - 2026-10-30

### Added
- **Landmarks block** (Anatomy): from a vertebra or the sacrum, outputs its body
  frame (origin between the endplates; right, anterior and cranial axes), its
  superior and inferior endplate planes, and a marker body (endplate discs and
  axes) to check them in 3D. The sacrum gives the S1 endplate.
- New wire types **plane** and **frame**, and blocks Plane, Plane parts, Frame
  parts, Show plane and Show frame. For example, Frame parts → origin wired to
  Rotate's centre rotates a vertebra about its own body centre.
- `examples/anatomy.json` has a Landmarks block for every vertebra and the
  sacrum; their markers are in the "Landmarks" group (hidden at first).
- CI's GPU check also builds a test graph that uses every landmark block.

### Fixed
- Vertebral frames: the axial axis is now the bisector of the two endplate
  normals, refined until it settles, and the origin is midway between the
  endplate centres. Before, the axis came from the body's principal axes or
  the neighbouring centroids, which tilted L3 by about 25° and L5 by about 40°
  (its lower neighbour was the sacrum's centroid). The L5 endplates themselves
  were already right: this subject's sacrum is steep, so the L5–S1 disc faces
  posteriorly-inferiorly.
- `assets/bones/s0751/landmarks.json` is now included, so builds do not need to
  detect landmarks (or scipy).

## [0.4.0] - 2026-10-30

### Added
- **Block canvas** in the viewer: the panel at the bottom shows the scene's
  block graph. Add blocks from a searchable palette (double-click the canvas),
  wire outputs to inputs by dragging (wires are typed and refuse wrong
  connections and loops), drag a wire off an input to disconnect, delete,
  undo/redo. The 3D view rebuilds as you connect blocks; values update it
  instantly. Edits are kept in the browser; Download / Open… / Reset, and Save
  to the file with the dev server.
- **Selecting bodies in 3D**: click a bone to highlight it and list every block
  that builds it (e.g. its smoothing) with their settings; selecting a block
  highlights the bodies it feeds.
- **Collapsible panel**: Tab hides or shows it for a full-window 3D view; drag
  its top edge to resize.
- The browser builds the shader of an edited graph itself (`viewer/graph.js`),
  from GLSL recorded once from the Python blocks (`blocks.json`) and a library
  of every bone of the subject (`<scene>.library.json`), so editing works on
  the static site. The GPU check also tests the browser-built shader, and CI
  runs it on a test graph that uses every block.
- Block bounds at the current values (camera framing, STL export, probes) in
  addition to bounds over the whole slider range (shader).

### Changed
- `examples/anatomy.json` (block graph) replaces `examples/anatomy.py`. The
  endplate markers of the old script are not shown for now; landmarks will come
  back as blocks.
- Position inputs of blocks (centres, offsets, plane points) slide ±50 mm
  around their value; typing a value outside re-centres the range.
- The viewer's settings moved from the right-hand column into the panel's Scene tab.

## [0.3.0] - 2026-10-30

### Added
- **Block graphs** (`implicit.graph`): a model can be a JSON graph of blocks
  joined by typed wires (body, bone, scalar, vector) instead of a Python script.
  This is the file format for the upcoming block canvas in the viewer.
  - Blocks: number, vector, bone; sphere, box, cylinder; union, intersect,
    subtract (with blend radius); move, rotate; offset, shell, plane cut,
    smooth; output body.
  - Every literal number or vector in a graph is a live parameter, so editing a
    value never rebuilds the shader; only rewiring does.
  - Body bounds are propagated through the graph and hold over each slider's
    whole range (tested), so the viewer can still skip far-away bodies.
  - Graphs work wherever scripts do: `implicit.serve`, `implicit.build`,
    `implicit.export`. `python -m implicit.graph check GRAPH` validates one;
    `python -m implicit.graph blocks` prints the block list for the editor.
- `examples/lumbar.json`: the bundled bones as a graph (bone → smooth → output
  per bone, one shared "keep volume" number).

## [0.2.1] - 2026-10-30

### Changed
- Documentation describes the method on its own terms. The primer's background
  section now cites the published literature on function representation and
  distance fields, and its meshing notes describe the current marching-tetrahedra
  mesher.

## [0.2.0] - 2026-10-29

### Added
- **Live surface smoothing of bones** (`implicit.sdf.smooth`, `GridSmooth`):
  Gaussian blur of the implicit field, with the width σ (mm) as a slider per
  bone. The blur is precomputed as a small
  pyramid (σ = 2, 4, 8 voxels, each at matching coarser spacing, about +15 %
  texture memory), so moving the slider is instant. The smoothed field stays
  1-Lipschitz, so it ray-marches and meshes like any body. A
  "keeps bone volume" switch (on by default) cancels the shrinkage that
  blurring causes on convex features. On L3 (0.4 mm mesh), σ = 1 mm cuts surface
  area 8 % (CT noise) at −0.4 % volume; σ = 2.5 mm cuts it 19 % at −1.1 %.
- **Robust meshing** (`implicit.mesh.mesh_sdf`): narrow-band marching
  tetrahedra. Meshes are closed and manifold by construction (every edge shared
  by exactly two triangles), including thin and touching features where surface
  nets were not. Only blocks that can contain the surface are sampled.
- `python -m implicit.export SCENE [--body NAME] [--set PARAM=VALUE] [--voxel MM]`:
  mesh any body at any slider values to STL, with volume, area and closure report.
  The dev server's "Export STL" uses the same mesher.
- The GPU check also tests every scene with all sliders moved off their defaults
  (twice the checks), so parameter-dependent shader paths are compared with NumPy.

## [0.1.3] - 2026-10-29

### Fixed
- Most bones invisible on Intel Iris Xe with Chrome/Direct3D 11 (only the
  femurs rendered). The grid lookup measured "distance outside the grid" as
  `length(q - clamp(q)) * spacing` in scaled grid coordinates. That compiler
  left a rounding residue there, so every point counted as outside the grid
  for bones whose spacing is not exactly 1 mm. The lookup now clamps and measures
  in world coordinates (exactly zero inside the grid). Diagnosed on the
  affected GPU with per-bone numeric probes and a ray-trace probe.

### Added
- **GPU check** (`diagnostics.html`, linked from the viewer's About section):
  on the visitor's own GPU, compares the shader distance of every body with
  NumPy at probe points inside, on and outside each body, and traces a ray to
  each body. It reports PASS/FAIL with a copyable report.
- CI runs the GPU check on three graphics stacks for every push and PR:
  Linux/SwiftShader, Windows/Direct3D 11 (WARP) and macOS/Metal
  (`ci/gpu_check.mjs`). `ci/check.sh` runs it locally when Playwright is installed.

### Changed
- Shader sources and WebGL helpers moved to `viewer/gpu.js`, shared by the
  viewer and the GPU check.
- The viewer asks for the high-performance GPU (`powerPreference`), so dual-GPU
  laptops prefer the discrete GPU where the browser allows it.

## [0.1.2] - 2026-10-29

### Fixed
- Bones rendered as dots or not at all on some GPUs. The shader sampled the
  bone texture with implicit level of detail inside a branch, which is undefined
  in WebGL2 (GLSL ES 3.0) and returns garbage on many drivers. It now uses an
  explicit level of detail and no branch around the fetch.

## [0.1.1] - 2026-10-29

### Added
- MIT License for the code (`LICENSE`). The bundled TotalSegmentator-derived
  anatomy stays under CC BY 4.0 (see `NOTICE.md`).
- Viewer About section links to the licence.

### Changed
- Live viewer URL in the docs: https://siril.me/implicit/

## [0.1.0] - 2026-10-29

First release.

### Added
- **Implicit modelling core** (`implicit/`): signed distance field node graph with
  named parameters (sliders), primitives, booleans and smooth blends, offsets and
  shells, half-space cuts and rigid transforms. One graph runs in Python (NumPy,
  for meshing) and in the browser (generated GLSL).
- **Browser viewer**: WebGL2 ray marcher with live parameter sliders, body
  toggles, section cut, progressive rendering, and GPU diagnostics. No JavaScript
  dependencies.
- **Patient anatomy, converted once to implicit bodies**:
  `implicit.anatomy.convert` turns TotalSegmentator masks and CT into one
  signed distance file per bone. The surface is placed on the CT half-maximum
  edge, with the label as a ±2 mm guard. Tests show the surface recovered to
  ≤ 0.12 mm on synthetic data with labels off by ±1.5 mm.
- Bundled subject `assets/bones/s0751`: L1–L5, sacrum, hips, proximal femurs
  (TotalSegmentator v2, CC BY 4.0).
- `implicit.anatomy.landmarks`: vertebral frames and endplate planes computed
  from the implicit bones. L5 is not yet reliable.
- Surface-nets mesher and STL export; local dev server with live reload.
- Idealised L4–L5 example with parametric cage and PSO (Phase 0 demo).
- CI on every push and PR; release on merge to `main` (tag, GitHub release,
  live viewer on GitHub Pages).
