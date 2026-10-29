# Changelog

All notable changes to this project are listed here, newest first.
Format: one `## [X.Y.Z] - YYYY-MM-DD` section per release; the release workflow
publishes that section as the GitHub release notes. Versions follow
[Semantic Versioning](https://semver.org/) (0.x while the API is still moving).

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
