# Changelog

All notable changes to this project are listed here, newest first.
Format: one `## [X.Y.Z] - YYYY-MM-DD` section per release; the release workflow
publishes that section as the GitHub release notes. Versions follow
[Semantic Versioning](https://semver.org/) (0.x while the API is still moving).

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
