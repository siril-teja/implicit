# Third-party license policy

**Requirement:** every third-party dependency must allow commercial use.
This is an engineering policy, not legal advice. Have counsel review it before
commercial release.

Nearly all open-source licenses (GPL included) *permit* commercial use. What
differs is the **obligations** they place on you, so the policy has three tiers:

| Tier | Licenses | Rule |
|---|---|---|
| ✅ Allowed | MIT, BSD-2/3, Apache-2.0, ISC, Zlib, PSF, Boost, Unlicense/CC0 | Use freely. Keep the notices. |
| ⚠️ Review | MPL-2.0, LGPL-2.1/3.0, EPL-2.0 | OK if used unmodified and as a separate library (dynamic link or separate process). Modifications to *those files* must be published. |
| ❌ Avoid | GPL-2/3, AGPL-3, SSPL, "non-commercial" (CC BY-NC, academic-only) | Not in shipped code or a hosted service |

## Current dependencies

| Component | License | Tier |
|---|---|---|
| Python standard library | PSF | ✅ |
| NumPy | BSD-3 | ✅ |
| Viewer: raw WebGL2, no JS libraries | n/a (own code) | ✅ |
| GitHub Actions (checkout, setup-python, pages actions) | MIT | ✅ (CI only, not shipped) |

## Candidates for upcoming phases

Each license below was stated from memory or secondary sources. Re-check the
upstream LICENSE file before adopting any of them.

| Need | Candidate | License | Tier |
|---|---|---|---|
| NIfTI I/O | nibabel | MIT | ✅ |
| Distance transforms | SciPy | BSD-3 | ✅ |
| Marching cubes (alternative) | scikit-image | BSD-3 | ✅ |
| Mesh I/O | meshio | MIT | ✅ |
| Mesh utilities | trimesh | MIT | ✅ |
| Robust mesh booleans | Manifold | Apache-2.0 | ✅ |
| Tet meshing | fTetWild | MPL-2.0 (verify it has no GPL CGAL parts) | ⚠️ |
| Tet meshing | Netgen | LGPL-2.1 | ⚠️ |
| Remeshing | MMG | LGPL-3 | ⚠️ |
| Tet meshing | TetGen | AGPL-3 (commercial licence sold separately) | ❌ |
| Meshing | Gmsh | GPL-2+ with exceptions | ❌ in shipped code. As a user-installed external tool, ask counsel. |
| Geometry | CGAL (Mesh_3 etc.) | GPL-3 / commercial | ❌ |
| 3D web viewer (if ever needed) | three.js | MIT | ✅ |

The FE solver (MSC Marc) is your own commercial licence and is not a dependency
of this code. We only write input decks.

## Data: VerSe

The VerSe CT and segmentation data is **CC BY-SA 4.0** per the
[OSF project](https://osf.io/t98fz/wiki). An older challenge page says CC BY-SA 2.0.
Check `license.txt` on OSF before redistributing.

- Commercial use is permitted, with attribution (cite Löffler 2020, Liebl 2021,
  Sekuboyina 2021).
- **ShareAlike:** geometry derived from VerSe that you *distribute* (for example
  SDF grids or meshes published on GitHub Pages) must carry CC BY-SA 4.0 too.
  This does not extend to the code. Keep data out of the repo and download it
  with a script.
- For proprietary commercial models, use your own institution's imaging (with
  ethics/IRB clearance) instead of VerSe-derived geometry.
