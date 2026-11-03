# Changelog

All notable changes to this project are listed here, newest first.
Format: one `## [X.Y.Z] - YYYY-MM-DD` section per release; the release workflow
publishes that section as the GitHub release notes. Versions follow
[Semantic Versioning](https://semver.org/) (0.x while the API is still moving).

## [0.22.0] - 2026-11-03

### Changed
- **Discs rebuilt live from the bones** (roadmap 2.1, as discs are built for FE: rims
  joined, extended into the bones, cut by them). The Disc block now builds each disc
  in the shader from its two vertebrae as shown: the rims are each body's outline
  1.5 mm inside its endplate, sampled on the moved bone; the wall blends the two rims
  between the endplate planes; it reaches about 3 mm into both bones (tapering inward
  1 mm per mm, so no collar wraps the waisted bodies); then the bones as shown cut it.
  Whatever moves or reshapes the vertebrae (sliders, motion limits, posture), the disc
  follows, and its interface is the bone surface itself: tested equal at rest, within
  the motion limits and in an extreme posture with the limits off. Before, each disc
  was a grid built at rest and carried by the polyrigid field, which left gaps at
  concave endplates and drifted from them in large motions. The `disc_*.npz` files
  are gone; `python -m implicit.anatomy.intervertebral` now reports each level's
  Disc block inputs and rim areas.
- Rims from each body's outline with the body's depth and width (no endplate corners
  needed): **T1-T2 now has a disc** (T1's inferior endplate has no corners).
- Shorter shaders (faster compiles, notably with Direct3D): GLSL constants with 9
  significant digits (exact for 32-bit floats); a smoothed bone samples only the two
  pyramid levels it blends, and the shaders avoid `?:` (ANGLE turns each into an if
  block for Direct3D); the disc rims sample the bones as shown, so the viewer's own
  build of the graph gives the same discs.
- A Subtract keeps A's moving cull box (Python scenes).

## [0.21.0] - 2026-11-02

### Added
- **Intervertebral discs** (roadmap step 2.1): `python -m implicit.anatomy.intervertebral`
  builds a disc between every pair of neighbouring vertebrae with endplate corners
  (T2-T3 to L5-S1 on VIVA+ 50F) from the template at rest: its wall joins the two
  endplate rims (each body's outline 1.5 mm inside its endplate, within the ellipse
  through its corners, at most 15 % wider than the opposite endplate's: S1's corners
  reach the alae), and it ends at both bones, filling the endplate concavities only
  inside the rim. Written as bone files `disc_*.npz` with rim areas and frame.
- **Disc** block: nucleus (the disc scaled in its mid-plane to a fraction of its
  cross-section; defaults 0.22 from T10-T11 down, 0.11 above, from Moens et al. 2024)
  and annulus. Field warp: "polyrigid" carrier and a segment to move with, using a
  two-segment polyrigid inverse (`polyInv2`, stops once converged) and a cull box
  carried by the lower vertebra; the discs stay on both endplates in every posture
  (within 0.1 mm).
- Discs table: sagittal wedge angle per disc (between the endplate lines, as on a
  lateral radiograph), Python and browser.
- Annulus fibre field for FE (`intervertebral.fibres`): 20 lamellae, fibres 28 deg to
  the transverse plane at the outer wall to 45 deg at the nucleus, alternating
  (Cassidy et al. 1989; Marchand & Ahmed 1990).
- `ci/frame_time.mjs`: frame time of a scene with some groups shown and hidden.
- Literature rows 41-47.

### Notes
- Lumbar endplate rim areas 860-1,100 mm2, within the women's mean +- 2 SD of Louie
  et al. 2018 (1,819 +- 502 mm2, 3D surface area). Thoracic areas are not checked yet
  (Panjabi et al. 1991: full text not read).
- The discs are hidden by default: shown, a full-resolution frame takes about 3.7
  times as long in headless SwiftShader; the cost is the extra geometry (static
  discs cost the same). Roadmap 2.1's budget changed accordingly.
- The nucleus defaults come from children's volume fractions (no adult area fraction
  found); the roadmap's "40-50 % of the area" had no source.

## [0.20.0] - 2026-11-01

### Added
- **Global balance** (roadmap step 1.8): SV (T1 anterior of the posterior superior
  corner of S1: the sagittal vertical axis measured from T1, as the cervical spine is
  not modelled), the T1 pelvic angle (TPA, Protopsaltis et al. 2014) and PI - LL,
  measured live with CB and listed under **Balance** in the Scene tab.
- Compensation modes (SV and CM sliders): the body holds T1 at the SV target by
  pelvic retroversion (PT solved) or thoracic flattening (TK solved), as patients do
  (Barrey et al. 2011), solved by regula falsi (Illinois) in Python and the browser,
  within the motion limits when they are on; the SV shortfall is reported. As in
  Protopsaltis et al., pelvic retroversion leaves TPA unchanged.
- **Sagittal shape** presets (Scene tab): Roussouly types 1, 2, 3, 3 AP and 4 with the
  mean PI and PT of each type (Laouissat et al. 2017) and LL from PI (Le Huec &
  Hasegawa 2016); a preset sets ML = 0 (another person's shape, not a posture).
- Literature rows 34-40.

### Notes
- VIVA+ 50F stands at SV 15.9 mm, TPA 14.3 deg. LL 35 deg moves T1 about 100 mm
  forward; pelvic retroversion holds it with PT about 35 deg; thoracic flattening
  alone cannot (TK reaches 0), and with the limits on TK can change only about 7 deg.
- Types 1 and 2 differ by lordosis shape, which the sliders do not set: their presets
  differ only in PI and LL.

## [0.19.0] - 2026-11-01

### Added
- **Constrained distribution** (roadmap step 1.7): every disc's feasible range per
  plane and direction (`implicit/anatomy/ranges.py`, stored with the landmarks as
  "range"): how far it may turn about its rotation centres before a pair of bones
  across it (vertebrae, facets, ribs) comes closer than min(1 mm, its rest gap) or a
  disc rule fails (height change, thickness floor), within half its range of motion.
  Each constraint involves one disc only, so the spine's feasible set is the product
  of the discs'; the distribution (water-filling: the weighted least-squares solution
  closest to the subject's own distribution) now uses the tighter bound in each
  direction, and the planes of a disc share its range (|FE|/cap + |LB|/cap +
  |AR|/cap <= 1).
- PI kept where the sacrum stays clear of the hip bones (the sacrum's turn in the
  pelvis, about +-12 deg on VIVA+ 50F; "pi_range" with the pelvis landmarks); PI
  shortfall reported.
- Viewer: each disc at a limit says what stops it (range of motion, clearance pair,
  disc rule) in the Motion limits tooltip.
- `tests/test_constrained.py`: no collision and no flagged disc on a grid and at
  random settings of every slider (limits on).

### Changed
- Roadmap 1.7 criteria as built: a target that cannot be met is drawn at the nearest
  feasible posture and reported (it was "not drawn").

### Notes
- On VIVA+ 50F the facets bound L2-L3 and L4-L5 extension-side turns and lumbar axial
  rotation (10-20 % inside the range of motion), L5-S1 lateral bending and rotation,
  and T3-T4 lateral bending; the thickness floors bound T5-T6 and T8-T9 to about 0.2
  deg one way (the template opened them to their floors).
- The distribution takes 6.8 ms per slider move in the browser solver (Node); the
  full solver result with the field tables and flow lines takes about 170 ms, as
  before this step.

## [0.18.0] - 2026-11-01

### Added
- **Connected rib cage** (roadmap step 1.6): the sternum is a rigid segment, the
  least-squares fit (Horn 1987, unit quaternions) to where the ribs carry the ends
  of ribs 1-7; the costal cartilages bend between ribs and sternum by a new rib
  cage field (Field warp, "Carried by: rib cage"): a thin-plate spline through 5
  points of each costochondral junction (ribs 1-10) and the sternocostal contacts.
  Within the motion limits no junction opens or overlaps by more than 0.5 mm on a
  test grid (about 0.1 mm typically); at extreme settings with ML = 0 up to about
  1.2 mm (TK 60). Python and browser identical.
- Rib cage landmarks (`implicit/anatomy/ribcage.py`, "ribcage" in landmarks.json):
  costochondral and sternocostal junction samples.
- Viewer: a **Rib cage** list in the Scene tab: the sternum's motion and each true
  rib's cartilage length change (rib end to sternum) of 0.5 mm or more.
- Literature: rows 27-33 (costal cartilage properties, the rib cage's effect on
  thoracic stiffness, Horn's method).

### Changed
- Template: the costal cartilages are joined to ribs 1-10 and the sternum at rest
  (`template.close_junctions`): a cartilage end short of its rib is moved to it by a
  local morph (25 mm reach, 2.5 mm into the rib), then every cartilage is cut at the
  rib and sternum surfaces, so they share the junction surface. VIVA+ 50F had gaps
  of 0.3-1.5 mm (left) and 1.7-2.7 mm (right), and rib 1 left overlapped its
  cartilage by 4 mm.
- Roadmap: step 1.9 (cervical spine and head) paused; the trunk comes first.
- Template: a rib takes only the shifts its vertebra takes. Earlier runs had given
  ribs 1-10 a 0.05 mm shift their vertebrae did not take (below the 0.1 mm
  threshold); they are moved back, and shifts not applied are no longer recorded.
- Every PR checks all `.md` files before merging (PR template, CONTRIBUTING);
  LICENSING lists Playwright (CI) and how literature values are used; the VIVA+
  derived files' modifications are listed in LICENSING, NOTICE and the bones README.
- `alignment.chain` takes the rib cage landmarks; the solver gives a "sternum"
  segment, a "tps_ribcage" table and a rib cage report (in "limits").

### Notes
- No in vivo data on costal cartilage deformation or sternum motion with spinal
  posture were found; the sternum following the mean motion of the true rib ends is
  an assumption, and cartilage length changes are reported, not limited.
- Denser junction control points made the spline worse (close points with
  different motions condition it badly); 5 spread points per junction are used.

## [0.17.0] - 2026-11-01

### Added
- **Motion limits and coupling** (roadmap step 1.5): per-level ranges of motion
  (flexion-extension, lateral bending, axial rotation) from Wilke et al. 2025
  (JOR Spine, doi:10.1002/jsp2.70052; medians, thoracic levels with the rib cage,
  which agree with in vivo data), in `implicit/anatomy/motion.py` and stored with
  each disc of the chain data. With the new **ML** slider on (default in the Spine
  alignment block) no disc turns from rest beyond half its range either way; LL,
  TK, MT, TL and the coronal balance are spread over their discs as before until a
  disc reaches its limit, then over the others (water-filling: continuous in the
  sliders). The apex rotation (AR) is scaled down where it would exceed a disc's
  axial range.
- **CP** slider: thoracic lateral bending carries ipsilateral axial rotation,
  0.40 deg per deg at CP = 1 (6.2 deg per 15.6 deg over T1-L1, Fujimori et al.
  2014, doi:10.1016/j.spinee.2013.11.054); none at lumbar levels (in vivo coupling
  there is small and changes direction between levels).
- Viewer: a **Motion limits** list in the Scene tab gives each target that is not
  reached and by how much (deg; CB in mm), with the discs at a limit in the
  tooltip. The solver result has a `limits` entry (`short`, `at_limit`), the same
  in Python and the browser.
- `docs/LITERATURE.md`: every source read so far (26 rows): what it reports, what
  we concluded or assumed, where it is used, and whether it is used, considered or
  superseded. Updated in every step (CONTRIBUTING.md).

### Changed
- Docs brought up to date: README (project status, rib cage in the template, its
  drift from the rib ends with large slider changes, licences of both bundled
  subjects), NOTICE (derived rib cage files), primer (architecture, Python/JS
  parity checks), roadmap (goal, step numbers).
- Roadmap: new step 1.6 **Connected rib cage** (ribs, costal cartilages and
  sternum stay joined in every posture: rigid sternum fitted to the rib ends,
  cartilages deformed between rib and sternum); constrained distribution, global
  balance and cervical spine move to 1.7-1.9.
- `alignment.solve` takes `ML` and `CP` (both off by default there; the graph
  block defaults to on). The slider smoothness test runs with the limits off and
  on, and accepts a steeper stretch that splits into proportionally smaller steps.

### Notes
- The ranges are small: from the template, T4-T12 allow about +-6.8 deg of TK,
  L1-S1 about +-21.9 deg of LL, MT about +-7.8 deg, TL about +-15.4 deg, and the two
  lumbosacral discs below the curves about 8 deg of balancing bend. Posture
  differences between people beyond these come from vertebral and disc shape
  (roadmap 3.2, 3.3). Splitting each range equally into its two directions is an
  assumption (standing is not the middle of the range in vivo).

## [0.16.0] - 2026-10-31

### Added
- **Rib cage** (roadmap step 1.4, moved before motion limits): ribs 1-12 (both
  sides), sternum and costal cartilages of VIVA+ 50F (`--select full`; ribs and
  cartilages on 1.5 mm grids, the sternum on 0.8 mm; cartilages from their solid
  elements, whose boundary is closed). Each rib follows its vertebra with the
  alignment sliders; the sternum and cartilages are carried by the soft-tissue
  field. The template correction carries them (ribs with their vertebra's shift,
  sternum and cartilages with the rib ends) and carves the costovertebral joints
  to 1.3 mm (the thoracic facet joint space; assumption: the same for these
  synovial joints). Parts added after the vertebrae were shifted take the recorded
  shifts (`template.json`).
- Clearance: every rib against its vertebra and the one above, and against the
  next rib on its side (Python and viewer).

### Changed
- Roadmap: rib cage is step 1.4; motion limits, constrained distribution, global
  balance and cervical spine are 1.5-1.8; step 2.7 is rib motion (pump- and
  bucket-handle) with collision-free ribs.

### Notes
- At rest no rib touches a vertebra or another rib. MT 40 crowds the concave-side
  ribs 5-8 into each other (up to 2.7 mm); TK 60, TL -40 and LL 80 leave them clear.

## [0.15.0] - 2026-10-31

### Added
- **Disc thickness over the whole disc** (`implicit.anatomy.discs.thickness`, and
  `viewer/discs.js` from the atlas, same to 0.01 mm): the smallest distance from any
  surface point of either endplate to the other vertebra's bone, live with the
  sliders (Discs table: A / M / P · thinnest). Each disc has a floor: the population
  lower bound of its anterior and posterior heights (mean - 2 SD), at least 1.5 mm
  (the thinnest such bound of any level, Kunkel T4-T5 anterior). A disc thinner
  than its floor is red; e.g. TK 70 thins T4-T5, T5-T6 and T8-T9 to 1.3-1.5 mm.

### Changed
- **VIVA+ 50F template at population values** (`implicit.anatomy.template`, run by
  the VIVA+ converter):
  - Disc spaces: the thoracic and lumbar vertebrae are translated along their disc
    axes so that every measured disc space takes the population mean middle height
    (Shin et al. 2024; Kunkel et al. 2011), and no disc is thinner anywhere than its
    floor at rest. Translations only: orientations, Cobb angles and pelvic
    parameters unchanged (within 0.05 deg, the endplate refit).
  - Facet joints: VIVA+'s lumbar facets nearly touch (0.3 mm) and collide once the
    upper lumbar discs are closed. Every pair of neighbouring bones closer than its
    joint space is carved with an implicit Boolean cut back from the surface midway
    between them (f = max(f, (f - f_other + g) / 2) on both bones; 1-Lipschitz, so
    the bones end at least g apart, each losing only what that needs): g = 2.0 mm in
    the lumbar joints (lower normal limit on CT, Weishaupt et al. 1999 via Tiwari et
    al. 2021), 1.3 mm in the thoracic (mean on CT, Tiwari et al. 2021); room for the
    cartilage (0.57 mm a side, Woldtvedt et al. 2011). Carved: T12-L1 to L5-S1,
    16-326 mm3 a joint; gaps 2.00-2.03 mm on the exact surfaces.
  - Middle heights before -> after (mm): L1-L2 11.8 -> 6.4, L2-L3 11.1 -> 7.6, L3-L4
    8.2 -> 8.5, L4-L5 7.9 -> 9.0, L5-S1 8.5 -> 8.1 (the population means); thoracic
    discs opened 0.2-2.1 mm to the Kunkel means, T5-T6 and T8-T9 a little further,
    to their thickness floors. Thinnest point of any disc at rest: 2.4 mm (T3-T4,
    floor 2.2). Shifts and carving recorded in `template.json` and each bone file.
- **Smooth sliders**: the coronal balance (CB) jumped by up to 400 mm in one slider
  step (TL below -42 deg, MT beyond -80 deg, and extreme combinations): with only the
  discs below the TL curve (L4-L5, L5-S1) to balance, the solve ran out of its +-30
  deg interval and the bisection flipped to the opposite end. Now the balance is
  solved on the lean of the S1-T1 line (monotonic in the bend up to a half turn), on
  its monotonic branch around no bend, and the bend is exact up to 20 deg, easing
  to at most 30 deg beyond (tanh): every vertebra moves continuously and without
  kinks for every slider (largest second difference 0.06 mm per 0.45 deg step).
  Same in `viewer/solvers.js`; a test sweeps every knob and random paths.

### Notes
- LL 80 / TK 60: L4-L5 overlap 0.86 -> 0.07 mm, L3-L4 clear (was 0.10 mm).
- The sampled clearance (step 1.1) reads up to ~0.3 mm low (surface samples within
  half a voxel of the surface): carved joints show 1.75-1.85 mm there.

## [0.14.0] - 2026-10-31

### Added
- **Disc space heights** (roadmap step 1.3, new): anterior, middle, posterior,
  left and right heights of every disc space along the disc axis, from endplate
  corners and centres detected on the bone fields (stored in `landmarks.json`;
  endplates fitted worse than 1.5 mm rms, T1 of VIVA+ 50F, are skipped). Computed
  in Python (`implicit.anatomy.discs`, with a slider sweep) and the viewer
  (`viewer/discs.js`, same to 0.01 mm, checked by the GPU check); a Discs table in
  the viewer.
- Posture rules from in vivo data (Xi et al. 2025): middle height within +-10 % of
  rest, anterior/posterior/left/right within +-40 %; discs beyond them are red.
- Rest heights against population values, mean +- 2 SD (lumbar: Shin et al. 2024,
  CT, 350 women; thoracic: Kunkel et al. 2011): outliers marked *.

### Changed
- Roadmap: disc space heights inserted as step 1.3; motion limits, constrained
  distribution, global balance and cervical spine renumbered 1.4-1.7; the
  constrained distribution (1.5) takes the disc rules as constraints.

### Notes
- Average spine at rest: L1-L2 and L2-L3 are about twice the population height
  (middle 11.8 and 11.1 mm against 6.4 and 7.6 mm); L3-L4 to L5-S1 middle heights
  are within the population range, their anterior heights above it.
- TK 60 compresses T4-T5 and T5-T6 anteriorly by 46-48 %; LL 80 raises the L5-S1
  middle height by 10 %. Over the 243-setting sweep (LL, TK, PT, MT, TL), 237
  settings deform a disc beyond the rules (L5-S1 198, T7-T8 162, T12-L1 162,
  L4-L5 138, T4-T5 81, T5-T6 81, T8-T9 27); the coronal curves of +-40 deg compress
  the ends of their curves laterally.

## [0.13.0] - 2026-10-31

### Changed
- **Physiological rotation centres** (roadmap step 1.2): each disc of the alignment
  chain turns about its instantaneous axis of rotation instead of the disc centre,
  one point for sagittal and one for coronal turns (`implicit/anatomy/rotation.py`,
  with references). Lumbar flexion-extension: just below the lower vertebra's
  superior endplate, at the posterior third of the disc (Pearcy & Bogduk 1988);
  thoracic: on the antero-posterior centre of the lower vertebra, between its
  superior endplate and its centre (Qiu et al. 2003, 2004); lumbar lateral bending:
  the disc centre (Dimnet et al. 1978). Targets are met exactly as before.
- `landmarks.json` stores each disc's centres ("rotation", on the upper vertebra);
  landmarks regenerated for both subjects (the other values are unchanged).
- **Faster first load**: groups hidden when a scene loads (landmark markers, field
  views) are left out of the shader and compiled in when shown. The landmark
  markers were three quarters of the average spine's shader code; its viewer
  shader took 9-15 s to compile on the Windows (D3D11) CI runner, close to the
  15 s limit. The GPU check also compiles each scene with every group shown.

### Notes
- LL 80 / TK 60: L4-L5 overlap 1.70 -> 0.86 mm, L3-L4 0.27 -> 0.10 mm, L2-L3 now
  clear. Sagittal-only sweep settings: worst overlap 1.72 -> 0.88 mm. Full sweep
  (with coronal curves of +-40 deg): still 225 / 243 settings collide; worst
  overlap 3.94 -> 3.05 mm, summed overlap -5 %, more at T11-T12 and L1-L2. Motion
  limits and the constrained distribution (steps 1.3, 1.4) remain needed.

## [0.12.0] - 2026-10-31

### Added
- **Clearance** between neighbouring bones (roadmap step 1.1): for each pair
  (consecutive vertebrae, lowest vertebra and sacrum, sacrum and L5 with each hip
  bone, hip bone and femur) the smallest value of one bone's field at the other
  bone's surface points, both ways, at the current posture. Computed in Python
  (`implicit.anatomy.clearance`, with a slider sweep) and in the viewer from the
  same atlas values (same result to 0.01 mm, checked by the GPU check); shown as a
  table in the viewer, colliding bones tinted red.
- Library JSON: per bone its place in the atlas and its surface points.

### Notes
- The average spine is clean at its measured posture (smallest gap 0.4 mm, lumbar
  facets). At LL 80 / TK 60, L2-L3, L3-L4 and L4-L5 overlap (L4-L5 by 1.7 mm).
  Over a 243-setting sweep (LL, TK, PT, MT, TL), 225 settings collide; the sliders
  will avoid this in steps 1.2-1.4.

## [0.11.1] - 2026-10-31

### Added
- `docs/ROADMAP.md`: the roadmap to virtual patients in six phases (posture, soft
  tissue, morphology, generator, realism, FE), one objective and acceptance
  criteria per step; each step is one pull request.

## [0.11.0] - 2026-10-31

### Added
- **Polyrigid soft-tissue field** as an alternative to the thin-plate spline
  (choice on the Spine alignment block): rigid inside each vertebral body, the
  screw motion between neighbouring vertebrae in each disc space (log-Euclidean
  blend by position between the endplate planes); inverted in the shader by
  fixed-point iteration.
- **Field view** block: the soft-tissue field as flow lines (paths of tissue points
  near the spine while the sliders go from the subject's values to the current
  ones, coloured by the distance travelled; drawn from a grid of cells built on the
  CPU, exact near the lines), a lattice it carries, arrows of the displacement on a
  grid, or a slice coloured by the displacement, with a colour legend in the
  viewer. The average spine scene has all four as hidden groups.
- Bodies coloured by a scalar field (per-point colour), used by the field views.

### Changed
- The TPS warp block is now Field warp and uses the chosen field.
- Every third-party dependency is pinned to one exact version, transitive ones
  included: numpy 2.5.3, scipy 1.18.1, nibabel 5.4.2, packaging 26.3, pip 24.0;
  Playwright 1.50.1 with a lockfile (`ci/package-lock.json`, `npm ci`); Python
  3.13.17 and Node 22.22.0, the GitHub Actions by release and the runner images
  (ubuntu-26.04, ubuntu-24.04, windows-2025, macos-15) in CI (rule in
  CONTRIBUTING.md).

## [0.10.0] - 2026-10-31

### Added
- **Coronal alignment** of the average spine: main thoracic and thoracolumbar/
  lumbar Cobb sliders (curves with chosen end vertebrae and apex, + convex right),
  apical axial rotation toward the convexity (gain per degree of Cobb), pelvic
  obliquity (pelvis and femurs about the antero-posterior axis through the hips)
  and coronal balance (solved on the lumbosacral discs below the curves).
- **TPS warp** block: soft structures follow a 3-D thin-plate spline (kernel |r|)
  fitted to control points on every bone, which move rigidly; the spine curve is
  now the subject's curve carried by it. The warp's table is a computed texture
  (refilled on the CPU when a slider moves).

### Changed
- Bones move by full rigid motions (rotation and translation, 12 computed
  uniforms per segment) instead of sagittal ones; the femurs are a segment
  ("legs"). The Spinopelvic alignment block is now Spine alignment.
- Bounds of moved bodies sample each slider alone at its ends (instead of every
  combination); moved bodies are kept out of cluster boxes.

## [0.9.0] - 2026-10-30

### Added
- **Spinopelvic alignment**: PI, PT, LL and TK sliders reshape the spine and pelvis
  in the sagittal plane (chain model: pelvis about the hip axis, sacrum about the
  sacroiliac joints, lordosis and kyphosis spread over their discs in proportion to
  their angles; targets met exactly). New blocks: Spinopelvic alignment, Align,
  Spine curve (tube through the vertebral body centres). Both example scenes use
  them; the sliders start at the subject's measured values.
- `python -m implicit.anatomy.alignment <bones>` prints PI, PT, SS, LL, TK and the
  disc angles. `landmarks.json` gains "pelvis": femoral head centres (sphere fits,
  rms 0.6 mm) and sacroiliac joint centres.
- Computed uniforms: values a solver computes on the CPU from sliders (Python
  `expr.Solver`/`Computed`, browser `viewer/solvers.js`), checked against each
  other by the GPU check with the sliders moved.

### Changed
- Moved bodies are culled with their own box carried along by the motion, and the
  viewer marches rays only through the visible bodies at the current values: the
  average spine renders about 2.5x faster than in 0.8.
- A shared-code group with one visibility switch tests it once, outside its loop
  (hidden landmark markers cost nothing on software renderers either).

### Fixed
- Bone fields outside their grid: the larger of two lower bounds, continuous at the
  grid faces (a jump there could put GPU and NumPy on different sides).

## [0.8.2] - 2026-10-30

### Fixed
- GPU check: probe programs call each shared shader function once, and bodies
  sharing code are probed in one program (the Windows check took ~10 of its
  12 minutes).

## [0.8.1] - 2026-10-30

### Fixed
- The average spine took 40+ s to compile on Windows (Chrome/Edge translate
  WebGL to Direct3D, whose compiler inlines every function at every call site)
  and often never loaded. Bodies built alike (vertebrae, their markers, paired
  bones) now share one shader function, called from one loop; their numbers come
  from a small float texture and their parameters from a uniform array. The
  shader is 26 kB instead of 259 kB, with the same image, and renders faster.
- GPU check: fails when a viewer shader takes longer than 15 s to compile.

## [0.8.0] - 2026-10-30

### Added
- **Average (idealised) spine**: `assets/bones/vivaplus-50F`, T1–T12, L1–L5,
  sacrum, both hip bones and proximal femurs of the VIVA+ open human body model
  (average female, standing; v2.0.2, LGPL-3.0-or-later). It is the default
  scene of the viewer (`examples/average_spine.json`), with landmarks and a
  pose block per vertebra like the patient scene.
- `python -m implicit.anatomy.vivaplus`: downloads VIVA+ once and converts its
  bone surfaces (cortical shells; femurs: outer boundary of the solid elements)
  to implicit bone files, by ray-parity inside/outside tests along x, y and z
  (majority vote) on a 0.25–0.6 mm grid and the existing distance step.

### Changed
- The patient scene is now `examples/patient.json` (was `anatomy.json`).
- Block graphs from `anatomy_graph` list vertebrae cranial to caudal.

### Fixed
- Viewer: the render shader evaluates the scene from one place (march, normal
  and ambient-occlusion samples in one loop), so drivers that inline every call
  (D3D) compile large scenes in reasonable time. Same image.
- Landmarks: without a spinal-cord file, the posterior wall is found from the
  vertebral foramen, so pedicles no longer stay in the isolated body. L5 uses a
  larger opening radius (7 mm) to drop its transverse processes; this corrects
  the L5 inferior endplate (was tilted up to 20°) and hence S1, also for
  `s0751` (landmarks regenerated).
- GPU check: probes read their point and body from a data texture through one
  dispatch function, so large scenes (40 bodies) no longer read back zeros;
  at most 8 bodies per probe program (keeps D3D compile times short), and
  progress is printed to the CI log.

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

### Changed
- Smaller shaders: vector arithmetic is emitted as whole-vector GLSL instead of
  one expression per component, and moved planes and frames are calls to
  `posePoint`/`poseDir` helpers. The anatomy shader is 87 kB instead of
  620 kB with the expanded form, which Direct3D compiled too slowly.

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
