# Roadmap: virtual patients

**Goal.** Generate virtual patients: thoracic spine to femurs (later cervical spine,
head and rib cage) with realistic bones, discs, ligaments and muscles, that look like
real radiographs and CT scans and are ready for finite element analysis.

**How this roadmap is used.** Each step below is one pull request, taken one at a time
in the order of the critical path. A step is closed when every "Done when" item holds
and CI is green. Steps may be split into several PRs when that keeps each one small,
but never merged half-done. The Status column is updated in the PR that closes a step.

**Principles** (apply to every step)

1. Three layers, kept separate: **morphology** (bone shape), **posture** (rigid
   kinematics of the bones) and **soft tissue** (derived from the posed bones). A
   slider in one layer never silently changes another.
2. Every parameter has a clinical or anatomical definition and is measured back on
   the generated geometry with that definition.
3. Validity is checked, not assumed: no bone overlap, positive disc heights,
   ligament lengths and balance within stated limits.
4. Everything is FE-ready by construction: clean geometry, region labels, landmarks,
   densities.
5. Every new block works in Python and in the browser, and the GPU check compares the
   browser-built shader with NumPy; third-party code and data allow commercial use;
   dependencies are pinned (CONTRIBUTING.md).

**Already in place (0.1 to 0.11):** implicit bones and the block graph; viewer with
the block canvas; landmarks (vertebral frames, endplates, femoral heads, sacroiliac
joints); the average spine (VIVA+ 50F, T1 to femurs); sagittal and coronal alignment
sliders (PI, PT, LL, TK, MT, TL, AR, PO, CB); TPS and polyrigid soft-tissue fields;
field views (flow lines, lattice, arrows, slice).

---

## Phase 1: Physiological posture

The alignment sliders must only produce postures a real spine can have.

| Step | Objective | Done when | Depends on | Status |
|---|---|---|---|---|
| **1.1 Clearance measurement** | Measure, for any slider setting, the minimum gap between every pair of neighbouring bones (vertebral bodies, facets, spinous processes, sacrum with L5, hip bones with sacrum) and show where they collide. | Clearance per pair computed on the CPU (Python and browser, same result to 0.01 mm) from the bone fields; colliding pairs drawn in red; a clearance table in the viewer; a test reproduces a known collision (e.g. LL 80, TK 60) and a known clean posture (baseline); collision counts over a grid of slider settings reported in the PR. | none || Done (0.12.0) |
| **1.2 Physiological rotation centres** | Turn each segment about its physiological instantaneous axis of rotation instead of the disc centre. | Rotation centre per level from cited literature (lumbar: Pearcy & Bogduk 1988; thoracic: cited source), stored with the landmarks; targets still met exactly; facet/spinous collisions at the 1.1 test settings reduced, with numbers in the PR. | 1.1 | |
| **1.3 Motion limits and coupling** | Keep each level within its physiological range of motion, with the coupled motions of its region. | Per-level limits for flexion/extension, lateral bending and axial rotation, and coupling coefficients per region, from cited tables (White & Panjabi); the distribution never exceeds a limit; when a target needs more than the limits allow, the viewer says which target and by how much. | 1.2 | |
| **1.4 Constrained distribution** | Distribute every target over the levels so that targets are met, limits respected and no bones collide. | A small optimisation (levels x 3 angles; targets as equalities, limits as bounds, minimum clearance, ~1 mm at facets, as inequalities, closest to the subject's own distribution) solved identically in Python and the browser at interactive speed (< 50 ms per slider move); no collisions on a grid of slider settings that is feasible; infeasible settings reported, not drawn. | 1.3 | |
| **1.5 Global balance** | Express and control whole-body sagittal and coronal balance. | SVA, T1 pelvic angle (TPA), C7/T1 plumb line measured and shown; compensation modes (pelvic retroversion, thoracic flattening) selectable; Roussouly type presets; measured values match targets within 1 mm / 0.5 deg. | 1.4 | |
| **1.6 Cervical spine and head** | Complete the spine up to the skull. | C1-C7 and the skull imported from VIVA+ 50F with landmarks; cervical lordosis (C2-C7) and T1 slope sliders; the 1.1-1.4 machinery covers the cervical levels. | 1.4 | |

## Phase 2: Soft tissue

| Step | Objective | Done when | Depends on | Status |
|---|---|---|---|---|
| **2.1 Discs** | Generate each intervertebral disc between its posed endplates, with annulus and nucleus. | Endplate rims detected on every level (areas within published ranges); disc filling the space between the two endplates and conforming to both; nucleus as a fraction of the disc (knob, default 40-50 % of the area); annulus lamellae with fibre directions (+-30 deg alternating) available for FE; disc height (anterior, posterior, mean) and wedge angle measured live; discs follow every slider; GPU check covers them; frame time within the budget of the scene without discs + 30 %. | 1.4 | |
| **2.2 Disc degeneration** | Represent disc degeneration as clinical grades. | Pfirrmann-style grade per disc controlling height loss and nucleus size; osteophytes at endplate rims as a size knob; grade measured back (height ratio) matches the grade definition. | 2.1 | |
| **2.3 Robust soft-tissue field** | Make the soft-tissue field fold-free everywhere soft tissue can be. | No folding (Jacobian > 0) within 60 mm of the spine curve at every feasible slider setting (sampled test); inverse accurate to 0.01 mm there; cost per sample not above the current polyrigid field; TPS and polyrigid kept as options. | 1.4 | |
| **2.4 Posterior element landmarks** | Detect the posterior bony landmarks every ligament and joint needs. | Spinous process tips, laminae, transverse process tips, pedicles and superior/inferior facet surfaces found on every level from the bone fields; stored in landmarks.json; shown as markers; checked against the VIVA+ mesh parts where they exist (error reported in the PR). | none | |
| **2.5 Facet joints** | Model the facet joints as cartilage layers with a joint gap. | Cartilage layers (thickness knob) on the detected facet surfaces, following the posture; joint gap measured live; facet orientation (tropism) measured per level. | 2.4, 1.4 | |
| **2.6 Ligaments** | Generate the spinal ligaments between their attachment sites. | ALL, PLL, ligamentum flavum, interspinous, supraspinous, intertransverse and capsular ligaments per level, from the 2.4 landmarks, carried by the soft-tissue field; as volumes for display and as 1-D elements (with rest length and prestrain) for FE; lengths measured live, outside physiological ranges flagged. | 2.3, 2.4 | |
| **2.7 Rib cage** | Add the ribs, sternum and costal cartilages, moving with the thoracic spine. | Ribs, sternum and costal cartilages imported from VIVA+ 50F with costovertebral and costotransverse joint landmarks; ribs follow their vertebra, with pump-handle and bucket-handle motion as knobs; no rib collisions at feasible slider settings (1.1 machinery). | 1.4, 2.4 | |
| **2.8 Muscles** | Add the main trunk muscles. | Erector spinae, multifidus, psoas, quadratus lumborum and the abdominal wall as volumes and lines of action, attached to landmarks and carried by the posture; source of geometry stated and licence-checked. | 2.6, 2.7 | |

## Phase 3: Morphology

| Step | Objective | Done when | Depends on | Status |
|---|---|---|---|---|
| **3.1 Size and sex** | Generate patients of different size and sex. | VIVA+ 50M converted and landmarked as a second template; stature scaling (uniform and per-region factors) as a slider; measured vertebral dimensions follow published size relations within stated tolerances. | none | |
| **3.2 Vertebral deformities** | Shape individual vertebrae with the deformities seen in patients. | Per-vertebra blocks for body wedge angle, height loss by Genant grade and type (wedge, biconcave, crush; 20-25 / 25-40 / > 40 %), endplate concavity and Schmorl nodes, osteophytes, facet tropism; each measured back with its clinical definition; landmarks and endplates updated analytically; posture and discs adapt. | 2.1 | |
| **3.3 Scoliotic wedging share** | Split each scoliotic curve between vertebral and disc wedging, as in patients. | A "vertebral share" knob per curve puts that fraction of the Cobb change into body wedging (local wedge warp of the bodies only) and the rest into the discs; Cobb angles still met; share measured back. | 3.2 | |
| **3.4 Bone quality** | Represent bone density and cortical thickness. | Cortical shell thickness and trabecular density fields per bone; an osteoporosis level (T-score) slider; values available to the synthetic radiographs (5.1) and FE materials (6.2). | none | |
| **3.5 Statistical shape model** | Sample realistic shape variation learned from real patients. | Registration pipeline from public CT segmentations (TotalSegmentator, CC BY 4.0; VerSe after its share-alike licence is cleared) to the template's fields; PCA of the deformations per vertebra (or per region); shape mode sliders; sampled shapes are valid (no self-intersection, landmarks found); leave-one-out reconstruction error reported. | 3.1 | |

## Phase 4: Virtual patient generator

| Step | Objective | Done when | Depends on | Status |
|---|---|---|---|---|
| **4.1 Correlated sampling** | Sample whole patients from population distributions instead of independent sliders. | Distributions and correlations (PI-LL, age effects on kyphosis, disc height and bone density, sex differences) from cited cohorts; a "sample patient" action with a seed; generated cohorts reproduce the input statistics within stated tolerances. | 1.5, 3.1 | |
| **4.2 Presets** | Provide clinically recognisable patient types. | Presets for normal spines by Roussouly type, adolescent idiopathic scoliosis by Lenke type, adult degenerative scoliosis, hyperkyphosis, osteoporotic fractures and spondylolisthesis; each preset measured back against its definition. | 4.1, 3.2 | |
| **4.3 Validity report** | Certify every generated patient. | One report per patient: clearances, disc heights, ligament lengths, balance, parameter values; invalid patients flagged or rejected by the generator. | 1.4, 2.6 | |
| **4.4 Export** | Hand a patient to other tools. | Surfaces (STL) per structure, label volume (NIfTI), landmarks, parameter record (JSON), all in one folder with a manifest; re-importing the record regenerates the same patient. | 4.3 | |

## Phase 5: Realism against radiographs and CT

| Step | Objective | Done when | Depends on | Status |
|---|---|---|---|---|
| **5.1 Synthetic radiographs** | Show a virtual patient as an AP and a lateral radiograph. | Density integrated along rays in the shader (cortical, trabecular, soft tissue); AP and lateral views in the viewer and exported as images; side-by-side comparison with real radiographs on public example images. | 3.4 | |
| **5.2 Synthetic CT** | Produce a CT-like volume of a virtual patient. | Density volume at a chosen voxel size with scanner-like blur and noise; NIfTI export; Hounsfield ranges of bone and soft tissue within published ranges. | 3.4 | |
| **5.3 Quantitative validation** | Show that generated cohorts are anatomically plausible. | Parameters measured on generated cohorts with the clinical definitions; distributions compared with published cohorts (Legaye 1998, Schwab 2012, Roussouly 2005); comparison report in docs/. | 4.1 | |
| **5.4 Synthetic training data** | Generate labelled synthetic images for AI training (optional). | Batch generation of radiographs/CT with exact labels (segmentation, landmarks, parameters); documented format. | 5.1, 5.2, 4.4 | |

## Phase 6: Finite element pipeline

| Step | Objective | Done when | Depends on | Status |
|---|---|---|---|---|
| **6.1 Meshing** | Mesh any generated patient for FE. | Conforming hexahedral or tetrahedral meshes of bones, discs and cartilage from the implicit fields (shared nodes at interfaces), element quality within stated limits; ligaments as 1-D elements. | 2.1, 2.6 | |
| **6.2 Materials and loads** | Assign materials and boundary conditions. | Material regions (cortical and trabecular bone from 3.4, annulus with fibres, nucleus, cartilage, ligaments) with cited properties; standard load cases (pure moments, follower load). | 6.1 | |
| **6.3 Marc export** | Run generated patients in Marc. | Marc input deck exported and verified on a published functional spinal unit benchmark (range of motion within the reported corridor). | 6.2 | |
| **6.4 Browser solver** | Solve small FE models in the browser. | A lightweight solver (WebGPU) for linear and moderately nonlinear statics; matches Marc on the 6.3 benchmark. | 6.3 | |

---

## Critical path

```
1.1 -> 1.2 -> 1.3 -> 1.4 -> 2.1 -> 2.2
                      |      `-> 3.2 -> 3.3            3.4 -> 5.1, 5.2
                      |-> 2.3 -> 2.6 <- 2.4 -> 2.5
                      |-> 1.5 -> 4.1 -> 4.2, 4.3 -> 4.4 -> 5.3, 5.4
                      `-> 1.6, 2.7 -> 2.8              3.1 -> 3.5
2.1 + 2.6 -> 6.1 -> 6.2 -> 6.3 -> 6.4
```

Milestones: **M1 valid posture** (1.1-1.4), **M2 discs** (2.1-2.2), **M3 first
realistic virtual patients** (3.2-3.4, 5.1), **M4 posterior elements and
ligaments** (2.3-2.6), **M5 generator** (1.5, 1.6, 3.1, 4.x), **M6 shape model**
(3.5), **M7 rib cage and muscles** (2.7, 2.8), **M8 FE** (6.x).

## Open questions (resolved in the step that needs them)

- Thoracic rotation centres and motion limits are less documented than lumbar ones
  (1.2, 1.3): sources to be chosen and cited.
- The fold-free field (2.3): a stationary velocity field integrated in the shader is
  the likely route; its cost has to stay within the budget.
- VerSe's CC BY-SA terms against the intended distribution of derived models (3.5).
- Source of muscle geometry (2.8): VIVA+ parts or attachment-based construction.
