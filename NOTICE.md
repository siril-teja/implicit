# Notices: licences and attribution

**implicit** is released under the [MIT License](LICENSE). This file lists the
third-party software and data it uses, with their licences. The policy for
adding dependencies is in [docs/LICENSING.md](docs/LICENSING.md): only licences
that allow commercial use.

## Software

| Component | Used for | Licence |
|---|---|---|
| [Python](https://www.python.org/) standard library | everything | PSF License |
| [NumPy](https://numpy.org/) | field evaluation, meshing, file format | BSD-3-Clause |
| [SciPy](https://scipy.org/) | distance transforms, interpolation (conversion only) | BSD-3-Clause |
| [nibabel](https://nipy.org/nibabel/) | reading NIfTI segmentations and CT (conversion only) | MIT |

The browser viewer uses only the browser's WebGL2 API and has no JavaScript
dependencies. The CI uses GitHub Actions (`actions/checkout`,
`actions/setup-python`, `actions/setup-node`, `actions/upload-pages-artifact`,
`actions/deploy-pages`, all MIT) and [Playwright](https://playwright.dev/)
(Apache-2.0) with its bundled Chromium (BSD-3-Clause and others) for the GPU
check. They are not part of the distributed application.

## Data

### TotalSegmentator v2 (bundled anatomy in `assets/bones/`)

Wasserthal J., Breit H.-C., Meyer M.T., et al. *TotalSegmentator: robust
segmentation of 104 anatomical structures in CT images.* Radiology: Artificial
Intelligence, 2023. Dataset: doi:[10.5281/zenodo.8367088](https://doi.org/10.5281/zenodo.8367088).

Licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). The
files in `assets/bones/` are derived works: signed distance fields computed
from the segmentation masks and the CT of image `s0751`. They are not the
original data. See [assets/bones/README.md](assets/bones/README.md). These data
files stay under CC BY 4.0; the MIT License covers the code only.

### VIVA+ human body models (bundled anatomy in `assets/bones/vivaplus-50F/`)

VIVA+ open finite element human body models, v2.0.2, average female (50F)
standing; Chalmers University of Technology and partners.
[vivaplus.readthedocs.io](https://vivaplus.readthedocs.io),
doi:[10.5281/zenodo.7789362](https://doi.org/10.5281/zenodo.7789362).
Lumbar spine: Iraeus J. et al., *A new open-source finite element lumbar spine
model*, IRCOBI 2023.

Licensed under the [GNU LGPL v3.0 or later](https://www.gnu.org/licenses/lgpl-3.0.html).
The files in `assets/bones/vivaplus-50F/` are derived works (bone surfaces of
the model converted to signed distance fields; the thoracic and lumbar vertebrae
translated so that the disc spaces take population heights and the facet joints
carved to the population joint space, recorded in `template.json`) and stay under the LGPL-3.0-or-later;
the MIT License covers the code only. See [docs/LICENSING.md](docs/LICENSING.md).
