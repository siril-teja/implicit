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
`actions/setup-python`, `actions/upload-pages-artifact`, `actions/deploy-pages`,
all MIT). They are not part of the distributed application.

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
