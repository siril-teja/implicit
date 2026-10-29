# Processed anatomy models

Each folder holds one subject processed by `python -m implicit.anatomy.preprocess`:

- `bones.npz`: one signed-distance grid per bone (uint8, 0.1 mm per level, see `implicit.sdf.Grid`)
- `model.json`: grid placement plus vertebral landmarks (body frame, endplate planes)
- `ATTRIBUTION.txt`: data source and licence

| Subject | Source | Contents | Licence |
|---|---|---|---|
| `s0751` | TotalSegmentator v2, image `s0751` (62 y, male, CT angiography neck to legs, 1.5 mm) | C1–L5, sacrum (incl. S1), hips, proximal femurs | CC BY 4.0 |

Source: Wasserthal J. et al., *TotalSegmentator: robust segmentation of 104 anatomical
structures in CT images*, Radiology: Artificial Intelligence 2023;
dataset doi:10.5281/zenodo.8367088, CC BY 4.0. The files here are derived works
(signed distance fields and landmarks computed from the segmentation masks).
