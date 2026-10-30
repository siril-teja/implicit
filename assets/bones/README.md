# Implicit bones

One folder per subject, one file per bone, written by `python -m implicit.anatomy.convert`.
See the docstring of `implicit/anatomy/convert.py` for the file format; load a
file with `implicit.anatomy.convert.load_bone(path) -> (Grid, meta)`.

| Subject | Source | Bones | Grid spacing | Surface | Licence |
|---|---|---|---|---|---|
| `s0751` | TotalSegmentator v2, image `s0751` (62 y, male, CT angiography neck to legs, 1.5 mm) | L1–L5, sacrum (incl. S1), hips, proximal femurs (110 mm), spinal cord (reference, mask only) | L 0.6, sacrum 0.8, femurs 1.0, hips 1.5 mm | CT iso-surface (half-maximum threshold), label as ±2 mm guard | CC BY 4.0 |

Coordinates are the CT scanner frame (RAS, mm), so bones stay registered to each other.

Source: Wasserthal J. et al., *TotalSegmentator: robust segmentation of 104 anatomical
structures in CT images*, Radiology: Artificial Intelligence 2023;
dataset doi:10.5281/zenodo.8367088, CC BY 4.0. The files here are derived works
(signed distance fields computed from the segmentation masks and the CT).

`landmarks.json` (per subject) holds the vertebral frames and endplate planes in
the same frame, from `python -m implicit.anatomy.landmarks <folder>`. The
Landmarks block of the block canvas reads it.
