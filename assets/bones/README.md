# Implicit bones

One folder per subject, one file per bone, written by `python -m implicit.anatomy.convert`.
See the docstring of `implicit/anatomy/convert.py` for the file format; load a
file with `implicit.anatomy.convert.load_bone(path) -> (Grid, meta)`.

| Subject | Source | Bones | Grid spacing | Surface | Licence |
|---|---|---|---|---|---|
| `vivaplus-50F` | VIVA+ v2.0.2 human body model, average female (50F), standing | T1–T12, L1–L5, sacrum, hips, proximal femurs | T/L 0.6, sacrum 0.8, femurs 1.0, hips 1.5 mm | model surfaces (FE mesh, cortical shells) | LGPL-3.0-or-later |
| `s0751` | TotalSegmentator v2, image `s0751` (62 y, male, CT angiography neck to legs, 1.5 mm) | L1–L5, sacrum (incl. S1), hips, proximal femurs (110 mm), spinal cord (reference, mask only) | L 0.6, sacrum 0.8, femurs 1.0, hips 1.5 mm | CT iso-surface (half-maximum threshold), label as ±2 mm guard | CC BY 4.0 |

Coordinates are right-anterior-superior (mm): the CT scanner frame for `s0751`, the model frame
(x right, y anterior, z up) for `vivaplus-50F`. Bones of a subject stay registered to each other.

`vivaplus-50F` is written by `python -m implicit.anatomy.vivaplus --model 50F-standing --out assets/bones/vivaplus-50F`
(downloads the release from Zenodo once).

The vertebrae of `vivaplus-50F` are translated (not rotated) from the model's positions so
that the rest disc spaces have population middle heights (`python -m implicit.anatomy.template`,
run by the converter; the shifts are in `template.json` and in each bone file's metadata,
`template_shift_mm`). L1-L2 and L2-L3 stop where their facets come as close as the model's
closest pair already was (0.31 mm).

Source: Wasserthal J. et al., *TotalSegmentator: robust segmentation of 104 anatomical
structures in CT images*, Radiology: Artificial Intelligence 2023;
dataset doi:10.5281/zenodo.8367088, CC BY 4.0. The files here are derived works
(signed distance fields computed from the segmentation masks and the CT).

`landmarks.json` (per subject) holds the vertebral frames and endplate planes in
the same frame, from `python -m implicit.anatomy.landmarks <folder>`. The
Landmarks block of the block canvas reads it.
