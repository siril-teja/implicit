"""Fetch one subject's bone masks from the TotalSegmentator v2 dataset.

Dataset: Wasserthal et al., TotalSegmentator, Radiology: AI 2023,
https://doi.org/10.5281/zenodo.8367088 (CC BY 4.0). Only the requested mask
files are downloaded (HTTP range requests into the 23.6 GB zip).

    python -m implicit.anatomy.totalseg s0751 --out data/totalseg --ct

``--ct`` also downloads the CT (about 36 MB), which implicit.anatomy.convert
uses to place bone surfaces more accurately than the 1.5 mm masks.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from .remote_zip import open_remote_zip

URL = "https://zenodo.org/api/records/8367088/files/Totalsegmentator_dataset_v2.zip/content"
ATTRIBUTION = (
    "Anatomy: TotalSegmentator v2 dataset (Wasserthal et al., Radiology: AI 2023), "
    "CC BY 4.0, doi:10.5281/zenodo.8367088. Processed into signed distance fields."
)

LEVELS = ([f"C{i}" for i in range(1, 8)] + [f"T{i}" for i in range(1, 13)]
          + [f"L{i}" for i in range(1, 6)] + ["S1"])
VERTEBRAE = [f"vertebrae_{lv}" for lv in LEVELS]
PELVIS = ["sacrum", "hip_left", "hip_right", "femur_left", "femur_right"]
EXTRA = ["spinal_cord"]
ALL = VERTEBRAE + PELVIS + EXTRA


def fetch(subject, out_dir, names=ALL, url=URL, ct=False):
    out = Path(out_dir) / subject
    out.mkdir(parents=True, exist_ok=True)
    z = open_remote_zip(url)
    members = [(f"{subject}/segmentations/{n}.nii.gz", out / f"{n}.nii.gz") for n in names]
    if ct:
        members.append((f"{subject}/ct.nii.gz", out / "ct.nii.gz"))
    for member, dst in members:
        if dst.exists():
            continue
        dst.write_bytes(z.read(member))
        print(f"  {dst}")
    (out / "ATTRIBUTION.txt").write_text(ATTRIBUTION + "\n")
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("subject", help="TotalSegmentator image id, e.g. s0011")
    ap.add_argument("--out", default="data/totalseg")
    ap.add_argument("--ct", action="store_true", help="also fetch the CT (~36 MB)")
    a = ap.parse_args(argv)
    print(f"fetched {fetch(a.subject, a.out, ct=a.ct)}")


if __name__ == "__main__":
    main()
