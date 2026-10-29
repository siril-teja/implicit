"""Implicit bones of one subject: lumbar spine, sacrum, pelvis, proximal femurs.

Each bone is its own implicit body (signed distance field) loaded from
assets/bones/<subject>/<bone>.npz, converted once from TotalSegmentator masks
(+ CT). All modelling works on these fields.

Convert a subject (needs scipy + nibabel):
    python -m implicit.anatomy.totalseg s0751 --out data/totalseg --ct
    python -m implicit.anatomy.convert data/totalseg/s0751 --out assets/bones/s0751 --select lumbar
View:
    python -m implicit.serve examples/anatomy.py

Set IMPLICIT_BONES to view another converted subject.
"""

import os
from pathlib import Path

from implicit.anatomy.spine import build_scene

ROOT = Path(__file__).resolve().parent.parent
BONES = Path(os.environ.get("IMPLICIT_BONES", ROOT / "assets" / "bones" / "s0751"))

scene, model = build_scene(BONES, name="Lumbar spine + pelvis (implicit bones)")
