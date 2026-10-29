"""Full spine (C1-L5), sacrum, pelvis and proximal femurs from CT segmentations,
one implicit (signed distance) body per bone.

Prepare a subject once (needs scipy + nibabel):
    python -m implicit.anatomy.totalseg <subject> --out data/totalseg
    python -m implicit.anatomy.preprocess data/totalseg/<subject> --out assets/models/<subject>
Then:
    python -m implicit.serve examples/full_spine.py

Set IMPLICIT_MODEL to use another processed subject directory.
"""

import os
from pathlib import Path

from implicit.anatomy.spine import build_scene

ROOT = Path(__file__).resolve().parent.parent
MODEL = Path(os.environ.get("IMPLICIT_MODEL", ROOT / "assets" / "models" / "s0751"))

scene, model = build_scene(MODEL, name="Full spine + pelvis")
