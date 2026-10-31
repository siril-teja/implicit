"""TotalSegmentator masks (+ CT) -> implicit bodies, one file per bone.

    python -m implicit.anatomy.convert data/totalseg/s0751 --out assets/bones/s0751 --select lumbar

This is the only step that reads segmentation masks. Everything downstream
(landmarks, scenes, modelling) loads the per-bone implicit files written here.

Per bone
  1. mask = largest connected component (sacrum = sacrum + vertebrae_S1;
     femurs cropped to the proximal ``femur_keep`` mm);
  2. signed distance to the mask (two Euclidean distance transforms,
     half-voxel corrected);
  3. with a CT next to the masks (``ct.nii.gz``): the surface is moved onto
     the CT iso-surface. Occupancy = deep inside the label (> ``band`` mm) OR
     (within ``band`` of the label surface AND closer to this bone's label
     than to any other bone AND CT > threshold), holes filled, re-distanced.
     The threshold is the half-maximum edge between the soft-tissue level
     (3-5 mm outside the label) and the cortical peak (near the label surface),
     both measured so that a label error of up to +-band does not bias it.
     The label only guards the surface to +-band;
  4. sampled on a world-axis-aligned grid (scanner RAS frame, mm) and lightly
     smoothed.

File ``<bone>.npz`` (NumPy archive):
  sdf      uint8 [nx, ny, nz]: signed distance, (sdf - qzero) * qstep mm,
           negative inside, clamped to +-qstep * 127
  origin   float64 [3]: position (mm) of sample [0, 0, 0]
  spacing  float64: grid spacing (mm)
  qstep, qzero
  meta     JSON string: name, kind, source masks, CT refinement, threshold,
           volume, attribution, format version
Load with ``load_bone(path) -> (Grid, meta)``.

Requires scipy and nibabel (this step only; loading needs just NumPy).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ..sdf import QSTEP, QZERO, Grid
from .totalseg import ATTRIBUTION

FORMAT_VERSION = 1

# Output bone -> (kind, source mask names, default spacing mm with CT, without CT)
VERTEBRAE = ([f"C{i}" for i in range(1, 8)] + [f"T{i}" for i in range(1, 13)]
             + [f"L{i}" for i in range(1, 6)])
BONES = {lv: ("vertebra", [f"vertebrae_{lv}"]) for lv in VERTEBRAE}
BONES.update({
    "sacrum": ("sacrum", ["sacrum", "vertebrae_S1"]),
    "hip_left": ("hip", ["hip_left"]), "hip_right": ("hip", ["hip_right"]),
    "femur_left": ("femur", ["femur_left"]), "femur_right": ("femur", ["femur_right"]),
})
# Not bone: mask only (no CT refinement), cropped to the selected bones' extent.
REFERENCE = {"spinal_cord": ("reference", ["spinal_cord"])}
SPACING = {"vertebra": 0.6, "sacrum": 0.8, "hip": 1.5, "femur": 1.0, "reference": 1.5,
           "rib": 1.0, "sternum": 0.8, "cartilage": 1.0}
PRESETS = {
    "lumbar": [f"L{i}" for i in range(1, 6)] + ["sacrum", "hip_left", "hip_right",
                                                 "femur_left", "femur_right", "spinal_cord"],
    "all": VERTEBRAE + ["sacrum", "hip_left", "hip_right", "femur_left", "femur_right",
                        "spinal_cord"],
}


def _ndi():
    from scipy import ndimage
    return ndimage


def load_mask(path):
    import nibabel as nib
    im = nib.as_closest_canonical(nib.load(str(path)))
    return np.asarray(im.dataobj) > 0, im.affine


def largest_component(m):
    ndi = _ndi()
    lab, n = ndi.label(m)
    if n <= 1:
        return m
    sizes = ndi.sum(m, lab, range(1, n + 1))
    return lab == (1 + int(np.argmax(sizes)))


def _mask_sdf(mask, affine, margin):
    """Signed distance to a mask on its own (canonical CT) voxel grid, cropped.

    Returns (sdf, voxel offset of the crop, mean voxel size).
    """
    ndi = _ndi()
    zooms = np.sqrt((affine[:3, :3] ** 2).sum(0))
    idx = np.argwhere(mask)
    lo, hi = idx.min(0), idx.max(0) + 1
    m = int(np.ceil(margin / zooms.min())) + 2
    crop = np.pad(mask[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]], m)
    din = ndi.distance_transform_edt(crop, sampling=zooms)
    dout = ndi.distance_transform_edt(~crop, sampling=zooms)
    hv = float(zooms.mean())
    sdf = np.where(crop, -(din - 0.5 * hv), dout - 0.5 * hv).astype(np.float32)
    return sdf, lo - m, hv


def _world_grid(sdf_shape, off, affine, h, inset):
    """World-aligned sample grid covering a voxel crop (shrunk by ``inset`` mm).

    Returns (grid shape, origin, absolute voxel coordinates [3, N] of the samples).
    """
    n0 = np.array(sdf_shape) - 1
    corners = np.array([[i, j, k] for i in (0, n0[0]) for j in (0, n0[1]) for k in (0, n0[2])]) + off
    world = corners @ affine[:3, :3].T + affine[:3, 3]
    wlo, whi = world.min(0) + inset, world.max(0) - inset
    n = np.floor((whi - wlo) / h).astype(int) + 1
    axes = [wlo[i] + h * np.arange(n[i]) for i in range(3)]
    pts = np.stack(np.meshgrid(*axes, indexing="ij"), -1).reshape(-1, 3)
    inv = np.linalg.inv(affine)
    return n, wlo, (pts @ inv[:3, :3].T + inv[:3, 3]).T


class CTContext:
    """CT intensities + a nearest-bone partition over a region, for surface refinement."""

    def __init__(self, ct, affine, label_masks: dict, margin_mm=15.0, smooth=0.7):
        ndi = _ndi()
        zooms = np.sqrt((affine[:3, :3] ** 2).sum(0))
        union = np.zeros(ct.shape, bool)
        for m in label_masks.values():
            union |= m
        idx = np.argwhere(union)
        mv = np.ceil(margin_mm / zooms).astype(int)
        self.lo = np.maximum(idx.min(0) - mv, 0)
        hi = np.minimum(idx.max(0) + 1 + mv, ct.shape)
        sl = tuple(slice(a, b) for a, b in zip(self.lo, hi))
        self.ct = ndi.gaussian_filter(ct[sl].astype(np.float32), smooth)  # tame CT noise
        self.coef = ndi.spline_filter(self.ct, order=3)  # prefiltered once for cubic sampling
        lab = np.zeros(self.ct.shape, np.int16)
        self.ids = {}
        for k, (name, m) in enumerate(label_masks.items(), start=1):
            lab[m[sl]] = k
            self.ids[name] = k
        _, nearest = ndi.distance_transform_edt(lab == 0, sampling=zooms, return_indices=True)
        self.owner = lab[tuple(nearest)]
        self.lab = lab
        self.zoom = float(zooms.mean())

    def threshold(self, name, band=2.0, shell=(3.0, 5.0)):
        """Half-maximum edge threshold (HU) for one bone.

        Half-way between the soft-tissue level (median CT in a shell 3-5 mm
        outside the label, other bones excluded) and the cortical peak (90th
        percentile from ``band`` mm outside to ``band`` + 3 mm inside the label
        surface, voxels owned by this bone). Both levels are measured away from the label boundary, so a
        label that is off by up to ``band`` does not bias the threshold.
        """
        ndi = _ndi()
        m = self.lab == self.ids[name]
        k = lambda mm: max(int(np.ceil(mm / self.zoom)), 1)
        near = ndi.binary_dilation(m, iterations=k(band)) & ~ndi.binary_erosion(m, iterations=k(band + 3.0))
        bone = np.percentile(self.ct[near & (self.owner == self.ids[name])], 90)
        out = (ndi.binary_dilation(m, iterations=k(shell[1]))
               & ~ndi.binary_dilation(m, iterations=k(shell[0])) & (self.lab == 0))
        soft = np.median(self.ct[out])
        return 0.5 * (float(soft) + float(bone))

    def sample(self, vox_abs, name):
        """Cubic CT and ownership at absolute (canonical) voxel coordinates [3, N]."""
        ndi = _ndi()
        v = vox_abs - self.lo[:, None]
        ct = ndi.map_coordinates(self.coef, v, order=3, mode="nearest", prefilter=False)
        own = ndi.map_coordinates((self.owner == self.ids[name]).astype(np.float32), v,
                                  order=1, mode="constant", cval=0.0) > 0.5
        return ct, own


def bone_sdf(mask, affine, h, ctx: CTContext | None = None, name=None,
             band=2.0, pad=6.0, sigma=0.7):
    """Signed distance grid (world-aligned, spacing h) for one bone.

    Returns (values[nx, ny, nz] mm, origin[3], CT threshold or None).
    """
    ndi = _ndi()
    s_mask, off, hv = _mask_sdf(mask, affine, pad + band)
    if ctx is None:  # mask only: smooth the voxel staircase on the source grid
        n, wlo, vox = _world_grid(s_mask.shape, off, affine, h, hv)
        vals = ndi.map_coordinates(ndi.gaussian_filter(s_mask, 0.6), vox - off[:, None],
                                   order=1, mode="nearest")
        return vals.reshape(n).astype(np.float32), wlo, None
    n, wlo, vox = _world_grid(s_mask.shape, off, affine, h, hv)
    sm = ndi.map_coordinates(s_mask, vox - off[:, None], order=1, mode="nearest")
    ct, own = ctx.sample(vox, name)
    T = ctx.threshold(name)
    occ = (sm < -band) | ((sm < band) & own & (ct > T))
    occ = largest_component(ndi.binary_fill_holes(occ.reshape(n)))
    din = ndi.distance_transform_edt(occ, sampling=h)
    dout = ndi.distance_transform_edt(~occ, sampling=h)
    sdf = np.where(occ, -(din - 0.5 * h), dout - 0.5 * h).astype(np.float32)
    if sigma:
        sdf = ndi.gaussian_filter(sdf, sigma)
    return sdf, wlo, T


# ---------------------------------------------------------------- bone files

def save_bone(path, values, origin, spacing, meta):
    g = Grid(values, origin, spacing, meta["name"])  # quantises and checks the margin
    meta = {**meta, "format_version": FORMAT_VERSION, "frame": "scanner RAS, mm",
            "volume_mm3": float((g.values < 0).sum() * spacing ** 3)}
    np.savez_compressed(path, sdf=g.q, origin=g.origin, spacing=np.float64(spacing),
                        qstep=np.float64(QSTEP), qzero=np.int64(QZERO),
                        meta=np.array(json.dumps(meta)))
    return meta


def load_bone(path):
    """Load one implicit bone file -> (Grid, meta dict)."""
    with np.load(path) as d:
        if float(d["qstep"]) != QSTEP or int(d["qzero"]) != QZERO:
            raise ValueError(f"{path}: unsupported quantisation")
        meta = json.loads(str(d["meta"]))
        g = Grid.from_quantized(d["sdf"], d["origin"], float(d["spacing"]), meta["name"])
    return g, meta


def load_bones(directory):
    """All ``*.npz`` bone files of a directory -> {name: (Grid, meta)}."""
    return {p.stem: load_bone(p) for p in sorted(Path(directory).glob("*.npz"))}


# ---------------------------------------------------------------- conversion

def _bbox(mask, affine):
    idx = np.argwhere(mask)
    pts = np.array([idx.min(0), idx.max(0)]) @ affine[:3, :3].T + affine[:3, 3]
    return pts.min(0), pts.max(0)


def convert(subject_dir, out_dir, select="lumbar", use_ct=True, spacing=None,
            femur_keep=110.0, log=print):
    """Convert the selected structures of one subject; returns {bone: meta}."""
    subject_dir, out_dir = Path(subject_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    names = PRESETS[select] if isinstance(select, str) and select in PRESETS else list(select)
    spec = {**BONES, **REFERENCE}
    unknown = [n for n in names if n not in spec]
    if unknown:
        raise ValueError(f"unknown structures: {unknown}")
    h_of = {**SPACING, **(spacing or {})}

    def load(name):
        sources, mask, aff = spec[name][1], None, None
        for s in sources:
            p = subject_dir / f"{s}.nii.gz"
            if p.exists():
                m, aff = load_mask(p)
                mask = m if mask is None else mask | m
        if mask is None or not mask.any():
            return None, None
        mask = largest_component(mask)
        if name.startswith("femur"):
            zs = np.argwhere(mask)[:, 2]
            zoom = np.sqrt((aff[:3, :3] ** 2).sum(0))[2]
            mask[:, :, :max(int(zs.max() - femur_keep / zoom), 0)] = False
        return mask, aff

    masks, aff = {}, None
    for n in names:
        m, a = load(n)
        if m is None:
            log(f"  {n}: no mask, skipped")
            continue
        masks[n], aff = m, a

    bones = [n for n in masks if spec[n][0] != "reference"]
    # Reference structures (spinal cord) are cropped to the selected bones' extent.
    if bones:
        lo_v = np.min([np.argwhere(masks[n]).min(0) for n in bones], axis=0)
        hi_v = np.max([np.argwhere(masks[n]).max(0) for n in bones], axis=0)
        mv = np.ceil(20.0 / np.sqrt((aff[:3, :3] ** 2).sum(0))).astype(int)
        keep = np.zeros(next(iter(masks.values())).shape, bool)
        keep[tuple(slice(max(a - m, 0), b + m + 1) for a, b, m in zip(lo_v, hi_v, mv))] = True
        for n in masks:
            if spec[n][0] == "reference":
                masks[n] = largest_component(masks[n] & keep)

    ctx = None
    ct_path = subject_dir / "ct.nii.gz"
    if use_ct and ct_path.exists() and bones:
        import nibabel as nib
        ct_im = nib.as_closest_canonical(nib.load(str(ct_path)))
        if not np.allclose(ct_im.affine, aff, atol=1e-3):
            raise RuntimeError("CT and mask grids differ")
        # Ownership partition: the selected bones plus every other bone whose
        # box comes within 20 mm, so e.g. L1 is not given T12's surface.
        lo = np.min([_bbox(masks[n], aff)[0] for n in bones], axis=0) - 20
        hi = np.max([_bbox(masks[n], aff)[1] for n in bones], axis=0) + 20
        part = {n: masks[n] for n in bones}
        for n in BONES:
            if n in part:
                continue
            m, _ = load(n)
            if m is not None:
                b0, b1 = _bbox(m, aff)
                if np.all(b1 > lo) and np.all(b0 < hi):
                    part[n] = m
        log(f"  CT refinement; ownership partition over {len(part)} bones")
        ctx = CTContext(np.asarray(ct_im.dataobj), aff, part)

    out = {}
    for n, m in masks.items():
        kind, sources = spec[n]
        h = h_of[kind]
        refine = ctx if kind != "reference" else None
        v, o, T = bone_sdf(m, aff, h, refine, n)
        meta = {"name": n, "kind": kind, "subject": subject_dir.name, "source_masks": sources,
                "ct_refined": refine is not None, "ct_threshold_hu": T,
                "attribution": ATTRIBUTION}
        out[n] = save_bone(out_dir / f"{n}.npz", v, o, h, meta)
        log(f"  {n}: {v.shape} @ {h} mm, {out[n]['volume_mm3'] / 1000:.1f} cm3"
            + (f", CT threshold {T:.0f} HU" if T is not None else ", mask only"))
    (out_dir / "ATTRIBUTION.txt").write_text(ATTRIBUTION + "\n")
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("subject_dir", help="folder with TotalSegmentator <structure>.nii.gz (+ ct.nii.gz)")
    ap.add_argument("--out", required=True, help="output folder for <bone>.npz files")
    ap.add_argument("--select", default="lumbar",
                    help='"lumbar", "all" or a comma list (e.g. L3,L4,sacrum)')
    ap.add_argument("--no-ct", action="store_true", help="masks only, even if ct.nii.gz exists")
    for kind, h in SPACING.items():
        ap.add_argument(f"--h-{kind}", type=float, default=h, help=f"grid spacing for {kind} (mm)")
    a = ap.parse_args(argv)
    select = a.select if a.select in PRESETS else a.select.split(",")
    convert(a.subject_dir, a.out, select, use_ct=not a.no_ct,
            spacing={k: getattr(a, f"h_{k}") for k in SPACING})


if __name__ == "__main__":
    main()
