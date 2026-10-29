"""Parametric L4-L5 functional spinal unit with an interbody cage and a PSO.

Idealised geometry (mm), not patient-specific. Axes: x lateral, y anterior,
z cranial. Every slider in the viewer is a ``Param`` below; dragging it only
changes a shader uniform, so the whole assembly updates in real time.

Run:  python -m implicit.serve examples/spine_segment.py
"""

from implicit import (
    Box, Capsule, Cylinder, EllipticCylinder, HalfSpace, Param, Scene,
    cos, radians, sin, union,
)

# ---------------------------------------------------------------- parameters
body_w = Param("body_w", 48, 36, 60, 0.5, "Body width", "mm")
body_d = Param("body_d", 35, 26, 44, 0.5, "Body depth (AP)", "mm")
body_h = Param("body_h", 27, 20, 34, 0.5, "Body height", "mm")
cort_t = Param("cort_t", 1.0, 0.3, 3.0, 0.05, "Cortical shell thickness", "mm")
disc_h = Param("disc_h", 10, 4, 16, 0.1, "Disc height (mid)", "mm")
lordosis = Param("lordosis", 8, -5, 25, 0.1, "Segmental lordosis = cage wedge", "deg")
cage_w = Param("cage_w", 16, 6, 24, 0.5, "Cage half-length (lateral)", "mm")
cage_d = Param("cage_d", 9, 4, 15, 0.5, "Cage half-width (AP)", "mm")
cage_y = Param("cage_y", 0, -10, 10, 0.1, "Cage AP position", "mm")
cage_rot = Param("cage_rot", 0, -45, 45, 0.5, "Cage axial obliquity", "deg")
window = Param("window", 0.55, 0.0, 0.8, 0.01, "Cage graft window fraction", "")
pso_angle = Param("pso_angle", 0, 0, 35, 0.5, "L4 PSO wedge angle", "deg")
pso_close = Param("pso_close", 0, 0, 1, 0.01, "L4 PSO closure (0 open, 1 closed)", "")

a, b, hz = body_w / 2, body_d / 2, body_h / 2
BIG = 200.0


# ---------------------------------------------------------------- vertebra
def footprint(half_height):
    """Kidney-shaped axial section of the vertebral body (posterior notch), extruded in z."""
    notch = Cylinder(14, half_height + 1).translate((0, -b - 10, 0))
    return EllipticCylinder(a, b, half_height) - notch


def vertebra():
    pedicle = Capsule((12.5, -b + 4, 3), (13.5, -b - 12, 3), 4.5)
    lamina = Capsule((13.5, -b - 12, 2), (0, -b - 24, 0), 3.5)
    transverse = Capsule((13, -b - 10, 4), (34, -b - 12, 6), 3.0)
    sup_facet = Capsule((14, -b - 14, 3), (16, -b - 16, 14), 4.0)
    inf_facet = Capsule((11, -b - 22, -2), (12, -b - 24, -14), 3.5)
    spinous = Capsule((0, -b - 24, 0), (0, -b - 44, -5), 4.0)
    posterior = union(pedicle, lamina, transverse, sup_facet, inf_facet, k=2.0).mirror_x()
    return footprint(hz).smooth_union(posterior | spinous, 2.0)


def pso(v, angle, closure):
    """Pedicle subtraction osteotomy: posterior-based wedge, hinge at the anterior cortex.

    The fragment above the cut is rotated about the hinge by angle * closure,
    so closure = 1 brings the two cut faces together.
    """
    hinge = (0, b, 0)
    t = radians(angle / 2)
    above_cut = HalfSpace(hinge, (0, -sin(t), -cos(t)))
    below_cut = HalfSpace(hinge, (0, -sin(t), cos(t)))
    upper = (v & above_cut).rotate_x(angle * closure, center=hinge)
    return upper | (v & below_cut)


# ---------------------------------------------------------------- assembly
# L5 centred at the origin; L4 stacked above and rotated about the disc centre.
z_ep = hz                       # L5 superior endplate
z_mid = hz + disc_h / 2         # disc centre = lordosis pivot
th = radians(lordosis)

l5 = vertebra()
l4 = pso(vertebra(), pso_angle, pso_close)
l4 = l4.translate((0, 0, body_h + disc_h)).rotate_x(lordosis, center=(0, 0, z_mid))

# Disc space bounded by the two endplates (planes move with the lordosis param).
lower_ep = HalfSpace((0, 0, z_ep), (0, 0, -1))
upper_ep_point = (0, -sin(th) * disc_h / 2, z_mid + cos(th) * disc_h / 2)
upper_ep = HalfSpace(upper_ep_point, (0, -sin(th), cos(th)))
disc_space = footprint(BIG) & lower_ep & upper_ep

# Cage: footprint box with graft window, clipped by the endplates, so its
# wedge angle and height always equal the local disc space.
cage_local = Box((cage_w, cage_d, BIG), round=1.5) - Box(
    (cage_w * window, cage_d * window, BIG + 1), round=1.0
)
cage = cage_local.rotate_z(cage_rot).translate((0, cage_y, 0)) & lower_ep & upper_ep

scene = Scene("L4-L5 FSU, cage + PSO", bounds=((-42, -85, -32), (42, 34, 80)))
scene.add("L5 cortical", l5.shell(cort_t), (0.93, 0.88, 0.78))
scene.add("L5 cancellous", l5.offset(-cort_t), (0.85, 0.55, 0.45))
scene.add("L4 cortical", l4.shell(cort_t), (0.93, 0.88, 0.78))
scene.add("L4 cancellous", l4.offset(-cort_t), (0.85, 0.55, 0.45))
scene.add("Disc", disc_space - cage.offset(0.2), (0.45, 0.65, 0.9))
scene.add("Cage", cage, (0.35, 0.8, 0.45))
