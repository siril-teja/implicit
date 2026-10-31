"""Field-driven (implicit / SDF) modeling with a browser viewer."""

__version__ = "0.16.0"  # single source of truth; bump in every release PR (see CONTRIBUTING.md)

from .expr import Param, cos, radians, sin, tan, vec3
from .scene import Scene
from .sdf import (
    Box, Capsule, Cylinder, Difference, EllipticCylinder, HalfSpace, Intersect,
    MirrorX, Offset, Rotate, Scale, Shell, Sphere, Translate, Union, union,
)

__all__ = [
    "Param", "cos", "radians", "sin", "tan", "vec3", "Scene",
    "Box", "Capsule", "Cylinder", "Difference", "EllipticCylinder", "HalfSpace",
    "Intersect", "MirrorX", "Offset", "Rotate", "Scale", "Shell", "Sphere",
    "Translate", "Union", "union",
]
