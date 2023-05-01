from warnings import warn

from .control_affine_system import ControlAffineSystem
from .kinematic_single_track_car import KSCar

__all__ = [
    "ControlAffineSystem",
    "KSCar",
    "STCar",
]

try:
    from .f16 import F16  # noqa

    __all__.append("F16")
except ImportError:
    warn("Could not import F16 module; is AeroBench installed")
