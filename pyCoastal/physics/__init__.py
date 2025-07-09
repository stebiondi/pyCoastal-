# pycoastal/physics/__init__.py

"""
pycoastal.physics
=================

Governing equations and closure models for pyCoastal.

Submodules
----------
- shallow_water: 2D nonlinear shallow-water equations
- navier_stokes: 2D/3D incompressible Navier–Stokes
- turbulence:    eddy-viscosity closures (Smagorinsky, k–ε, k–ω)
"""

from .shallow_water import ShallowWater2D
from .turbulence    import SmagorinskyModel, KEpsilonModel, KOmegaModel
from .navier_stokes import initialize_state, rhs


__all__ = [
    "ShallowWater2D",
    "SmagorinskyModel",
    "KEpsilonModel",
    "KOmegaModel",
    "initialize_state",
    "rhs",
]
