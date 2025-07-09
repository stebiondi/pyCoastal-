"""
pyCoastal.numerics

Numerical infrastructure for coastal‐flow solvers:
  - grid:      cell/face data structures
  - scheme:    time‐integration and spatial‐discretization schemes
  - solver:    high‐level runner tying together grid, physics, BCs, and time stepping
  - operators: common numerical operators (e.g. Laplacian, advection, etc.)
"""
__all__ = ["grid", "scheme", "solver", "operators", "boundary", "domain"] 
