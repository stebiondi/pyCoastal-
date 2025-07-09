# pycoastal/boundary.py

"""
pycoastal.boundary
------------------
Boundary‐condition classes for pyCoastal solvers.  
Defines Dirichlet, Neumann, Wall (no‐flow), and Sponge (damping) BCs.

Each BC must implement `apply(fields, grid, t)` where:
  - fields: dict of solution arrays, e.g. {"eta": η_array, "q": q_array, ...}
  - grid:   an object providing boundary indices, e.g. grid.boundary_indices["west"]
  - t:      current simulation time (s)
"""

import numpy as np
from typing import Callable, Union, Sequence, Mapping

class BoundaryCondition:
    """Abstract base class for all boundary conditions."""
    def __init__(self, location: str, var_names: Sequence[str]):
        """
        Args:
            location: name of boundary (e.g. "west","east","south","north")
            var_names: list of field keys this BC applies to, e.g. ["eta"], ["q"], etc.
        """
        self.location = location
        self.vars = var_names

    def apply(self, fields: Mapping[str, np.ndarray], grid, t: float):
        """Apply this BC to the given fields at time t."""
        raise NotImplementedError("BoundaryCondition.apply must be overridden")


class DirichletBC(BoundaryCondition):
    """Enforce a prescribed value at a boundary (ghost cell or face)."""
    def __init__(self, location, var_names, value):
        super().__init__(location, var_names)
        self.value = value

    def apply(self, fields, grid, t):
        val = self.value(t) if callable(self.value) else self.value
        idx = grid.boundary_indices[self.location]
        for vn in self.vars:
            flat = fields[vn].ravel()
            flat[idx] = val


class NeumannBC(BoundaryCondition):
    """Zero‐gradient (or specified) Neumann BC: copy interior value to boundary."""
    def __init__(self, location, var_names, gradient=0.0):
        super().__init__(location, var_names)
        self.gradient = gradient

    def apply(self, fields, grid, t):
        grad = self.gradient(t) if callable(self.gradient) else self.gradient
        bd_idx, inner_idx = grid.neumann_indices(self.location)
        for vn in self.vars:
            flat = fields[vn].ravel()
            # pick the proper cell spacing direction
            dx = grid.spacing[0] if self.location in ("west","east") else grid.spacing[1]
            flat[bd_idx] = flat[inner_idx] + grad * dx


class WallBC(BoundaryCondition):
    """No‐flow wall: set normal velocity (or flux) to zero."""
    def __init__(self, location, var_names):
        super().__init__(location, var_names)

    def apply(self, fields, grid, t):
        idx = grid.boundary_indices[self.location]
        for vn in self.vars:
            flat = fields[vn].ravel()
            flat[idx] = 0.0



class SpongeBC(BoundaryCondition):
    """Damping (sponge) layer: relax solution toward reference (often zero)."""
    def __init__(
        self,
        location: str,
        var_names: Sequence[str],
        damping: Union[float, Callable[[float, int], float]]
    ):
        super().__init__(location, var_names)
        self.damping = damping

    def apply(self, fields, grid, t):
        # grab the flat indices along that boundary
        idx = grid.sponge_indices[self.location]

        for vn in self.vars:
            arr = fields[vn]
            # view it as a 1-D array
            flat = arr.reshape(-1)

            if callable(self.damping):
                # spatially‐varying damping
                for k in idx:
                    α = self.damping(t, k)
                    flat[k] *= α
            else:
                # constant α on the whole strip
                flat[idx] *= self.damping


class BoundaryManager:
    """Holds and applies a list of boundary conditions each timestep."""
    def __init__(self, bcs: Sequence[BoundaryCondition] = ()):
        self._bcs = list(bcs)

    def add(self, bc: BoundaryCondition):
        self._bcs.append(bc)

    def apply_all(self, fields: Mapping[str, np.ndarray], grid, t: float):
        """
        Apply every registered BC to the solution fields at time t.

        Args:
            fields: dict of field arrays (modified in place)
            grid: solution grid, with boundary_indices, sponge_indices, etc.
            t: current time
        """
        for bc in self._bcs:
            bc.apply(fields, grid, t)
