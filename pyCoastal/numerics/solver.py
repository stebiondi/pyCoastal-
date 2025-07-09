"""
High‐level Solver: ties together grid, physics, boundary conditions, and time‐integrator.
"""
import numpy as np
from .scheme import EulerIntegrator

class Solver:
    def __init__(self, grid, physics, bc, integrator=None):
        """
        grid      : an instance of numerics.grid.UniformGrid
        physics   : module or object with 'rhs(state, t)' method
        bc        : boundary‐condition object with 'apply(state, t)'
        integrator: TimeIntegrator (defaults to first‐order Euler)
        """
        self.grid       = grid
        self.physics    = physics
        self.bc         = bc
        self.integrator = integrator or EulerIntegrator(dt=physics.dt)

        # Initialize state variables dict
        self.state = physics.initialize_state(grid)

    def run(self, t0, t_end, callback=None):
        """
        March from t0 to t_end.
        callback(state, t) will be called after each step (for output/plotting).
        """
        t = t0
        while t < t_end - 1e-12:
            # apply boundary conditions
            self.bc.apply(self.state, t)

            # one time step
            self.state, t = self.integrator.step(
                self.state, self.physics.rhs, t, grid=self.grid, bc=self.bc
            )

            # user hook
            if callback is not None:
                callback(self.state, t)

        return self.state
