"""
Time‐integration and spatial discretization schemes.
"""
import numpy as np

class TimeIntegrator:
    """Base class for explicit time integrators (e.g. Euler, RK2, RK3)."""
    def __init__(self, dt):
        self.dt = dt

    def step(self, state, rhs, t, **kwargs):
        """
        Advance 'state' one time step.
        state : dict of numpy arrays (e.g. { "u":…, "h":…, … })
        rhs   : function(state, t, **kwargs) → dict of tendencies
        t     : current time
        returns (new_state, new_time)
        """
        raise NotImplementedError("choose EulerIntegrator or RK2Integrator")

class EulerIntegrator(TimeIntegrator):
    def step(self, state, rhs, t, **kwargs):
        rates = rhs(state, t, **kwargs)
        new_state = {var: state[var] + self.dt * rates[var]
                     for var in state}
        return new_state, t + self.dt


class SSPRK2Integrator(TimeIntegrator):
    def step(self, state, rhs, t, **kwargs):
        # Shu–Osher 2nd‐order TVD RK
        k1 = rhs(state, t, **kwargs)
        mid = {var: state[var] + self.dt*k1[var] for var in state}
        k2 = rhs(mid, t+self.dt, **kwargs)
        new_state = {var: 0.5*(state[var] + mid[var] + self.dt*k2[var])
                     for var in state}
        return new_state, t + self.dt


def central_difference(phi, spacing, axis):
    """Second‐order central difference in 'axis'."""
    d = spacing[axis]
    return (np.roll(phi, -1, axis=axis) - np.roll(phi, +1, axis=axis)) / (2*d)


def upwind(phi, vel, spacing, axis):
    """
    First‐order upwind scheme for advection term vel·∇phi.
    vel: array of the same shape giving velocity component in 'axis'.
    """
    d = spacing[axis]
    phi_up = np.where(vel > 0,
                      (phi - np.roll(phi, +1, axis=axis))/d,
                      (np.roll(phi, -1, axis=axis) - phi)/d)
    return vel * phi_up
