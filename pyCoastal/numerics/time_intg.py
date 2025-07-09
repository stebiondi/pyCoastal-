import numpy as np

# time_intg.py
# A collection of time-integration schemes for 2D fields.
# Each function advances the solution one time step.

def euler_step(u, t, dt, rhs):
    """
    Forward Euler scheme (1st-order). u_{n+1} = u_n + dt * rhs(u_n, t)
    Parameters:
        u   : 2D numpy array, current solution field
        t   : float, current time
        dt  : float, time step size
        rhs : function(u, t) -> 2D array of same shape as u, computes time derivative
    Returns:
        u_new : 2D array, solution at t + dt
    """
    return u + dt * rhs(u, t)


def rk2_step(u, t, dt, rhs):
    """
    Heun's method / explicit midpoint (2nd-order Runge-Kutta).
    k1 = rhs(u, t)
    k2 = rhs(u + dt * k1, t + dt)
    u_{n+1} = u_n + (dt/2) * (k1 + k2)
    """
    k1 = rhs(u, t)
    k2 = rhs(u + dt * k1, t + dt)
    return u + dt * (0.5 * k1 + 0.5 * k2)


def rk4_step(u, t, dt, rhs):
    """
    Classic 4th-order Runge-Kutta.
    k1 = rhs(u, t)
    k2 = rhs(u + dt/2 * k1, t + dt/2)
    k3 = rhs(u + dt/2 * k2, t + dt/2)
    k4 = rhs(u + dt * k3, t + dt)
    u_{n+1} = u_n + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
    """
    k1 = rhs(u, t)
    k2 = rhs(u + dt * k1 * 0.5, t + dt * 0.5)
    k3 = rhs(u + dt * k2 * 0.5, t + dt * 0.5)
    k4 = rhs(u + dt * k3, t + dt)
    return u + dt * (k1 + 2*k2 + 2*k3 + k4) / 6


def ab2_step(u, u_prev, t, dt, rhs):
    """
    Second-order Adams-Bashforth.
    Requires solution at two previous times: u_n and u_{n-1}.
    u_{n+1} = u_n + dt * (3/2 * rhs(u_n,t) - 1/2 * rhs(u_{n-1}, t-dt))
    """
    f_n = rhs(u, t)
    f_nm1 = rhs(u_prev, t - dt)
    return u + dt * (1.5 * f_n - 0.5 * f_nm1)


def rk3_ssp_step(u, t, dt, rhs):
    """
    Strong Stability-Preserving 3rd-order Runge-Kutta (SSP RK3).
    Stage 1: u1 = u + dt * rhs(u, t)
    Stage 2: u2 = 0.75 u + 0.25 (u1 + dt * rhs(u1, t+dt))
    Stage 3: u_{n+1} = (1/3) u + (2/3) (u2 + dt * rhs(u2, t+0.5*dt))
    """
    k1 = rhs(u, t)
    u1 = u + dt * k1
    k2 = rhs(u1, t + dt)
    u2 = 0.75*u + 0.25*(u1 + dt*k2)
    k3 = rhs(u2, t + 0.5*dt)
    return (1/3)*u + (2/3)*(u2 + dt*k3)

# Dictionary of schemes for easy lookup
time_integrators = {
    'euler': euler_step,
    'rk2':    rk2_step,
    'rk3':    rk3_ssp_step,
    'rk4':    rk4_step,
    'ab2':    ab2_step,
}
