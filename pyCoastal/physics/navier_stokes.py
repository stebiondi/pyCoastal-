# pyCoastal/physics/navier_stokes.py

import numpy as np

def initialize_state(grid):
    """Return a dict with zero u,v,p fields on the grid."""
    shape = grid.shape
    return {
        "u": np.zeros(shape),
        "v": np.zeros(shape),
        "p": np.zeros(shape),
    }

def rhs(state, t, grid, bc_mgr, ν=1e-3):
    """
    Compute convection + diffusion terms for u,v (no pressure).
    Enforce velocity BCs on the resulting rhs arrays, then return them.
    """
    u = state["u"]
    v = state["v"]
    dx, dy = grid.spacing

    # --- convection, centered differences ---
    ux = (np.roll(u, -1, axis=1) - np.roll(u, +1, axis=1)) / (2*dx)
    uy = (np.roll(u, -1, axis=0) - np.roll(u, +1, axis=0)) / (2*dy)
    vx = (np.roll(v, -1, axis=1) - np.roll(v, +1, axis=1)) / (2*dx)
    vy = (np.roll(v, -1, axis=0) - np.roll(v, +1, axis=0)) / (2*dy)
    conv_u = u*ux + v*uy
    conv_v = u*vx + v*vy

    # --- viscous Laplacian ---
    u_xx = (np.roll(u,-1,1) - 2*u + np.roll(u,1,1)) / dx**2
    u_yy = (np.roll(u,-1,0) - 2*u + np.roll(u,1,0)) / dy**2
    v_xx = (np.roll(v,-1,1) - 2*v + np.roll(v,1,1)) / dx**2
    v_yy = (np.roll(v,-1,0) - 2*v + np.roll(v,1,0)) / dy**2
    diff_u = ν*(u_xx + u_yy)
    diff_v = ν*(v_xx + v_yy)

    # --- assemble RHS ---
    rhs_u = -conv_u + diff_u
    rhs_v = -conv_v + diff_v

    # --- enforce velocity BCs on RHS ---
    fields_rhs = {"u": rhs_u, "v": rhs_v}
    bc_mgr.apply_all(fields_rhs, grid, t)

    # no pressure RHS
    rhs_p = np.zeros_like(u)
    return {"u": rhs_u, "v": rhs_v, "p": rhs_p}
