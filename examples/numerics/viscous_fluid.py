"""
viscous_fluid.py

A simple 2D incompressible Navier–Stokes solver (projection omitted) demonstrating
the roll-up and advection of a pair of counter-rotating vortices in a uniform grid.

• Governing equations (no pressure solve):
    u_t + (u·∇)u = ν ∇²u
    v_t + (u·∇)v = ν ∇²v

• Spatial discretization: second-order central differences for convection and diffusion
• Time stepping: forward (explicit) Euler
• Boundary conditions: free-slip (zero normal velocity) on all four sides
• Initial condition: two Gaussian vortices of opposite sign
• Visualization: real-time animation of vorticity field

This demo illustrates the basic mechanisms of vortex interaction and viscous diffusion
on a Cartesian mesh and can be extended with pressure projection, buoyancy forcing,
or more advanced time‐integration schemes.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from pyCoastal.numerics.grid          import UniformGrid
from pyCoastal.physics.navier_stokes  import initialize_state, rhs
from pyCoastal.numerics.boundary               import BoundaryManager

def main():
    # 1) Build a periodic grid
    Nx, Ny = 100, 100
    Lx, Ly = 2.0, 2.0
    dx, dy = Lx/(Nx-1), Ly/(Ny-1)
    grid   = UniformGrid((Nx, Ny), (dx, dy))
    X, Y   = grid.Xc

    # 2) Create a dummy BC manager (no BCs → periodic!)
    bc_mgr = BoundaryManager()

    # 3) Initialize state and superimpose a uniform U0 flow
    U0 = 1.5   # background advection speed
    state = initialize_state(grid)
    state["u"] += U0

    # 4) Add a small Gaussian vortex perturbation in the center
    xc, yc = Lx/2, Ly/2
    sigma  = 0.2
    r2 = (X - xc)**2 + (Y - yc)**2
    vortex = np.exp(-r2/sigma**2)
    # swirl about center:
    state["u"] +=  -(Y - yc)*vortex
    state["v"] +=   (X - xc)*vortex

    # 5) Time‐step parameters
    ν  = 1e-3    # viscosity
    dt = 0.001   # Δt
    T  = 4.0     # total sim time
    nt = int(T/dt)

    # 6) Set up the plot
    fig, ax = plt.subplots()
    speed = np.sqrt(state["u"]**2 + state["v"]**2)
    pcm   = ax.pcolormesh(X, Y, speed, cmap='inferno', shading='auto')
    fig.colorbar(pcm, ax=ax, label="speed")
    ax.set_aspect("equal")
    txt = ax.set_title("t = 0.00")

    # 7) Time‐stepping / animation callback
    def update(frame):
        nonlocal state
        t = frame*dt

        # (a) compute RHS (conv + diff)
        R = rhs(state, t, grid, bc_mgr, ν=ν)

        # (b) advance with forward Euler
        state["u"] += dt * R["u"]
        state["v"] += dt * R["v"]

        # (c) add background drift back in (so U0 stays constant)
        state["u"] -= (state["u"].mean() - U0)

        # (d) update plot
        s = np.sqrt(state["u"]**2 + state["v"]**2)
        pcm.set_array(s.ravel())
        txt.set_text(f"t = {t:.2f}")
        return [pcm, txt]

    ani = animation.FuncAnimation(
        fig, update, frames=nt, interval=30, blit=True, repeat=False
    )
    plt.show()

if __name__=="__main__":
    main()
