#!/usr/bin/env python3
"""
viscous_fluid.py

2D incompressible Navier–Stokes simulation with:
- Two counter-rotating vortices
- Central-difference convection and diffusion
- Forward Euler time integration
- Periodic boundary conditions
- Real-time animation of speed field
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from pyCoastal.io import read_data
from pyCoastal.numerics.grid import UniformGrid
from pyCoastal.numerics.boundary import BoundaryManager
from pyCoastal.physics.navier_stokes import initialize_state, rhs

def main():
    # --------------------------------------------------------------
    # 1) Load configuration
    # --------------------------------------------------------------
    cfg = read_data("../configs/viscous_fluid.yaml")

    Nx = cfg["grid"]["Nx"]
    Ny = cfg["grid"]["Ny"]
    Lx = cfg["grid"]["Lx"]
    Ly = cfg["grid"]["Ly"]
    dx, dy = Lx / (Nx - 1), Ly / (Ny - 1)
    grid = UniformGrid((Nx, Ny), (dx, dy))
    X, Y = grid.Xc

    ν     = cfg["physics"]["viscosity"]
    U0    = cfg["physics"]["background_speed"]
    sigma = cfg["physics"]["vortex_sigma"]

    dt      = cfg["solver"]["dt"]
    t_final = cfg["solver"]["t_final"]
    nt      = int(t_final / dt)

    # --------------------------------------------------------------
    # 2) Initial state: uniform + vortex perturbation
    # --------------------------------------------------------------
    bc_mgr = BoundaryManager()  # periodic
    state = initialize_state(grid)
    state["u"] += U0

    xc, yc = Lx / 2, Ly / 2
    r2 = (X - xc) ** 2 + (Y - yc) ** 2
    vortex = np.exp(-r2 / sigma ** 2)
    state["u"] += -(Y - yc) * vortex
    state["v"] +=  (X - xc) * vortex

    # --------------------------------------------------------------
    # 3) Plot setup
    # --------------------------------------------------------------
    fig, ax = plt.subplots()
    speed = np.sqrt(state["u"]**2 + state["v"]**2)
    pcm = ax.pcolormesh(X, Y, speed, cmap="inferno", shading="auto")
    fig.colorbar(pcm, ax=ax, label="speed")
    ax.set_aspect("equal")
    title = ax.set_title("t = 0.00 s")

    # --------------------------------------------------------------
    # 4) Update function
    # --------------------------------------------------------------
    def update(frame):
        nonlocal state
        t = frame * dt

        # RHS from convection + diffusion
        R = rhs(state, t, grid, bc_mgr, ν=ν)

        # Forward Euler step
        state["u"] += dt * R["u"]
        state["v"] += dt * R["v"]

        # Enforce mean U0 drift (prevent bulk slowing)
        state["u"] -= (state["u"].mean() - U0)

        # Update plot
        speed = np.sqrt(state["u"]**2 + state["v"]**2)
        pcm.set_array(speed.ravel())
        title.set_text(f"t = {t:.2f} s")
        return [pcm, title]

    ani = animation.FuncAnimation(
        fig, update, frames=nt, interval=30, blit=True, repeat=False
    )
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
