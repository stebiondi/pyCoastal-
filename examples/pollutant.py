#!/usr/bin/env python3
"""
pollutant.py

2D pollutant advection–diffusion in a shallow pond:
  - UniformGrid from pyCoastal
  - Operators: gradient, laplacian
  - Upwind advection, central diffusion
  - Neumann (zero-flux) boundary conditions
  - Animated visualization of scalar field
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from pyCoastal.io import read_data
from pyCoastal.numerics.grid import UniformGrid
from pyCoastal.numerics.operators import gradient, laplacian

# -------------------------------------------------------------------
# 1) Load config
# -------------------------------------------------------------------
cfg = read_data("configs/pollutant.yaml")

# Domain and grid
Lx = cfg["grid"]["Lx"]
Ly = cfg["grid"]["Ly"]
Nx = cfg["grid"]["Nx"]
Ny = cfg["grid"]["Ny"]
dx, dy = Lx / Nx, Ly / Ny
grid = UniformGrid((Nx, Ny), (dx, dy))
X, Y = grid.Xc

# Physics
D  = cfg["physics"]["diffusivity"]
U0 = cfg["physics"]["U0"]
R  = cfg["physics"]["R"]

# Solver
dt      = 0.5 * min(dx, dy) / U0
t_final = cfg["solver"]["t_final"]
nt      = int(t_final / dt)
sigma   = cfg["solver"]["c0_sigma"]

# -------------------------------------------------------------------
# 2) Prescribed vortex velocity field
# -------------------------------------------------------------------
x0, y0 = Lx / 2, Ly / 2
u =  U0 *  (Y - y0) / R * np.exp(-((X - x0)**2 + (Y - y0)**2) / R**2)
v = -U0 *  (X - x0) / R * np.exp(-((X - x0)**2 + (Y - y0)**2) / R**2)

# -------------------------------------------------------------------
# 3) Initial Gaussian concentration at center
# -------------------------------------------------------------------
c     = np.exp(-(((X - x0)**2 + (Y - y0)**2) / sigma**2))
c_old = c.copy()
c_new = np.zeros_like(c)

# -------------------------------------------------------------------
# 4) Plot setup
# -------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(6, 3))
pcm = ax.pcolormesh(X, Y, c, cmap="viridis", vmin=0, vmax=1)
title = ax.set_title("t = 0.00 s")
ax.set_xlabel("x (m)")
ax.set_ylabel("y (m)")
cb = fig.colorbar(pcm, ax=ax, label="C")

# -------------------------------------------------------------------
# 5) Update function
# -------------------------------------------------------------------
def update(frame):
    global c, c_old, c_new

    cx, cy = gradient(c, grid)
    adv = u * cx + v * cy
    diff = D * laplacian(c, grid)
    c_new[:] = c + dt * (-adv + diff)

    # Zero-flux (Neumann) boundaries
    c_new[0, :]  = c_new[1, :]
    c_new[-1, :] = c_new[-2, :]
    c_new[:, 0]  = c_new[:, 1]
    c_new[:, -1] = c_new[:, -2]

    c_old, c = c, c_new

    pcm.set_array(c.ravel())
    title.set_text(f"t = {frame*dt:.2f} s")
    return [pcm, title]

# -------------------------------------------------------------------
# 6) Run animation
# -------------------------------------------------------------------
ani = animation.FuncAnimation(
    fig, update, frames=nt,
    interval=30, blit=False, repeat=False
)

plt.tight_layout()
plt.show()
