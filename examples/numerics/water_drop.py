#!/usr/bin/env python3
"""
water_drop.py

2D linear wave equation:
    η_tt = c² ∇²η

- Zero-Dirichlet BCs
- Initial Gaussian bump
- 2nd-order finite differences (explicit)
- Animated circular wave propagation
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from pyCoastal.io import read_data
from pyCoastal.numerics.grid import UniformGrid
from pyCoastal.numerics.operators import laplacian

# -------------------------------------------------------------------
# 1) Load config
# -------------------------------------------------------------------
cfg = read_data("../configs/water_drop.yaml")

Nx = cfg["grid"]["Nx"]
Ny = cfg["grid"]["Ny"]
Lx = cfg["grid"]["Lx"]
Ly = cfg["grid"]["Ly"]
dx, dy = Lx / Nx, Ly / Ny

c       = cfg["physics"]["wave_speed"]
sigma   = cfg["physics"]["sigma"]
CFL     = cfg["solver"]["CFL"]
t_final = cfg["solver"]["t_final"]
dt      = CFL * min(dx, dy) / c
nt      = int(t_final / dt)

# -------------------------------------------------------------------
# 2) Build grid and initialize
# -------------------------------------------------------------------
grid = UniformGrid((Nx, Ny), (dx, dy))
X, Y = grid.Xc

eta     = np.zeros((Nx, Ny))
eta_old = np.zeros_like(eta)
eta_new = np.zeros_like(eta)

# Gaussian initial bump at center
x0, y0 = Lx / 2, Ly / 2
eta[:] = np.exp(-((X - x0)**2 + (Y - y0)**2) / (2 * sigma**2))
eta_old[:] = eta.copy()

# -------------------------------------------------------------------
# 3) Plot setup
# -------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(6, 5))
pcm = ax.pcolormesh(X, Y, eta, cmap="RdBu_r", vmin=-0.5, vmax=0.5, shading="auto")
cb = fig.colorbar(pcm, ax=ax, label="η (m)")
title = ax.set_title("t = 0.00 s")
ax.set_xlabel("x")
ax.set_ylabel("y")

# -------------------------------------------------------------------
# 4) Time stepping
# -------------------------------------------------------------------
coef = (c * dt) ** 2

def update(frame):
    global eta, eta_old, eta_new
    t = frame * dt

    lap = laplacian(eta, grid)
    eta_new = 2 * eta - eta_old + coef * lap

    # Zero Dirichlet BCs
    eta_new[ 0, :] = 0.0
    eta_new[-1, :] = 0.0
    eta_new[:, 0 ] = 0.0
    eta_new[:, -1] = 0.0

    eta_old[:], eta[:] = eta, eta_new

    pcm.set_array(eta.ravel())
    title.set_text(f"t = {t:.2f} s")
    return [pcm, title]

ani = animation.FuncAnimation(
    fig, update, frames=nt, interval=30, blit=True, repeat=False
)

plt.tight_layout()
plt.show()
