#!/usr/bin/env python3
"""
waves2D_irregular.py

2D depth-averaged wave model with irregular forcing at y=0.
Uses pyCoastal:
  - UniformGrid for spatial mesh
  - generate_irregular_wave for η(t) boundary condition
  - YAML config for setup
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from pyCoastal.io import read_data
from pyCoastal.numerics.grid import UniformGrid
from pyCoastal.tools.wave import generate_irregular_wave

# -------------------------------------------------------------------
# 1) Load configuration
# -------------------------------------------------------------------
cfg = read_data("configs/waves2D_irregular.yaml")

Nx, Ny = cfg["grid"]["nx"], cfg["grid"]["ny"]
dx, dy = cfg["grid"]["dx"], cfg["grid"]["dy"]
g      = cfg["physics"]["gravity"]
h      = cfg["physics"]["depth"]
c      = np.sqrt(g * h)

spectrum_type = cfg["forcing"]["type"]
gamma         = cfg["forcing"].get("gamma", 3.3)
Hs            = cfg["forcing"]["Hs"]
Tp            = cfg["forcing"]["Tp"]

dt       = cfg["solver"]["dt"]
duration = cfg["solver"]["duration"]
nt       = int(duration / dt)

obs_point = tuple(cfg["output"]["gauge"])

# -------------------------------------------------------------------
# 2) Create grid
# -------------------------------------------------------------------
grid = UniformGrid((Nx, Ny), (dx, dy))
X, Y = grid.Xc

# -------------------------------------------------------------------
# 3) Generate wave boundary forcing at y=0
# -------------------------------------------------------------------
t_vec, eta_bc = generate_irregular_wave(
    Hs=Hs, Tp=Tp,
    duration=duration,
    dt=dt,
    spectrum=spectrum_type,
    gamma=gamma
)

# -------------------------------------------------------------------
# 4) Initialize fields and storage
# -------------------------------------------------------------------
eta     = np.zeros((Nx, Ny))
eta_new = np.zeros_like(eta)

ts_obs = []
times  = []

# -------------------------------------------------------------------
# 5) Set up plots
# -------------------------------------------------------------------
fig = plt.figure(figsize=(10, 4))

# Field panel
ax1 = fig.add_subplot(1, 2, 1)
pcm = ax1.pcolormesh(X, Y, eta, cmap="viridis", vmin=-Hs, vmax=Hs)
ax1.set_title("η(x, y)")
ax1.set_xlabel("x (m)")
ax1.set_ylabel("y (m)")
fig.colorbar(pcm, ax=ax1, label="η (m)")

# Time-series panel
ax2 = fig.add_subplot(1, 2, 2)
ax2.set_xlim(0, duration)
ax2.set_ylim(-1.1 * Hs, 1.1 * Hs)
line, = ax2.plot([], [], "k-")
ax2.set_title(f"η at ({obs_point[0]*dx:.0f} m, {obs_point[1]*dy:.0f} m)")
ax2.set_xlabel("t (s)")
ax2.set_ylabel("η (m)")

# -------------------------------------------------------------------
# 6) Animation update
# -------------------------------------------------------------------
def update(n):
    global eta, eta_new

    t = n * dt
    times.append(t)

    # Dirichlet BC at y = 0
    eta_new[:, 0] = eta_bc[n] if n < len(eta_bc) else 0.0

    # Upwind advection
    eta_new[:, 1:] = eta[:, 1:] - (c * dt / dy) * (eta[:, 1:] - eta[:, :-1])

    eta, eta_new = eta_new, eta

    ts_obs.append(eta[obs_point])
    line.set_data(times, ts_obs)
    pcm.set_array(eta.ravel())

    return [pcm, line]

# -------------------------------------------------------------------
# 7) Run animation
# -------------------------------------------------------------------
ani = animation.FuncAnimation(
    fig, update, frames=nt,
    interval=50, blit=True, repeat=False
)

plt.tight_layout()
plt.show()

