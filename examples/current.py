#!/usr/bin/env python3
"""
current.py

2D passive x-direction current advection with:
  - Dirichlet inlet (west), Neumann outlet (east)
  - UniformGrid mesh
  - Upwind scheme for u_t + c·u_x = 0
  - Real-time animation + time series at a gauge
  - Parameters from YAML config
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from pyCoastal.io import read_data
from pyCoastal.numerics.grid import UniformGrid
from pyCoastal.numerics.boundary import DirichletBC, NeumannBC, BoundaryManager

# -------------------------------------------------------------------
# 1) Load config
# -------------------------------------------------------------------
cfg = read_data("configs/current.yaml")

Nx, Ny = cfg["grid"]["nx"], cfg["grid"]["ny"]
dx, dy = cfg["grid"]["dx"], cfg["grid"]["dy"]
c0     = cfg["physics"]["speed"]
CFL    = cfg["solver"]["cfl"]
t_final = cfg["solver"]["t_final"]
dt     = CFL * dx / c0
nt     = int(t_final / dt)
ix, iy = tuple(cfg["output"]["gauge"])

# -------------------------------------------------------------------
# 2) Build grid and initialize fields
# -------------------------------------------------------------------
grid = UniformGrid((Nx, Ny), (dx, dy))
X, Y = grid.Xc
u     = np.zeros((Nx, Ny))
u_new = np.zeros_like(u)
fields = {"u": u}

# -------------------------------------------------------------------
# 3) Boundary conditions
# -------------------------------------------------------------------
bc_mgr = BoundaryManager()
bc_mgr.add(DirichletBC("west", ["u"], value=c0))      # Inlet
bc_mgr.add(NeumannBC("east", ["u"], gradient=0.0))    # Outlet

# -------------------------------------------------------------------
# 4) Plot setup
# -------------------------------------------------------------------
fig = plt.figure(figsize=(10, 4))
ax1 = fig.add_subplot(1, 2, 1)
pcm = ax1.pcolormesh(X, Y, u, cmap='viridis', vmin=0, vmax=c0)
ax1.set_title("u field")
ax1.set_xlabel("x")
ax1.set_ylabel("y")
fig.colorbar(pcm, ax=ax1, label="u (m/s)")

ax2 = fig.add_subplot(1, 2, 2)
ax2.set_xlim(0, t_final)
ax2.set_ylim(0, 1.1 * c0)
line_ts, = ax2.plot([], [], "k-")
ax2.set_title(f"u at ({ix}, {iy})")
ax2.set_xlabel("t (s)")
ax2.set_ylabel("u (m/s)")

times = []
ts_obs = []

# -------------------------------------------------------------------
# 5) Update function
# -------------------------------------------------------------------
def update(n):
    global u, u_new
    t = n * dt
    times.append(t)

    bc_mgr.apply_all(fields, grid, t)

    u_new[1:, :] = u[1:, :] - (c0 * dt / dx) * (u[1:, :] - u[:-1, :])
    u_new[0, :] = u[0, :]
    u_new[-1, :] = u[-1, :]  # Neumann handled explicitly via copy

    u, u_new = u_new, u
    fields["u"] = u

    ts_obs.append(u[ix, iy])
    line_ts.set_data(times, ts_obs)

    pcm.set_array(u.ravel())
    return [pcm, line_ts]

# -------------------------------------------------------------------
# 6) Animate
# -------------------------------------------------------------------
ani = animation.FuncAnimation(
    fig, update, frames=nt,
    interval=50, blit=False, repeat=False
)

plt.tight_layout()
plt.show()
