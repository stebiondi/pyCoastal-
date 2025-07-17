#!/usr/bin/env python3
"""
waves2D.py

2D depth-averaged wave propagation model with observation points.

- Uniform water depth h
- Solves 1D linear advection in y for each x-column
- Periodic in x, wave forcing at y=0 boundary
- Forward Euler in time, upwind in space
- Real-time animation of the 2D η-field and time-series at specified gauges

Parameters are loaded from a YAML config file.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from pyCoastal.io import read_data
from pyCoastal.numerics.grid import UniformGrid

# -------------------------------------------------------------------
# 1) Load YAML configuration
# -------------------------------------------------------------------
cfg = read_data("configs/waves2D.yaml")

# Grid parameters
Nx = cfg["grid"]["nx"]
Ny = cfg["grid"]["ny"]
dx = cfg["grid"]["dx"]
dy = cfg["grid"]["dy"]

# Physical parameters
g  = cfg["physics"]["gravity"]
h  = cfg["physics"]["depth"]
c  = np.sqrt(g * h)

# Solver parameters
CFL     = cfg["solver"]["cfl"]
dt      = CFL * dy / c
t_final = cfg["solver"]["t_final"]
nt      = int(t_final / dt)

# Forcing parameters
A     = cfg["forcing"]["amplitude"]
T     = cfg["forcing"]["period"]
omega = 2 * np.pi / T

# Observation points
obs_points = [tuple(p) for p in cfg["output"]["obs_points"]]

# -------------------------------------------------------------------
# 2) Build the computational grid
# -------------------------------------------------------------------
grid = UniformGrid(shape=(Nx, Ny), spacing=(dx, dy))
X, Y = grid.Xc

# -------------------------------------------------------------------
# 3) Initialize fields and time series
# -------------------------------------------------------------------
eta     = np.zeros(grid.shape)
eta_new = np.zeros_like(eta)

time_series = {pt: [] for pt in obs_points}
time_axis   = []

# -------------------------------------------------------------------
# 4) Set up the figure and axes
# -------------------------------------------------------------------
fig = plt.figure(figsize=(10, 5))

# η-field panel
ax_field = fig.add_subplot(1, 2, 1)
cax = ax_field.pcolormesh(X, Y, eta, cmap='viridis', vmin=-A, vmax=+A)
ax_field.set_title('η(x, y)')
ax_field.set_xlabel('x (m)')
ax_field.set_ylabel('y (m)')
plt.colorbar(cax, ax=ax_field, label='η (m)')

# Time-series panel
ax_ts = fig.add_subplot(1, 2, 2)
ax_ts.set_xlim(0, t_final)
ax_ts.set_ylim(-1.1 * A, 1.1 * A)
ax_ts.set_title('Time series at gauges')
ax_ts.set_xlabel('t (s)')
ax_ts.set_ylabel('η (m)')

lines = {}
for pt in obs_points:
    ix, iy = pt
    x_real = grid.Xc[0][ix, iy]
    y_real = grid.Xc[1][ix, iy]
    label = f'({x_real:.1f} m, {y_real:.1f} m)'
    line, = ax_ts.plot([], [], label=label)
    lines[pt] = line

ax_ts.legend(loc='upper right')

# -------------------------------------------------------------------
# 5) Animation update function
# -------------------------------------------------------------------
def update(frame):
    global eta, eta_new

    t = frame * dt
    time_axis.append(t)

    # Dirichlet forcing at y=0
    eta_new[:, 0] = A * np.sin(omega * t)

    # Upwind advection along y
    eta_new[:, 1:] = eta[:, 1:] - (c * dt / dy) * (eta[:, 1:] - eta[:, :-1])

    # Swap arrays
    eta, eta_new = eta_new, eta

    # Record gauges
    for pt in obs_points:
        ix, iy = pt
        time_series[pt].append(eta[ix, iy])

    # Update η-field
    cax.set_array(eta.ravel())

    # Update time-series lines
    for pt, line in lines.items():
        line.set_data(time_axis, time_series[pt])

    return [cax] + list(lines.values())

# -------------------------------------------------------------------
# 6) Run the animation
# -------------------------------------------------------------------
ani = animation.FuncAnimation(
    fig, update, frames=nt,
    interval=50, blit=True, repeat=False
)

plt.tight_layout()
plt.show()
