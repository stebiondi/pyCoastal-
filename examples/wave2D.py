#!/usr/bin/env python3
"""
wave2d.py

Simple 2D depth-averaged wave propagation model with observation points:
  - Uniform depth h
  - Solves 1D linear advection in y for each x-column
  - Periodic in x, wave forcing at y=0 boundary
  - Forward Euler in time, upwind in space
  - Real-time animation of the 2D η-field + time-series at user-defined observation points
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from pyCoastal.numerics.grid import UniformGrid

# -------------------------------------------------------------------
# 1) Physical parameters
# -------------------------------------------------------------------
g   = 9.81     # gravity (m/s^2)
h   = 5.0      # water depth (m)
c   = np.sqrt(g * h)

# -------------------------------------------------------------------
# 2) Build the grid
# -------------------------------------------------------------------
Nx, Ny = 100, 100         # number of cells in x & y
dx, dy = 0.1, 0.1         # grid spacing (m)
grid   = UniformGrid(shape=(Nx, Ny), spacing=(dx, dy))
X, Y   = grid.Xc          # meshgrid of cell-centers

# -------------------------------------------------------------------
# 3) Time-stepping parameters
# -------------------------------------------------------------------
CFL     = 1.0
dt      = CFL * dy / c
t_final = 20.0
nt      = int(t_final / dt)

# -------------------------------------------------------------------
# 4) Wave forcing at southern boundary (y=0)
# -------------------------------------------------------------------
A     = 0.2              # amplitude (m)
T     = 2.3              # period (s)
omega = 2*np.pi / T      # angular frequency

# -------------------------------------------------------------------
# 5) Observation (gauge) locations (in grid indices)
# -------------------------------------------------------------------
obs_points = [ (20, 50), ]

# -------------------------------------------------------------------
# 6) Initialize fields and storage
# -------------------------------------------------------------------
eta     = np.zeros(grid.shape)
eta_new = np.zeros_like(eta)

time_series = {pt: [] for pt in obs_points}
time_axis   = []

# -------------------------------------------------------------------
# 7) Set up the figure and axes
# -------------------------------------------------------------------
fig = plt.figure(figsize=(10,5))

# 7a) η-field panel
ax_field = fig.add_subplot(1,2,1)
cax = ax_field.pcolormesh(X, Y, eta, cmap='viridis', vmin=-A, vmax=+A)
ax_field.set_title('η(x,y)')
ax_field.set_xlabel('x (m)')
ax_field.set_ylabel('y (m)')
plt.colorbar(cax, ax=ax_field, label='η (m)')

# 7b) Time-series panel
ax_ts = fig.add_subplot(1,2,2)
ax_ts.set_xlim(0, t_final)
ax_ts.set_ylim(-1.1*A, 1.1*A)
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
# 8) Update function for animation
# -------------------------------------------------------------------
def update(frame):
    global eta, eta_new

    t = frame * dt
    time_axis.append(t)

    # 8a) Southern Dirichlet forcing: η(x, y=0)
    eta_new[:, 0] = A * np.sin(omega * t)

    # 8b) Upwind advection in y for j=1…Ny-1
    eta_new[:, 1:] = eta[:, 1:] - (c * dt / dy) * (eta[:, 1:] - eta[:, :-1])

    # 8c) (If needed) periodic in x step would go here

    # 8d) Swap new/old arrays
    eta, eta_new = eta_new, eta

    # 8e) Record each gauge
    for pt in obs_points:
        ix, iy = pt
        time_series[pt].append(eta[ix, iy])

    # 8f) Update the η-field plot
    cax.set_array(eta.ravel())

    # 8g) Update the time-series lines
    for pt, line in lines.items():
        line.set_data(time_axis, time_series[pt])

    return [cax] + list(lines.values())

# -------------------------------------------------------------------
# 9) Launch animation
# -------------------------------------------------------------------
ani = animation.FuncAnimation(
    fig, update, frames=nt,
    interval=50, blit=True, repeat=False
)

plt.tight_layout()
plt.show()
