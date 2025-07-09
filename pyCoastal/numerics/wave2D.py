#!/usr/bin/env python3
"""
wave2d.py

Simple 2D depth‐averaged wave propagation model with observation points:
  - Uniform depth h
  - Solves 1D linear advection in y for each x‐column
  - Periodic in x, wave forcing at y=0 boundary
  - Forward Euler in time, upwind in space
  - Real‐time animation of the 2D η‐field + time‐series at user‐defined observation points
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Physical parameters
g   = 9.81    # gravity (m/s^2)
h   = 5.0     # water depth (m)
c   = np.sqrt(g * h)  # wave celerity

# Grid parameters
Nx  = 1000     # points in x
Ny  = 1000     # points in y
dx  = dy = 0.1 # grid spacing (m)

# Time‐stepping
CFL     = 1.0
dt      = CFL * dy / c
t_final = 20.0
nt      = int(t_final / dt)

# Wave forcing at y=0
A       = 0.2     # amplitude (m)
T       = 2.3       # period (s)
omega   = 2.0 * np.pi / T

# Observation points (enter as (ix, iy) grid‐indices)
# e.g. pick three points: near inlet, mid‐domain, downstream
obs_points = [
    (20, 50),   # (x=20*dx, y=50*dy)
]

# Initialize fields
eta     = np.zeros((Nx, Ny))
eta_new = np.zeros_like(eta)

# Prepare storage for time series at obs points
time_series = {pt: [] for pt in obs_points}
time_axis   = []

# Domain coordinates (for plotting)
x = np.linspace(0, (Nx-1)*dx, Nx)
y = np.linspace(0, (Ny-1)*dy, Ny)
X, Y = np.meshgrid(x, y, indexing='ij')

# Set up figure with two panels:
#   left: 2D colorplot of η(x,y)
#   right: time‐series of η at each obs point
fig = plt.figure(figsize=(10,5))
ax_field = fig.add_subplot(1,2,1)
cax = ax_field.pcolormesh(X, Y, eta, cmap='viridis', vmin=-A, vmax=+A)
ax_field.set_xlabel('x (m)')
ax_field.set_ylabel('y (m)')
ax_field.set_title('η(x,y)')
plt.colorbar(cax, ax=ax_field, label='η (m)')

ax_ts = fig.add_subplot(1,2,2)
ax_ts.set_xlabel('Time (s)')
ax_ts.set_ylabel('η (m)')
ax_ts.set_title('Time Series at Observation Points')
lines = {}
for pt in obs_points:
    (ix, iy) = pt
    # label with real‐world coords
    label = f'({ix*dx:.0f}m, {iy*dy:.0f}m)'
    line, = ax_ts.plot([], [], label=label)
    lines[pt] = line

ax_ts.legend(loc='upper right')
ax_ts.set_xlim(0, t_final)
ax_ts.set_ylim(-1.1*A, 1.1*A)

def update(frame):
    global eta, eta_new

    t = frame * dt
    time_axis.append(t)

    # 1) apply forcing at y=0
    eta_new[:, 0] = A * np.sin(omega * t)

    # 2) upwind advection in y for j=1..Ny-1
    eta_new[:, 1:] = eta[:, 1:] - (c * dt / dy) * (eta[:, 1:] - eta[:, :-1])

    # 3) periodic in x (no change since no x‐advection)
    #    but if you had x‐advection you'd wrap around here

    # 4) swap fields
    eta, eta_new = eta_new, eta

    # 5) update time-series for each obs point
    for pt in obs_points:
        (ix, iy) = pt
        time_series[pt].append(eta[ix, iy])

    # 6) update the 2D field plot
    cax.set_array(eta.ravel())

    # 7) update each time‐series line
    for pt, line in lines.items():
        ys = time_series[pt]
        line.set_data(time_axis, ys)

    return [cax] + list(lines.values())

# Create animation
ani = animation.FuncAnimation(
    fig,
    update,
    frames=nt,
    interval=50,
    blit=True,
    repeat=False
)

plt.tight_layout()
plt.show()
