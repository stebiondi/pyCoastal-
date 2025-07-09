#!/usr/bin/env python3
"""
wave_2D_irregular.py

2D depth‐averaged upwind‐in‐y wave propagation with an irregular
wave train (Hs, Tp) at the southern boundary.  Uses pyCoastal tools:
  - UniformGrid for the mesh
  - generate_irregular_wave for the boundary η(t)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from pyCoastal.numerics.grid import UniformGrid
from pyCoastal.tools.wave   import generate_irregular_wave

# -------------------------------------------------------------------
# 1) Choice of spectrum: "pm" or "jonswap"
# -------------------------------------------------------------------
spectrum = "jonswap"       # <— change to "jonswap" if you prefer JONSWAP
gamma    = 3.3        # only used when spectrum == "jonswap"

# -------------------------------------------------------------------
# 2) Physical parameters
# -------------------------------------------------------------------
g   = 9.81
h   = 5.0
c   = np.sqrt(g*h)

# -------------------------------------------------------------------
# 3) Grid
# -------------------------------------------------------------------
Nx, Ny = 200, 200
dx, dy = 1.0, 1.0
grid   = UniformGrid((Nx, Ny), (dx, dy))
X, Y   = grid.Xc

# -------------------------------------------------------------------
# 4) Generate irregular‐wave train at southern boundary
# -------------------------------------------------------------------
Hs, Tp   = 0.5, 3.0     # significant height & peak period
duration = 60.0
dt       = 0.1

# pass the spectrum choice and gamma through to your generator
t_irreg, eta_south = generate_irregular_wave(
    Hs, Tp,
    duration=duration,
    dt=dt,
    spectrum=spectrum,
    gamma=gamma
)

# -------------------------------------------------------------------
# 5) Initialize fields & observation
# -------------------------------------------------------------------
eta     = np.zeros((Nx, Ny))
eta_new = np.zeros_like(eta)

obs     = (Nx//2, Ny//2)
ts_obs  = []
times   = []

# -------------------------------------------------------------------
# 6) Plot setup
# -------------------------------------------------------------------
fig = plt.figure(figsize=(10,4))

ax1 = fig.add_subplot(1,2,1)
pcm = ax1.pcolormesh(X, Y, eta, cmap='viridis', vmin=-Hs, vmax=Hs)
ax1.set_title("η field")
ax1.set_xlabel("x"); ax1.set_ylabel("y")
fig.colorbar(pcm, ax=ax1, label="η (m)")

ax2 = fig.add_subplot(1,2,2)
ax2.set_xlim(0, duration)
ax2.set_ylim(-1.1*Hs, 1.1*Hs)
line_obs, = ax2.plot([], [], 'k-')
ax2.set_title(f"η at {obs[0]*dx:.0f} m, {obs[1]*dy:.0f} m")
ax2.set_xlabel("t (s)"); ax2.set_ylabel("η (m)")

# -------------------------------------------------------------------
# 7) Animation update
# -------------------------------------------------------------------
def update(n):
    global eta, eta_new

    t = n * dt
    times.append(t)

    # a) Imposed BC on southern edge y=0
    if n < len(eta_south):
        eta_new[:, 0] = eta_south[n]
    else:
        eta_new[:, 0] = 0.0

    # b) Upwind advection in y
    eta_new[:, 1:] = eta[:, 1:] - (c * dt / dy) * (eta[:, 1:] - eta[:, :-1])

    # c) Swap fields
    eta, eta_new = eta_new, eta

    # d) Update surface plot
    pcm.set_array(eta.ravel())

    # e) Record and update gauge
    ts_obs.append(eta[obs])
    line_obs.set_data(times, ts_obs)

    return [pcm, line_obs]

# -------------------------------------------------------------------
# 8) Launch animation
# -------------------------------------------------------------------
ani = animation.FuncAnimation(
    fig,
    update,
    frames = int(duration / dt),
    interval = 50,
    blit     = True,
    repeat   = False
)

plt.tight_layout()
plt.show()
