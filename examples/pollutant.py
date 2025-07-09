#!/usr/bin/env python3
"""
pollutant.py

2D pollutant advection–diffusion in a shallow pond:
  - UniformGrid for mesh
  - Operators: gradient, divergence, laplacian
  - Upwind for advection; central‐diffusion
  - Zero‐flux (Neumann) walls
  - Animation of concentration field
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from pyCoastal.numerics.grid      import UniformGrid
from pyCoastal.numerics.operators import gradient, laplacian

# 1) Physical & numerical parameters
Lx, Ly = 2.0, 1.0         # pond size [m]
Nx, Ny = 200, 100         # grid
dx, dy = Lx/Nx, Ly/Ny
grid   = UniformGrid((Nx, Ny), (dx, dy))
X, Y   = grid.Xc

D      = 1e-3             # diffusivity [m²/s]
U0     = 0.2              # characteristic speed [m/s]
dt     = 0.5 * min(dx,dy)/U0
t_final= 5.0
nt     = int(t_final/dt)

# 2) Prescribed circulation: a single vortex
x0, y0 = Lx/2, Ly/2
R      = 0.4
u =  U0 *  ( Y - y0 )/R * np.exp(-((X-x0)**2 + (Y-y0)**2)/(R**2))
v = -U0 *  ( X - x0 )/R * np.exp(-((X-x0)**2 + (Y-y0)**2)/(R**2))

# 3) Initialize concentration: Gaussian drop in center
c      = np.exp(-(((X-x0)**2 + (Y-y0)**2)/(0.05**2)))
c_old  = c.copy()
c_new  = np.zeros_like(c)

# 4) Set up figure
fig, ax = plt.subplots(figsize=(6,3))
pcm = ax.pcolormesh(X, Y, c, cmap="viridis", vmin=0, vmax=1)
ax.set_title("Pollutant concentration")
ax.set_xlabel("x"); ax.set_ylabel("y")
cb = fig.colorbar(pcm, ax=ax, label="c")

def update(frame):
    global c, c_old, c_new
    # (a) compute advective fluxes via upwind
    cx, cy = gradient(c, grid)
    adv = u*cx + v*cy

    # (b) diffusion term
    diff = D * laplacian(c, grid)

    # (c) time‐stepping: forward Euler
    c_new[:] = c + dt * (-adv + diff)

    # (d) enforce zero‐flux (Neumann) on walls:
    c_new[0 ,:] = c_new[1 ,:]
    c_new[-1,:] = c_new[-2,:]
    c_new[:, 0] = c_new[:, 1 ]
    c_new[:,-1] = c_new[:,-2]

    # rotate
    c_old, c = c, c_new

    # update plot
    pcm.set_array(c.ravel())
    ax.set_title(f"t = {frame*dt:.2f} s")
    return [pcm]

ani = animation.FuncAnimation(
    fig, update, frames=nt,
    interval=30, blit=True, repeat=False
)

plt.tight_layout()
plt.show()
