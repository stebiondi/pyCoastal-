#!/usr/bin/env python3
"""
water_drop.py

2D linear wave equation demo ("droplet in the pond"):
    η_tt = c^2 ∇²η
with
    - zero‐Dirichlet walls (η=0 at all edges)
    - initial condition: Gaussian bump at center
    - explicit 2nd‐order finite differences in space & time
    - animation of η(x,y,t) showing expanding circular waves
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# import our UniformGrid for coordinates (no BC machinery needed here)
from pyCoastal.numerics.grid import UniformGrid
from pyCoastal.numerics.operators import laplacian
# -------------------------------------------------------------------
# 1) Physical & numerical parameters
# -------------------------------------------------------------------
c       = 1.0        # wave speed
Lx, Ly  = 2.0, 2.0   # domain size [m]
Nx, Ny  = 500, 500  # grid resolution
dx, dy  = Lx/Nx, Ly/Ny
CFL     = 0.5        # for stability must have c*dt/dx < 1/√2 in 2D
dt      = CFL * min(dx,dy) / c
t_final = 4.0        # seconds
nt      = int(t_final/dt)

# -------------------------------------------------------------------
# 2) Build the grid & allocate arrays
# -------------------------------------------------------------------
grid = UniformGrid((Nx,Ny),(dx,dy))
X,Y  = grid.Xc

# η at time n, n-1, and n+1
eta    = np.zeros((Nx,Ny))
eta_old= np.zeros_like(eta)
eta_new= np.zeros_like(eta)

# -------------------------------------------------------------------
# 3) Initial condition: Gaussian hump at center
# -------------------------------------------------------------------
x0, y0 = Lx/2, Ly/2
sigma  = 0.1
eta[:] = np.exp(-((X-x0)**2 + (Y-y0)**2)/(2*sigma**2))
# zero initial velocity ==> eta_old = eta - dt*eta_t → eta_old=eta
eta_old[:] = eta.copy()

# -------------------------------------------------------------------
# 4) Set up the figure for animation
# -------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(6,5))
pcm = ax.pcolormesh(X, Y, eta, cmap='RdBu_r',
                    vmin=-0.5, vmax=0.5, shading='auto')
ax.set_title("2D circular wave (linear PDE)")
ax.set_xlabel("x"); ax.set_ylabel("y")
cb = plt.colorbar(pcm, ax=ax, label="η (m)")

# -------------------------------------------------------------------
# 5) Time‐stepping: explicit 2nd‐order scheme
#    η_new = 2η - η_old + (c dt)^2 ∇²η
#    with η=0 on all boundaries at every time
# -------------------------------------------------------------------
coef = (c*dt)**2

def update(frame):
    global eta, eta_old, eta_new

    # (1) compute 2D Laplacian via our reusable operator
    #     it will internally pull dx, dy from grid.spacing
    lap = laplacian(eta, grid)

    # (2) classic second-order-in-time wave update
    eta_new = 2*eta - eta_old + coef * lap

    # (3) zero-Dirichlet walls
    eta_new[ 0, : ] = 0.0
    eta_new[-1, : ] = 0.0
    eta_new[:,  0 ] = 0.0
    eta_new[:, -1 ] = 0.0

    # (4) rotate time-levels
    eta_old[:], eta[:] = eta, eta_new

    # (5) update the plot
    pcm.set_array(eta.ravel())
    return [pcm]

ani = animation.FuncAnimation(
    fig, update, frames=nt,
    interval=30, blit=True, repeat=False)

plt.tight_layout()
plt.show()
