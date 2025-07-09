#!/usr/bin/env python3
"""
wave2D_current.py

2D passive current advection in x‐direction with specified inlet/outlet BCs:
  - UniformGrid for the mesh
  - BoundaryManager with Dirichlet (west inlet) and Neumann (east outlet)
  - Upwind finite‐difference in x
  - Real‐time animation of the 2D u‐field + time‐series at a center gauge
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from pyCoastal.numerics.grid import UniformGrid
from pyCoastal.numerics.boundary import DirichletBC, BoundaryManager

# -------------------------------------------------------------------
# 1) Physical & numerical setup
# -------------------------------------------------------------------
# Current speed (m/s)
c0    = 1.0
# CFL for advection
CFL   = 0.5

# Grid dimensions
Nx, Ny = 200, 200  # cells in x,y
dx, dy = 1.0, 1.0  # spacing (m)
grid   = UniformGrid((Nx, Ny), (dx, dy))
X, Y   = grid.Xc  # cell centers for plotting

# Time‐stepping
dt    = CFL * dx / c0
t_final = 500.0
nt    = int(t_final / dt)

# -------------------------------------------------------------------
# 2) Initialize fields and boundary manager
# -------------------------------------------------------------------
# Velocity field u (only x‐component)
u     = np.zeros((Nx, Ny))
# placeholder for update
u_new = np.zeros_like(u)
fields = {"u": u}

# Setup BCs
bc_mgr = BoundaryManager()
# West inlet: fixed u = c0 (Dirichlet)
bc_mgr.add(DirichletBC("east", ["u"], value=c0))
# East outlet: zero‐gradient (Neumann)
bc_mgr.add(DirichletBC("west", ["u"], value=c0))

# -------------------------------------------------------------------
# 3) Observation (gauge at center)
# -------------------------------------------------------------------
ix, iy = Nx//2, Ny//2
ts_obs = []
times  = []

# -------------------------------------------------------------------
# 4) Plot setup
# -------------------------------------------------------------------
fig = plt.figure(figsize=(10,4))
ax_field = fig.add_subplot(1,2,1)
pcm = ax_field.pcolormesh(X, Y, u, cmap='viridis', vmin=0, vmax=c0)
ax_field.set_title('u field'); ax_field.set_xlabel('x'); ax_field.set_ylabel('y')
cb = fig.colorbar(pcm, ax=ax_field, label='u (m/s)')

ax_ts = fig.add_subplot(1,2,2)
ax_ts.set_xlim(0, t_final)
ax_ts.set_ylim(0, 1.1*c0)
line_ts, = ax_ts.plot([], [], 'k-')
ax_ts.set_title(f'u at ({ix},{iy})'); ax_ts.set_xlabel('t (s)'); ax_ts.set_ylabel('u (m/s)')

# -------------------------------------------------------------------
# 5) Time‐loop update function
# -------------------------------------------------------------------
def update(n):
    global u, u_new
    t = n * dt
    times.append(t)

    # apply BCs before update
    bc_mgr.apply_all(fields, grid, t)

    # upwind advection in x: u_t + c0 * u_x = 0
    # j index for y loop but no y‐advection here
    # for each j, use 1D upwind in x
    # interior: i=1..Nx-1
    u_new[1:, :] = u[1:, :] - (c0*dt/dx)*(u[1:, :] - u[:-1, :])
    # west boundary u_new[0] set by BC, east u_new[-1] via BC
    u_new[0, :] = u[0, :]
    u_new[-1,:] = u_new[-1,:]

    # swap
    u, u_new = u_new, u
    fields['u'] = u

    # record gauge
    ts_obs.append(u[ix, iy])
    line_ts.set_data(times, ts_obs)

    # update plot
    pcm.set_array(u.ravel())
    return [pcm, line_ts]

# -------------------------------------------------------------------
# 6) Launch animation
# -------------------------------------------------------------------
ani = animation.FuncAnimation(
    fig, update, frames=nt,
    interval=50, blit=False, repeat=False   # ← turn off blitting
)

# workaround matplotlib oddity: ensure ._resize_id always exists
if not hasattr(ani, "_resize_id"):
    ani._resize_id = None

plt.tight_layout()
plt.show()
