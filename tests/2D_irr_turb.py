# examples/2d_bc_test.py

import numpy as np
import matplotlib.pyplot as plt

from pyCoastal.numerics.grid import UniformGrid
from pyCoastal.boundary import (
    DirichletBC, NeumannBC, WallBC, SpongeBC, BoundaryManager
)

def main():
    # 1) Build a 400×400 m domain with 200×200 cells (dx=dy=2 m)
    grid = UniformGrid(shape=(200, 200), spacing=(2.0, 2.0))

    # 2) Initialize fields: eta, u, v
    eta = np.zeros(grid.shape)
    u   = np.zeros(grid.shape)
    v   = np.zeros(grid.shape)
    fields = {"eta": eta, "u": u, "v": v}

    # 3) Set up our BC manager
    bc_mgr = BoundaryManager()

    # 3a) Southern Dirichlet on η: 0.5 m × sin(2πt/3s)
    ω = 2*np.pi/3.0
    bc_mgr.add(DirichletBC(
        "south", ["eta"], lambda t: 0.5*np.sin(ω*t)
    ))

    # 3b) Northern sponge: 1-cell-wide, linear α from 1→0
    def north_damp(t, idx):
        # idx is a flat index; convert to row i,j
        i = idx // grid.shape[1]
        # one-cell wide => either i==199
        # but we'll demonstrate spatially-varying over 20 cells:
        ramp = 20
        offset = grid.shape[0] - ramp
        α = 1.0 - np.clip((i-offset)/ramp, 0.0, 1.0)
        return α
    bc_mgr.add(SpongeBC("north", ["eta","u","v"], damping=north_damp))

    # 3c) West inflow: u=0.2 m/s, zero‐gradient η
    bc_mgr.add(DirichletBC("west", ["u"], value=0.2))
    bc_mgr.add(NeumannBC("west", ["eta"], gradient=0.0))

    # 3d) East outflow: zero‐gradient u,η,v
    bc_mgr.add(NeumannBC("east", ["u","eta","v"], gradient=0.0))

    # 3e) Side walls on south & north for v
    bc_mgr.add(WallBC("south", ["v"]))
    bc_mgr.add(WallBC("north", ["v"]))

    # 4) Time‐loop: no physics, just BCs + record gauge
    t0, t1, dt = 0.0,  60.0, 0.1
    nt = int((t1-t0)/dt)
    times = np.linspace(t0, t1, nt+1)

    # gauge at center of domain
    ci = grid.shape[0]//2
    cj = grid.shape[1]//2
    flat_gauge = ci*grid.shape[1] + cj
    gauge = []

    for n,t in enumerate(times):
        # apply all BCs in place
        bc_mgr.apply_all(fields, grid, t)

        # no physics update (pure BC test)
        # record eta at gauge
        gauge.append(fields["eta"].flat[flat_gauge])

        if n%100==0:
            print(f"t = {t:.1f}s   southern η(row=0) ≈ {fields['eta'].flat[cj]:.3f} m")

    # 5) plot
    plt.figure(figsize=(8,3))
    plt.plot(times, gauge, 'b-')
    plt.xlabel("time [s]")
    plt.ylabel("η at center [m]")
    plt.title("BC‐test: Southern sine wave + north sponge")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__=="__main__":
    main()
