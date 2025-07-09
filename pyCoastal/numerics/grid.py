# pycoastal/numerics/grid.py

"""
Basic Cartesian grid for 1D/2D/3D domains, now with boundary‐index support
for Dirichlet/Neumann/Sponge BCs.
"""

import numpy as np

# pycoastal/numerics/grid.py

import numpy as np

class UniformGrid:
    def __init__(self, shape, spacing, origin=None):
        self.shape   = tuple(shape)
        self.spacing = tuple(spacing)
        self.ndim    = len(self.shape)
        if origin is None:
            origin = (0.0,)*self.ndim
        self.origin  = tuple(origin)

        # build cell‐center and face coordinate arrays
        centers = []
        faces   = []
        for n, d, o in zip(self.shape, self.spacing, self.origin):
            centers.append(np.linspace(o + 0.5*d, o + (n-0.5)*d, n))
            faces.append(  np.linspace(o,       o + n*d,       n+1))
        self.Xc = np.meshgrid(*centers, indexing="ij")
        self.Xf = np.meshgrid(*faces,   indexing="ij")

        # Precompute flat indices of each face‐side
        self._compute_boundary_indices()

    def _compute_boundary_indices(self):
        n_cells = int(np.prod(self.shape))
        self.boundary_indices = {}
        self.sponge_indices   = {}

        if self.ndim == 2:
            nx, ny = self.shape

            # flat index formula: idx = i*ny + j
            # WEST  = i=0, j=0..ny-1 --> idx = 0*ny + j
            west  = np.arange(0,              ny,      1)

            # EAST  = i=nx-1, j=0..ny-1 --> idx = (nx-1)*ny + j
            east  = np.arange((nx-1)*ny,     nx*ny,   1)

            # SOUTH = j=0, i=0..nx-1 --> idx = i*ny + 0
            south = np.arange(0,              n_cells, ny)

            # NORTH = j=ny-1, i=0..nx-1 --> idx = i*ny + (ny-1)
            north = np.arange(ny-1,           n_cells, ny)

            self.boundary_indices = {
                "west":  west,
                "east":  east,
                "south": south,
                "north": north
            }

        elif self.ndim == 1:
            n = self.shape[0]
            self.boundary_indices = {
                "west":  np.array([0]),
                "east":  np.array([n-1])
            }
        else:
            raise NotImplementedError("UniformGrid._compute_boundary_indices supports 1D/2D only")

        # by default sponge = same as boundary
        for side, idx in self.boundary_indices.items():
            self.sponge_indices[side] = idx.copy()

    def neumann_indices(self, side: str):
        bd = self.boundary_indices[side]
        if self.ndim == 2:
            nx, ny = self.shape
            if side == "west":
                interior = bd + ny   # one cell inward in +x direction
            elif side == "east":
                interior = bd - ny
            elif side == "south":
                interior = bd + 1    # one cell inward in +y direction
            elif side == "north":
                interior = bd - 1
            else:
                raise KeyError(f"Unknown boundary side '{side}'")
        elif self.ndim == 1:
            if side == "west":
                interior = bd + 1
            elif side == "east":
                interior = bd - 1
            else:
                raise KeyError(f"Unknown boundary side '{side}'")
        else:
            raise NotImplementedError("neumann_indices only implemented for 1D/2D")

        return bd, interior

    @property
    def n_cells(self):
        return int(np.prod(self.shape))

    @property
    def cell_volume(self):
        return float(np.prod(self.spacing))

    def cell_indices(self):
        return np.ndindex(self.shape)

