"""
Domain and mesh definitions for pyCoastal.
Provides 1D and 2D structured grid classes.
"""

import numpy as np

class Mesh1D:
    """
    Simple 1D mesh.
    Attributes
    ----------
    x : np.ndarray
        Cell center coordinates.
    dx : float
        Uniform grid spacing.
    """
    def __init__(self, x0: float, x1: float, nx: int):
        self.x0 = x0
        self.x1 = x1
        self.nx = nx
        self.dx = (x1 - x0) / nx
        # cell centers
        self.x = x0 + self.dx * (0.5 + np.arange(nx))


class Mesh2D:
    """
    Rectangular 2D mesh.
    Attributes
    ----------
    x, y : np.ndarray
        2D arrays of cell‐center coordinates.
    dx, dy : float
        Uniform spacings in x and y.
    """
    def __init__(self, x0: float, x1: float, nx: int,
                       y0: float, y1: float, ny: int):
        self.x0, self.x1, self.nx = x0, x1, nx
        self.y0, self.y1, self.ny = y0, y1, ny
        self.dx = (x1 - x0) / nx
        self.dy = (y1 - y0) / ny
        xc = x0 + self.dx * (0.5 + np.arange(nx))
        yc = y0 + self.dy * (0.5 + np.arange(ny))
        # meshgrid with (ny, nx) shape
        self.x, self.y = np.meshgrid(xc, yc)


class Domain:
    """
    High‐level domain object. Reads geometry from config
    and instantiates the appropriate mesh.
    """
    def __init__(self, cfg: dict):
        geom = cfg.get("domain", {})
        dim = geom.get("dimension", 1)
        if dim == 1:
            self.mesh = Mesh1D(
                x0 = float(geom["x0"]),
                x1 = float(geom["x1"]),
                nx = int(  geom["nx"])
            )
        elif dim == 2:
            self.mesh = Mesh2D(
                x0 = float(geom["x0"]),
                x1 = float(geom["x1"]),
                nx = int(  geom["nx"]),
                y0 = float(geom["y0"]),
                y1 = float(geom["y1"]),
                ny = int(  geom["ny"])
            )
        else:
            raise ValueError(f"Unsupported dimension: {dim}")

    def info(self):
        """Print a summary of the domain."""
        if hasattr(self.mesh, "x"):
            print(f"1D Domain: x∈[{self.mesh.x0}, {self.mesh.x1}] with {self.mesh.nx} cells")
        else:
            print(f"2D Domain: x∈[{self.mesh.x0}, {self.mesh.x1}] @ {self.mesh.nx} cells; "
                  f"y∈[{self.mesh.y0}, {self.mesh.y1}] @ {self.mesh.ny} cells")
