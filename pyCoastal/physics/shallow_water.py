# pycoastal/physics/shallow_water.py

import numpy as np

class ShallowWater2D:
    """
    2D nonlinear shallow water equations:
      ∂h/∂t + ∂(h u)/∂x + ∂(h v)/∂y = 0
      ∂(h u)/∂t + ∂(h u^2 + ½ g h^2)/∂x + ∂(h u v)/∂y = Sx
      ∂(h v)/∂t + ∂(h u v)/∂x + ∂(h v^2 + ½ g h^2)/∂y = Sy
    """
    def __init__(self, g: float = 9.81):
        self.g = g

    def fluxes(self, h: np.ndarray, hu: np.ndarray, hv: np.ndarray):
        """
        Compute the three flux‐pairs (in x and y) for the SWEs.
        Returns (Fh, Gh), (Fhu, Ghu), (Fhv, Ghv)
        """
        u = hu / h
        v = hv / h

        Fh  = hu
        Gh  = hv

        Fhu = hu * u + 0.5 * self.g * h**2
        Ghu = hu * v

        Fhv = hv * u
        Ghv = hv * v + 0.5 * self.g * h**2

        return (Fh, Gh), (Fhu, Ghu), (Fhv, Ghv)

    def source_bed_slope(self, h: np.ndarray, zb: np.ndarray):
        """
        Return bed‐slope source terms Sx, Sy for momentum:
          Sx = - g h ∂zb/∂x,  Sy = - g h ∂zb/∂y
        """
        dzdx = np.gradient(zb, axis=1)
        dzdy = np.gradient(zb, axis=0)
        Sx = - self.g * h * dzdx
        Sy = - self.g * h * dzdy
        return Sx, Sy

    # you can add friction, coriolis, rainfall, etc.

