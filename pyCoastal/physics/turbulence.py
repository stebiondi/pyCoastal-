# pycoastal/physics/turbulence.py

import numpy as np

class SmagorinskyModel:
    """
    Large‐eddy eddy‐viscosity: ν_t = (C_s Δ)^2 |S|
    """
    def __init__(self, Cs: float = 0.17, filter_width: float = 1.0):
        self.Cs = Cs
        self.Δ  = filter_width

    def eddy_viscosity(self, u: np.ndarray, v: np.ndarray):
        # Strain‐rate magnitude |S| = sqrt(2 S_ij S_ij)
        du_dx = np.gradient(u, axis=1)
        dv_dy = np.gradient(v, axis=0)
        du_dy = np.gradient(u, axis=0)
        dv_dx = np.gradient(v, axis=1)

        Sxx = du_dx
        Syy = dv_dy
        Sxy = 0.5*(du_dy + dv_dx)

        S_mag = np.sqrt(2*(Sxx**2 + Syy**2 + 2*Sxy**2))
        return (self.Cs * self.Δ)**2 * S_mag


class KEpsilonModel:
    """
    k–ε two‐equation closure in eddy‐viscosity form:
      ν_t = C_μ k^2/ε
      transport eqns for k and ε must be added by solver
    """
    def __init__(self,
                 Cmu: float      = 0.09,
                 sigma_k: float  = 1.0,
                 sigma_e: float  = 1.3,
                 C1: float       = 1.44,
                 C2: float       = 1.92):
        self.Cmu    = Cmu
        self.sigma_k= sigma_k
        self.sigma_e= sigma_e
        self.C1     = C1
        self.C2     = C2

    def eddy_viscosity(self, k: np.ndarray, eps: np.ndarray):
        return self.Cmu * k**2 / np.maximum(eps, 1e-12)

    # production Pk = ν_t (∂u_i/∂x_j + ∂u_j/∂x_i) ∂u_i/∂x_j /2
    # diffusion and source terms belong in your solver coupling.


class KOmegaModel:
    """
    k–ω (Wilcox) two‐equation closure:
      ν_t = k/ω
    """
    def __init__(self,
                 sigma_k: float = 2.0,
                 sigma_ω: float = 2.0,
                 beta_star: float= 0.09,
                 beta: float     = 0.075,
                 gamma: float    = 0.52):
        self.sigma_k   = sigma_k
        self.sigma_w   = sigma_ω
        self.beta_star = beta_star
        self.beta      = beta
        self.gamma     = gamma

    def eddy_viscosity(self, k: np.ndarray, ω: np.ndarray):
        return k / np.maximum(ω, 1e-12)

    # Again, your time‐integrator + solver will need to form the full k & ω eqns.

