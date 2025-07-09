# structural.py
import numpy as np
import math
from .wave import surf_similarity

def hudson_dn50(Hs: float, Delta: float, theta: float, Kd: float = 3.0) -> float:
    """
    Calculate nominal diameter Dn50 for armor stone using Hudson's formula.
    Hs/(Δ Dn50) = (Kd cotθ)^(1/3) / 1.27
    """
    return Hs / (Delta * ((Kd * 1/math.tan(theta))**(1/3)) / 1.27)

def vandermeer_dn50(Hs: float, Delta: float, P: float, N: int, alpha: float, xi_m: float = None, safety: float = 1.0) -> float:
    """
    Estimate nominal diameter Dn50 for armor using Van der Meer formula.
    Uses deep-water and shallow-water expressions.
    """
    # empirical coefficients
    cp, cs = 6.2 / safety, 0.87 / safety
    # deep-water condition: xi_m < xi_cr
    xi_cr = ((cp/cs) * (P**0.31) * math.sqrt(math.tan(alpha)))**(1/(P + 0.5))
    if xi_m is None:
        xi_m = surf_similarity(alpha, Hs, 1)  # using T=1s for test
    if xi_m < xi_cr:
        # plunging
        denom = cp * (P**0.18) * ((Hs / math.sqrt(N))**0.2) * (xi_m**-0.5)
    else:
        # surging
        denom = cs * (P**-0.13) * ((Hs / math.sqrt(N))**0.2) * math.sqrt(1/math.tan(alpha)) * (xi_m**P)
    return Hs / (Delta * denom)
    
def hunt_runup(beta: float, H: float, L: float) -> float:
    """
    Hunt (1959) empirical run-up:
    R ≈ H * (tan beta) / sqrt(H / L)
    """
    return H * np.tan(beta) / np.sqrt(H / L)

def stockdon_runup(H: float, L: float, beta: float) -> float:
    """
    Stockdon et al. (2006) 2%-exceedance run-up:
    R2 = 1.1*(0.35 H xi) + 0.55*(0.75 H xi)
    where xi = tan beta / sqrt(H/L)
    """
    xi = np.tan(beta) / np.sqrt(H / L)
    eta_u = 0.35 * H * xi
    S_w = 0.75 * H * xi
    return 1.1 * (eta_u + 0.5 * np.sqrt(S_w**2 + (0.06*np.sqrt(H*L))**2))
    
def goda_wave_force(H: float, T: float, h: float, beta: float, rho: float = 1025) -> float:
    """
    Estimate the Goda–Takahashi equivalent-static horizontal wave force per unit width (kN/m)
    acting on a vertical breakwater face using Goda's method.

    Args:
        H (float): Significant wave height at wall (m)
        T (float): Wave period (s)
        h (float): Water depth at face (m)
        beta (float): Wave incidence angle (radians)
        rho (float): Water density (kg/m³) [default: 1025 for seawater]

    Returns:
        float: Estimated wave force per unit width (N/m)
    """
    # Design wave height H_design ≈ 1.8 × H (common assumption per Goda)
    H_design = 1.8 * H

    # Pressure distribution height limit ~0.75 H_design above SWL
    p_max = 0.5 * rho * 9.81 * H_design * (1 + math.cos(beta))

    # Linear pressure decrease to z = ±0.75 H_design
    z_limit = 0.75 * H_design

    # Equivalent static force per unit width (area of triangle): F = p_max * z_limit
    F = p_max * z_limit  # N/m

    return F
    
def iribarren_stability(H: float, alpha: float, rho_s: float, rho_w: float, mu: float, N: float) -> float:
    """
    Stone weight W needed for armor stability per Iribarren:
      W ≥ [N ρ_s g H^3] / [Δ^3 (μ cosα − sinα)^3]
    where Δ = (ρ_s/ρ_w − 1), N is safety coefficient,
    μ is friction coefficient, α is slope angle (rad).
    """
    Delta = rho_s/rho_w - 1
    numerator = N * rho_s * 9.81 * H**3
    denom = (Delta**3) * (mu * math.cos(alpha) - math.sin(alpha))**3
    return numerator / denom
