# structure.py

import math
from .wave_tools import surf_similarity

def bruuns_rule(S: float, beta: float, L: float = None, h: float = None, B: float = None) -> float:
    """
    Estimate shoreline recession R using the Bruun rule.
    R = S*L / (h + B) or S / tan(beta)
    """
    if L is not None and h is not None and B is not None:
        return S * L / (h + B)
    return S / math.tan(beta)

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
