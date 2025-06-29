# morphodynamics.py

import math

def bruuns_rule(S: float, beta: float, L: float = None, h: float = None, B: float = None) -> float:
    """
    Estimate shoreline retreat (R) using Bruun's rule.
    If L, h, and B are provided: R = S*L / (h + B)
    Else: R = S / tan(beta)
    """
    if L is not None and h is not None and B is not None:
        return S * L / (h + B)
    return S / math.tan(beta)

def exner_change(qs_dx: float, porosity: float = 0.64) -> float:
    """
    Compute bed elevation change rate (∂η/∂t) from sediment divergence based on the Exner equation.
    ∂η/∂t = -1/(1–n) * qs_dx
    """
    return -qs_dx / (1 - porosity)
