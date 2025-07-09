# sediment_transport.py

import math
import numpy as np

def shields_parameter(tau_b: float, rho_s: float, rho: float, d: float, g: float = 9.81) -> float:
    """
    Dimensionless Shields parameter for initiation of sediment motion.
    tau* = tau_b / ((rho_s - rho) * g * d)
    """
    return tau_b / ((rho_s - rho) * g * d)

def van_rijn_bedload(Ue: float, h: float, d50: float, rho_s: float, rho: float, nu: float = 1e-6) -> float:
    """
    Van Rijn (1984) bed-load transport per unit width:
    qb = 0.015 * rho_s * Ue * h * (d50 / h)^1.2 * Me^1.5
    with Me = (Ue - Ucr) / sqrt((s-1)*g*d50)
    """
    s = rho_s / rho
    Ucr = 0.19 * (d50**0.1) * math.log(12*h / (3*d50))
    Me = max((Ue - Ucr) / math.sqrt((s -1)*9.81 * d50), 0)
    qb = 0.015 * rho_s * Ue * h * (d50 / h)**1.2 * Me**1.5
    return qb

def van_rijn_suspended(Ue: float, h: float, d50: float, rho_s: float, rho: float, nu: float = 1e-6) -> float:
    """
    Van Rijn suspended-load formula:
    qs = 0.008 * rho_s * Ue * d50 * Me^2.4 * D*^-0.6
    where D* = (d50*((s-1)*g/nu^2))^(1/3)
    """
    s = rho_s / rho
    Me = max((Ue - (0.19 * (d50**0.1) * math.log(12*h / (3*d50)))) / math.sqrt((s -1)*9.81 * d50), 0)
    Dstar = ((d50*((s -1)*9.81)/nu**2))**(1/3)
    qs = 0.008 * rho_s * Ue * d50 * Me**2.4 * Dstar**(-0.6)
    return qs

def bijker_bedload(tau_wave: float, tau_current: float, rho_s: float, rho: float, d50: float, g: float = 9.81) -> float:
    """
    Bijker (1971) formula combining wave and current effects for bedload:
    qb_sb ∝ sqrt(tau_total)
    """
    tau_total = tau_wave + tau_current
    return math.sqrt(tau_total / ((rho_s - rho) * g * d50))

def cerc_transport(Eb: float, angle: float, K: float = 0.39, rho: float = 1025, g: float = 9.81) -> float:
    """
    CERC longshore transport rate:
    Q = K * Eb/(ρg) * sinφb cosφb
    """
    return K * Eb / (rho * g) * np.sin(angle) * np.cos(angle)

def bagnold_sediment(H: float, c: float, rho_s: float, rho: float = 1025) -> float:
    """
    Bagnold’s load estimate: qs ∝ (ρs/ρ) * wave power.
    Simplified form here for energy-driven sediment transport.
    """
    return (rho_s / rho) * H**2 * c

def izbash_current(rho_s: float, rho: float, d: float, g: float = 9.81) -> float:
    """
    Izbash critical current for stone stability:
      u_c = 1.7 * sqrt[Δ g d]
    """
    Delta = (rho_s - rho) / rho
    return 1.7 * math.sqrt(Delta * g * d)

def einstein_bedload(tau_star: float, d: float, s: float, g: float = 9.81) -> float:
    """
    Einstein’s bedload formula (probabilistic):
      q_b = 8 √[g (s − 1) d^3] * τ*^1.5
    Useful for fine sand transport upstream.
    """
    return 8 * math.sqrt(g * (s - 1) * d**3) * tau_star**1.5

