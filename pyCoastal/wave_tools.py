# wave_tools.py

import numpy as np

def dispersion(T: float, h: float, g: float = 9.81) -> float:
    """
    Calculate the wavelength (L) using the dispersion relationship for water waves.

    Args:
        T (float): Wave period (s)
        h (float): Water depth (m)
        g (float, optional): Accel. due to gravity (m/s²). Defaults to 9.81.

    Returns:
        float: Wave length (m)
    """
    # Deep-water wavelength guess
    L0 = g * T**2 / (2 * np.pi)
    L = L0
    # Fixed-point iteration
    for _ in range(100):
        L_new = L0 * np.tanh(2 * np.pi * h / L)
        if abs(L_new - L) < 1e-3:
            break
        L = L_new
    return L

def wave_number(T: float, h: float) -> float:
    """
    Angular wave number for given wave period and depth.
    k = 2π / L

    Args:
        T (float): Wave period (s)
        h (float): Water depth (m)

    Returns:
        float: Wave number (rad/m)
    """
    L = dispersion(T, h)
    return 2 * np.pi / L

def surf_similarity(alpha: float, H: float, T: float) -> float:
    """
    Compute Iribarren (surf similarity) number:
    ξ = tan(alpha) / sqrt(H / L0), where L0 = g*T^2/(2π)

    Args:
        alpha (float): Slope angle (radians)
        H (float): Wave height (m)
        T (float): Wave period (s)

    Returns:
        float: Iribarren number
    """
    g = 9.81
    L0 = g * T**2 / (2 * np.pi)
    xi = np.tan(alpha) / np.sqrt(H / L0)
    return xi

def ursell_number(H: float, T: float, h: float) -> tuple:
    """
    Calculate the Ursell number (U = H * L^2 / h^3) and interpret wave non-linearity.

    Args:
        H (float): Wave height (m)
        T (float): Wave period (s)
        h (float): Water depth (m)

    Returns:
        (float, str): The Ursell number and its interpretation:
                      U < 32 → "Linear regime..."
                      U ≥ 32 → "Non‑linear regime..."
    """
    L = dispersion(T, h)
    U = H * L**2 / h**3

    if U < 32:
        interp = f"U = {U:.1f}: Linear regime; shallow-long waves; linear theory applicable."
    else:
        interp = f"U = {U:.1f}: Nonlinear regime; use Stokes/Boussinesq or higher-order wave theory."

    return U, interp
