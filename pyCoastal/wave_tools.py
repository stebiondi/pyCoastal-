# wave_tools.py

import numpy as np

def dispersion(T: float, h: float, g: float = 9.81) -> float:
    """
    Calculate the wavelength (L) via the dispersion relationship for water waves.

    Args:
        T (float): Wave period (s)
        h (float): Water depth (m)
        g (float, optional): Gravity acceleration (m/s²). Default: 9.81.

    Returns:
        float: Wavelength L (m)
    """
    L0 = g * T**2 / (2 * np.pi)  # deep-water wavelength guess
    L = L0
    for _ in range(100):
        L_new = L0 * np.tanh(2 * np.pi * h / L)
        if abs(L_new - L) < 1e-3:
            break
        L = L_new
    return L

def wave_number(T: float, h: float) -> float:
    """
    Compute wave number k = 2π / L.

    Args:
        T (float): Period (s)
        h (float): Depth (m)

    Returns:
        float: Wave number k (rad/m)
    """
    L = dispersion(T, h)
    return 2 * np.pi / L

def surf_similarity(alpha: float, H: float, T: float) -> float:
    """
    Compute the Iribarren (surf similarity) number ξ.

    ξ = tan(alpha) / sqrt(H / L0), where L0 = g*T²/(2π)

    Args:
        alpha (float): Slope angle (radians)
        H (float): Wave height (m)
        T (float): Wave period (s)

    Returns:
        float: Iribarren number ξ
    """
    g = 9.81
    L0 = g * T**2 / (2 * np.pi)
    return np.tan(alpha) / np.sqrt(H / L0)

def breaker_type(alpha: float, H: float, T: float) -> str:
    """
    Determine wave breaking type based on Iribarren number.

    Classifications:
    - ξ < 0.4     : Spilling breaker
    - 0.4 ≤ ξ ≤ 2 : Plunging breaker
    - ξ > 2       : Collapsing/surging breaker
    
    Args:
        alpha (float): Slope angle (rad)
        H (float): Wave height (m)
        T (float): Wave period (s)

    Returns:
        str: 'Spilling', 'Plunging', or 'Surging' (collapsing)
    """
    xi = surf_similarity(alpha, H, T)
    if xi < 0.4:
        return f"Spilling (ξ = {xi:.2f})"
    elif xi <= 2:
        return f"Plunging (ξ = {xi:.2f})"
    else:
        return f"Collapsing/Surging (ξ = {xi:.2f})"

def ursell_number(H: float, T: float, h: float) -> tuple:
    """
    Calculate Ursell number U = H*L²/h³ and provide interpretation.

    Args:
        H (float): Wave height (m)
        T (float): Period (s)
        h (float): Depth (m)

    Returns:
        (float, str): Ursell number U and interpretation.
    """
    L = dispersion(T, h)
    U = H * L**2 / h**3
    if U < 32:
        interp = f"U = {U:.1f}: Linear regime; linear theories applicable."
    else:
        interp = f"U = {U:.1f}: Nonlinear regime; use Stokes/Boussinesq or higher."
    return U, interp
