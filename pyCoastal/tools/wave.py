# wave.py

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

def wave_setup(Hb: float, gamma: float = 0.8) -> float:
    """
    Estimate mean water level increase at shoreline due to wave setup.
    η = (5/16) * γ * Hb
    """
    return 5/16 * gamma * Hb

def _pm_spectrum(f: np.ndarray, Tp: float, g: float = 9.81) -> np.ndarray:
    """
    Pierson–Moskowitz spectral density S(f) for a fully developed sea:
      S(f) = α g^2 / (2π)^4 f^{-5} exp[-5/4 (f_p/f)^4]
    where f_p = 1/Tp, α = 0.0081

    Args:
        f: 1D array of frequencies (Hz)
        Tp: peak period (s)
        g: gravity (m/s²)
    Returns:
        S: spectral density [m²/Hz]
    """
    α = 0.0081
    fp = 1.0 / Tp
    S = (α * g**2 / (2 * np.pi)**4) * f**(-5) * np.exp(-1.25 * (fp / f)**4)
    return S

def _jonswap_spectrum(
    f: np.ndarray,
    Tp: float,
    Hs: float,
    gamma: float = 3.3,
    g: float = 9.81
) -> np.ndarray:
    """
    JONSWAP spectral density S(f):
      S(f) = S_pm(f) * gamma^exp[- (f - fp)^2 / (2 * sigma^2 * fp^2)]
    with sigma=0.07 for f<=fp, else 0.09.
    We also scale by Hs² to get correct variance.
    """
    fp = 1.0 / Tp
    S_pm = _pm_spectrum(f, Tp, g)
    sigma = np.where(f <= fp, 0.07, 0.09)
    r = np.exp(- (f - fp)**2 / (2 * sigma**2 * fp**2))
    S_j = S_pm * gamma**r

    # normalize so that ∫2S df = Hs²/8
    df = f[1] - f[0]
    var = 2 * np.sum(S_j) * df
    return S_j * (Hs**2 / (8 * var))


def generate_irregular_wave(
    Hs: float,
    Tp: float,
    duration: float,
    dt: float,
    spectrum: str = "pm",
    gamma: float = 3.3
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate one realization of an irregular wave time-series η(t).

    Args:
      Hs        : significant wave height [m]
      Tp        : peak period               [s]
      duration  : total record length       [s]
      dt        : time step                 [s]
      spectrum  : 'pm' or 'jonswap'
      gamma     : peak enhancement factor (JONSWAP only)

    Returns:
      t   : time array of length N = ceil(duration/dt)
      eta : η(t) time-series
    """
    N = int(np.ceil(duration / dt))
    t = np.arange(N) * dt
    df = 1.0 / duration
    f = np.arange(1, N//2 + 1) * df

    if spectrum.lower() == "pm":
        S = _pm_spectrum(f, Tp)
        # scale to Hs
        var_pm = 2 * np.sum(S) * df
        S *= (Hs**2 / (8 * var_pm))
    elif spectrum.lower() == "jonswap":
        S = _jonswap_spectrum(f, Tp, Hs, gamma)
    else:
        raise ValueError("`spectrum` must be 'pm' or 'jonswap'")

    phases     = np.random.uniform(0, 2*np.pi, size=f.shape)
    amplitudes = np.sqrt(2 * S * df)

    eta = np.zeros_like(t)
    for A, fi, phi in zip(amplitudes, f, phases):
        eta += A * np.cos(2*np.pi*fi*t + phi)

    return t, eta