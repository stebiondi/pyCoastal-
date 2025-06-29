import numpy as np

def dispersion(T, h):
    """
    Computes the wavelength (L) from the dispersion relationship,
    given wave period T and water depth h.
    
    Parameters:
        T (float): Wave period [s]
        h (float): Water depth [m]

    Returns:
        float: Wave length [m]
    """
    g = 9.81  # gravity [m/s²]
    L0 = g * float(T)**2 / (2 * np.pi)
    L = L0
    for _ in range(100):
        Lnew = L0 * np.tanh(2 * np.pi * float(h) / L)
        if abs(Lnew - L) < 0.001:
            L = Lnew
            break
        L = Lnew
    return L
