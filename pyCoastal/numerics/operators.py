# pyCoastal/numerics/operators.py
import numpy as np

def laplacian(field: np.ndarray, grid) -> np.ndarray:
    """
    Compute the 5-point Laplacian of `field` on a UniformGrid, using
    centered finite differences and periodic (roll) indexing.
    
    Parameters
    ----------
    field : (nx,ny) array
      Input scalar field.
    grid : UniformGrid
      The grid, provides grid.spacing = (dx, dy).
    
    Returns
    -------
    lap : (nx,ny) array
      The discrete Laplacian ∂²/∂x² + ∂²/∂y².
    """
    dx, dy = grid.spacing

    # shift in x (axis=0) and y (axis=1)
    lap_x = (np.roll(field, -1, axis=0) - 2*field + np.roll(field, 1, axis=0)) / dx**2
    lap_y = (np.roll(field, -1, axis=1) - 2*field + np.roll(field, 1, axis=1)) / dy**2

    return lap_x + lap_y

def gradient(field, grid):
    """Centered gradient ∇f → (fx, fy) on cell centers."""
    dx, dy = grid.spacing
    fx = (np.roll(field, -1, axis=1) - np.roll(field, +1, axis=1)) / (2*dx)
    fy = (np.roll(field, -1, axis=0) - np.roll(field, +1, axis=0)) / (2*dy)
    return fx, fy


def grad_x(field, grid):
    """∂/∂x using centered differences."""
    dx = grid.spacing[0]
    return (np.roll(field, -1, axis=0) - np.roll(field, 1, axis=0)) / (2*dx)

def grad_y(field, grid):
    """∂/∂y using centered differences."""
    dy = grid.spacing[1]
    return (np.roll(field, -1, axis=1) - np.roll(field, 1, axis=1)) / (2*dy)

def upwind_x(field, u, grid):
    """First-order upwind in x: uses flow sign in u to pick difference."""
    dx = grid.spacing[0]
    # roll forwards/backwards
    f_fwd = (field - np.roll(field,  1, axis=0)) / dx
    f_bwd = (np.roll(field, -1, axis=0) - field)  / dx
    return np.where(u > 0, f_fwd, f_bwd)

def upwind_y(field, v, grid):
    """First-order upwind in y."""
    dy = grid.spacing[1]
    f_fwd = (field - np.roll(field,  1, axis=1)) / dy
    f_bwd = (np.roll(field, -1, axis=1) - field)  / dy
    return np.where(v > 0, f_fwd, f_bwd)

def divergence(ux, uy, grid):
    """Centered divergence ∇·u on cell centers."""
    dx, dy = grid.spacing
    dux_dx = (np.roll(ux, -1, axis=1) - np.roll(ux, +1, axis=1)) / (2*dx)
    duy_dy = (np.roll(uy, -1, axis=0) - np.roll(uy, +1, axis=0)) / (2*dy)
    return dux_dx + duy_dy


def curl_z(u, v, grid):
    """∇×(u,v) in 2D gives scalar k-component: ∂v/∂x − ∂u/∂y."""
    return grad_x(v, grid) - grad_y(u, grid)

def biharmonic(field, grid):
    """Δ² field = Laplacian(Laplacian(field))."""
    return laplacian(laplacian(field, grid), grid)

def mixed_xy(field, grid):
    """∂²/∂x∂y of field (central)."""
    dx, dy = grid.spacing
    return (
        np.roll(np.roll(field, -1, axis=0), -1, axis=1)
      - np.roll(np.roll(field, -1, axis=0),  1, axis=1)
      - np.roll(np.roll(field,  1, axis=0), -1, axis=1)
      + np.roll(np.roll(field,  1, axis=0),  1, axis=1)
    ) / (4*dx*dy)

def advect(u, v, field, grid, scheme="upwind"):
    """Compute u·∇field + v·∇field using chosen scheme."""
    if scheme == "upwind":
        fx = upwind_x(field, u, grid)
        fy = upwind_y(field, v, grid)
    else:
        fx = grad_x(field, grid)
        fy = grad_y(field, grid)
    return u*fx + v*fy

def smooth3(field, grid):
    """Simple 3×3 box filter."""
    return 0.25*field + 0.125*(
        np.roll(field,  1, axis=0) +
        np.roll(field, -1, axis=0) +
        np.roll(field,  1, axis=1) +
        np.roll(field, -1, axis=1)
    )
