# pycoastal/numerics/poisson.py

import numpy as np

try:
    import scipy.sparse as sp
    import scipy.sparse.linalg as spla
    SCIPY_OK = True
except ImportError:
    SCIPY_OK = False


def apply_dirichlet_bc(phi, bc_mask, bc_values):
    """
    Enforce Dirichlet BC on phi array in place.
    
    bc_mask   : boolean array of same shape as phi; True where BC applies
    bc_values : array same shape as phi; the prescribed phi values
    """
    phi[bc_mask] = bc_values[bc_mask]
    return phi


def build_laplacian(nx, ny, dx, dy):
    """
    Build 2D Laplacian operator with Dirichlet zero BC on a regular grid.
    Returns a scipy CSR sparse matrix of shape (nx*ny, nx*ny).
    """
    if not SCIPY_OK:
        raise RuntimeError("SciPy not available; cannot build sparse Laplacian.")
    N = nx * ny
    # 1D operators
    dx2 = dx * dx
    dy2 = dy * dy
    main_x = -2.0 / dx2 * np.ones(nx)
    off_x  = 1.0  / dx2 * np.ones(nx-1)
    Tx = sp.diags([off_x, main_x, off_x], offsets=[-1, 0, 1], shape=(nx, nx))

    main_y = -2.0 / dy2 * np.ones(ny)
    off_y  = 1.0  / dy2 * np.ones(ny-1)
    Ty = sp.diags([off_y, main_y, off_y], offsets=[-1, 0, 1], shape=(ny, ny))

    # 2D Laplacian = Ix ⊗ Ty  +  Tx ⊗ Iy
    Ix = sp.eye(nx)
    Iy = sp.eye(ny)
    L = sp.kron(Ix, Ty) + sp.kron(Tx, Iy)
    return L.tocsr()


def solve_direct(rhs, dx, dy, bc_mask=None, bc_values=None):
    """
    Solve ∇²φ = rhs with Dirichlet BC via a sparse direct solver (requires SciPy).
    - rhs : 2D array of shape (ny, nx)
    - dx, dy : grid spacing
    - bc_mask, bc_values : optional masks/values for Dirichlet φ
    Returns φ as 2D array.
    """
    ny, nx = rhs.shape
    # Flatten
    b = rhs.ravel()

    # Build matrix
    L = build_laplacian(nx, ny, dx, dy)

    # If Dirichlet BC: modify system (zero rows, ones on diag)
    if bc_mask is not None and bc_values is not None:
        mask_flat = bc_mask.ravel()
        vals_flat = bc_values.ravel()
        for i in np.where(mask_flat)[0]:
            L.data[L.indptr[i]:L.indptr[i+1]] = 0.0   # zero row
            L[i, i] = 1.0                             # diag=1
            b[i]    = vals_flat[i]

    # Solve
    φ_flat = spla.spsolve(L, b)
    φ = φ_flat.reshape((ny, nx))
    return φ


def solve_jacobi(rhs, dx, dy, bc_mask=None, bc_values=None,
                 tol=1e-6, maxiter=5000):
    """
    Simple Jacobi iteration for ∇²φ = rhs with optional Dirichlet BC.
    Returns φ after convergence or maxiter.
    """
    ny, nx = rhs.shape
    φ = np.zeros_like(rhs)
    φ_new = np.zeros_like(rhs)

    dx2 = dx*dx
    dy2 = dy*dy
    denom = 2.0*(1.0/dx2 + 1.0/dy2)

    for it in range(maxiter):
        # interior updates
        φ_new[1:-1,1:-1] = (
            (φ[1:-1,2:] + φ[1:-1,:-2]) / dx2 +
            (φ[2:,1:-1] + φ[:-2,1:-1]) / dy2 -
            rhs[1:-1,1:-1]
        ) / denom

        # enforce Dirichlet BC if provided
        if bc_mask is not None and bc_values is not None:
            φ_new = apply_dirichlet_bc(φ_new, bc_mask, bc_values)

        # compute residual
        diff = np.linalg.norm(φ_new - φ, ord=np.inf)
        φ[:] = φ_new
        if diff < tol:
            break

    return φ


def solve_poisson(rhs, dx, dy, bc_mask=None, bc_values=None,
                  method='auto', **kwargs):
    """
    Poisson solver interface.
      rhs        : 2D array of RHS values
      dx, dy     : grid spacing
      bc_mask    : bool array where Dirichlet BC applies
      bc_values  : array of same shape with prescribed φ values
      method     : 'direct', 'jacobi', or 'auto'
      **kwargs   : passed to the chosen solver (tol, maxiter, etc.)
    """
    if method == 'direct' or (method=='auto' and SCIPY_OK):
        return solve_direct(rhs, dx, dy, bc_mask, bc_values)
    else:
        return solve_jacobi(rhs, dx, dy, bc_mask, bc_values, **kwargs)
