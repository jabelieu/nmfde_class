import numpy as np

import numpy as np

def tri_A_Jacobi_MG(b, ig, n_stop):
    """
    Tridiagonal Jacobi iteration for Poisson's equation
    with Dirichlet boundary conditions.
    
    Translation of Christlieb's CMSE 821 (Fall 2025) MATLAB code.

    Parameters
    ----------
    b : ndarray
        Right-hand side vector (scaled by h^2).
    ig : ndarray
        Initial guess vector.
    n_stop : int
        Number of Jacobi iterations to perform.

    Returns
    -------
    z : ndarray
        Updated solution vector after Jacobi iterations.
    """

    m = len(b)
    y = np.zeros_like(b)
    z = np.zeros_like(b)

    # Initial Jacobi sweep
    y[0] = -0.5 * (b[0] - ig[1])
    for i in range(1, m - 1):
        y[i] = -0.5 * (b[i] - (ig[i - 1] + ig[i + 1]))
    y[m - 1] = -0.5 * (b[m - 1] - ig[m - 2])

    cnt = 1
    while cnt < n_stop:
        z[0] = -0.5 * (b[0] - y[1])
        for i in range(1, m - 1):
            z[i] = -0.5 * (b[i] - (y[i - 1] + y[i + 1]))
        z[m - 1] = -0.5 * (b[m - 1] - y[m - 2])

        y[:] = z
        cnt += 1

    return z

import numpy as np

def tri_A_Jacobi(b, ig, n_stop, tol, true_sol):
    """
    Tridiagonal Jacobi iteration solver for Poisson's equation
    with Dirichlet boundary conditions.

    Translation from MATLAB code by Christlieb (CMSE 821, Fall 2025)

    Parameters
    ----------
    b : ndarray
        Right-hand side vector.
    ig : ndarray
        Initial guess vector.
    n_stop : int
        Maximum number of iterations.
    tol : float
        Convergence tolerance.
    true_sol : ndarray
        True solution (used for error check, as in the MATLAB code).

    Returns
    -------
    z : ndarray
        Approximate solution after Jacobi iterations.
    """

    m = len(b)
    y = np.zeros_like(b)
    z = np.zeros_like(b)

    # Initial iteration
    y[0] = -0.5 * (b[0] - ig[1])
    for i in range(1, m - 1):
        y[i] = -0.5 * (b[i] - (ig[i - 1] + ig[i + 1]))
    y[m - 1] = -0.5 * (b[m - 1] - ig[m - 2])

    e = 10.0
    cnt = 1

    while e > tol and cnt < n_stop:
        z[0] = -0.5 * (b[0] - y[1])
        for i in range(1, m - 1):
            z[i] = -0.5 * (b[i] - (y[i - 1] + y[i + 1]))
        z[m - 1] = -0.5 * (b[m - 1] - y[m - 2])

        # MATLAB used e = max(abs(true - z))
        e = np.max(np.abs(true_sol - z))

        y[:] = z
        cnt += 1

    return z


def MG_jacobi(N, ig, u0, uNGP, r, h, x, Num_Stop, A, B, lev):
    """
    Jocabit Multi Grid for Poisson Equation
    Example based on LeVeque (Recursive Function)

    Translated from MATLAB by Iota for Josh Blue
    CMSE 821 - Fall 2025
    """

    # -----------------------------------------------
    # RHS of Ax = b
    # -----------------------------------------------
    b = np.zeros(N)
    b[0] = h**2 * r[0] - u0
    b[1:N-1] = h**2 * r[1:N-1]
    b[-1] = h**2 * r[-1] - uNGP

    # -----------------------------------------------
    # Initial iteration
    # -----------------------------------------------
    u = tri_A_Jacobi_MG(b, ig, Num_Stop)

    # -----------------------------------------------
    # Recursive correction if more levels exist
    # -----------------------------------------------
    if (lev - 1) > 0:
        # -------------------------------------------
        # Compute residual
        # -------------------------------------------
        res = np.zeros_like(r)
        res[0] = r[0] - (1 / h**2) * (u0 - 2*u[0] + u[1])
        res[1:N-1] = r[1:N-1] - (1 / h**2) * (u[:-2] - 2*u[1:-1] + u[2:])
        res[-1] = r[-1] - (1 / h**2) * (uNGP - 2*u[-1] + u[-2])

        # -------------------------------------------
        # Coarse mesh setup (xc(i) = x(2i))
        # -------------------------------------------
        hc = 2 * h
        xc = np.arange(A, B + hc/2, hc)
        Nc = len(xc)
        xii = xc[1:-1]
        Nii = Nc - 2

        # -------------------------------------------
        # Initial guess on coarse grid
        # -------------------------------------------
        ig_c = np.zeros(Nc)

        # -------------------------------------------
        # Restrict residual to coarse mesh
        # -------------------------------------------
        rr = res[::2]

        # -------------------------------------------
        # Recursive call to compute coarse-grid error
        # -------------------------------------------
        l = lev - 1
        e = MG_jacobi(Nii, ig_c, 0, 0, rr, hc, xii, Num_Stop, A, B, l)

        # -------------------------------------------
        # Interpolate error back to fine mesh (linear)
        # -------------------------------------------
        eu = np.zeros(N)
        j = 0
        for i in range(1, N-1, 2):
            eu[i] = e[j]
            j += 1

        # Linear extrapolation at boundaries
        eu[0] = ((eu[1] - 0) / (x[1] - A)) * (x[0] - A)
        for i in range(2, N-2, 2):
            eu[i] = ((eu[i+1] - eu[i-1]) / (x[i+1] - x[i-1])) * (x[i] - x[i-1]) + eu[i-1]
        eu[-1] = -((0 - eu[-2]) / (B - x[-2])) * (x[-1] - B)

        # -------------------------------------------
        # Correct u on fine mesh
        # -------------------------------------------
        u += eu

    # -----------------------------------------------
    # Final relaxation on fine mesh
    # -----------------------------------------------
    u = tri_A_Jacobi_MG(b, u, Num_Stop)

    return u
