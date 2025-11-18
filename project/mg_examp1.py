import numpy as np
import matplotlib.pyplot as plt

# -----------------------------------------------
# Christlieb — CMSE 821 (Fall 2025)
# Jacobi iteration test for Poisson equation
# Fixed iteration count, varying mesh resolution
# -----------------------------------------------

def jacobiMP(NGP):
    """
    Jacobi smoother experiment for Poisson's equation.
    Replicates Christlieb's CMSE 821 LeVeque example.

    Parameters
    ----------
    NGP : int
        Number of grid points.

    Returns
    -------
    u : ndarray
        Approximate Jacobi solution.
    xii : ndarray
        Interior grid points.
    true : ndarray
        Analytical solution.
    """

    A, B = 0.0, 1.0
    h = (B - A) / (NGP - 1)
    xi = np.linspace(A, B, NGP)

    # Dirichlet BCs
    u0 = 1.0
    uNGP = 3.0

    # RHS of d²u/dx² = f(x)
    pi = 2 * np.arcsin(1)
    xii = xi[1:-1]
    y = 60 * pi * xii**2
    phi = 20 * pi * xii**3
    RHS = h**2 * (-20 + 0.5 * (120 * pi * xii * np.cos(phi))
                  - 0.5 * (y**2) * np.sin(phi))

    N = len(xii)
    b = np.zeros(N)
    b[0] = RHS[0] - u0
    b[1:-1] = RHS[1:-1]
    b[-1] = RHS[-1] - uNGP

    # Analytical "true" solution
    true = 1 + 12 * xii - 10 * xii**2 + 0.5 * np.sin(phi)

    # Initial guess: straight line
    ig = ((uNGP - u0) / (B - A)) * (xii - A) + u0

    # Jacobi iteration (fixed number of iterations)
    n_stop = 1000
    y = np.copy(ig)
    m = len(b)
    z = np.zeros_like(b)

    for _ in range(n_stop):
        z[0] = -0.5 * (b[0] - y[1])
        for i in range(1, m - 1):
            z[i] = -0.5 * (b[i] - (y[i - 1] + y[i + 1]))
        z[m - 1] = -0.5 * (b[m - 1] - y[m - 2])
        y[:] = z

    return z, xii, true


# -----------------------------------------------
# Run the experiment at different resolutions
# -----------------------------------------------
u1, x1, t1 = jacobiMP(4000)
u2, x2, t2 = jacobiMP(1000)
u3, x3, t3 = jacobiMP(200)

# -----------------------------------------------
# Plot results
# -----------------------------------------------
fig, axs = plt.subplots(3, 2, figsize=(10, 9))

axs[0, 0].plot(x1, u1, '.', label='Jacobi')
axs[0, 0].plot(x1, t1, label='True')
axs[0, 0].set_title('True vs Jacobi')
axs[0, 0].set_ylabel('1000 iter (4000 pts)')
axs[0, 0].legend()

axs[0, 1].plot(x1, u1 - t1)
axs[0, 1].set_title('Error')

axs[1, 0].plot(x2, u2, '.', label='Jacobi')
axs[1, 0].plot(x2, t2, label='True')
axs[1, 0].set_ylabel('1000 iter (1000 pts)')
axs[1, 0].legend()

axs[1, 1].plot(x2, u2 - t2)
axs[1, 1].set_title('Error')

axs[2, 0].plot(x3, u3, '.', label='Jacobi')
axs[2, 0].plot(x3, t3, label='True')
axs[2, 0].set_ylabel('1000 iter (200 pts)')
axs[2, 0].legend()

axs[2, 1].plot(x3, u3 - t3)
axs[2, 1].set_title('Error')

plt.tight_layout()
plt.show()
