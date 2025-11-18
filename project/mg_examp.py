import numpy as np
import matplotlib.pyplot as plt

from multigrid import tri_A_Jacobi  # or define in same file

# -----------------------------------------------
# Christlieb — CMSE 821 (Fall 2025)
# Jacobi iteration test for Poisson equation
# From LeVeque: demonstrates Jacobi smoothing behavior
# -----------------------------------------------

# Problem setup
NGP = 180
A, B = 0.0, 1.0
h = (B - A) / (NGP - 1)
xi = np.linspace(A, B, NGP)

# Dirichlet BCs
u0 = 1.0
uNGP = 3.0

# -----------------------------------------------
# RHS of d²u/dx² = f(x)
# -----------------------------------------------
pi = 2 * np.arcsin(1)
xii = xi[1:-1]  # interior points only
y = 60 * pi * xii**2
phi = 20 * pi * (xii**3)
RHS = h**2 * (-20 + 0.5 * (120 * pi * xii * np.cos(phi)) - 0.5 * (y**2) * np.sin(phi))

N = len(xii)

# -----------------------------------------------
# RHS of A u = b
# -----------------------------------------------
b = np.zeros(N)
b[0] = RHS[0] - u0
b[1:-1] = RHS[1:-1]
b[-1] = RHS[-1] - uNGP

# -----------------------------------------------
# True solution
# -----------------------------------------------
true = 1 + 12 * xii - 10 * xii**2 + 0.5 * np.sin(phi)

# -----------------------------------------------
# Initial guess — straight line
# -----------------------------------------------
ig = ((uNGP - u0) / (B - A)) * (xii - A) + u0

# -----------------------------------------------
# Jacobi iterations at different counts
# -----------------------------------------------
tol = 1e-16

u1 = tri_A_Jacobi(b, ig, n_stop=20, tol=tol, true_sol=true)
u2 = tri_A_Jacobi(b, ig, n_stop=100, tol=tol, true_sol=true)
u3 = tri_A_Jacobi(b, ig, n_stop=1000, tol=tol, true_sol=true)

# -----------------------------------------------
# Plotting
# -----------------------------------------------
fig, axs = plt.subplots(3, 2, figsize=(10, 9))

axs[0, 0].plot(xii, u1, '.', label='Jacobi (20 iter)')
axs[0, 0].plot(xii, true, label='True')
axs[0, 0].set_title('True vs Jacobi')
axs[0, 0].set_ylabel('20 iter')
axs[0, 0].legend()

axs[0, 1].plot(xii, u1 - true)
axs[0, 1].set_title('Error (180 points)')

axs[1, 0].plot(xii, u2, '.', label='Jacobi (100 iter)')
axs[1, 0].plot(xii, true, label='True')
axs[1, 0].set_ylabel('100 iter')
axs[1, 0].legend()

axs[1, 1].plot(xii, u2 - true)

axs[2, 0].plot(xii, u3, '.', label='Jacobi (1000 iter)')
axs[2, 0].plot(xii, true, label='True')
axs[2, 0].set_ylabel('1000 iter')
axs[2, 0].legend()

axs[2, 1].plot(xii, u3 - true)

plt.tight_layout()
plt.show()
