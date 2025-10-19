import numpy as np
import matplotlib.pyplot as plt

def sor_no_matrix(f, n, w=1.5, a=None, h=None, u0=None,
                  max_iterations=1000, tolerance=1e-8,
                  show_final_stats=True):
    if h is None:
        h = 2 / (n - 1)
    if u0 is None:
        u = np.zeros((n, n))
    else:
        u = u0.copy()
    if a is None:
        a = np.ones((n, n))

    error = np.inf
    iteration = 0
    while error > tolerance and iteration < max_iterations:
        error = 0.0
        for j in range(1, n - 1):
            for i in range(1, n - 1):
                u_center = u[i, j]
                # neighbors
                u_left, u_right = u[i - 1, j], u[i + 1, j]
                u_down, u_up = u[i, j - 1], u[i, j + 1]
                u_ul, u_ur = u[i - 1, j + 1], u[i + 1, j + 1]
                u_dl, u_dr = u[i - 1, j - 1], u[i + 1, j - 1]
                a_center = a[i, j]

                rhs = (4*(u_left + u_right + u_up + u_down)
                       + (u_ul + u_ur + u_dl + u_dr)
                       + (6*h**2/a_center)*f[i, j]) / 20.0
                u_new = (1 - w) * u_center + w * rhs
                error = max(error, abs(u_new - u_center))
                u[i, j] = u_new
        iteration += 1

    if show_final_stats:
        print(f"Final Stats:\n Iterations: {iteration}\n Error: {error:.2e}")
    return u, iteration


# --- Order Verification ---
n_list = np.arange(10, 51, 10)
error_list, h_list = [], []

for n in n_list:
    h = 2 / (n - 1)
    x = np.linspace(-1, 1, n)
    y = np.linspace(-1, 1, n)
    X, Y = np.meshgrid(x, y)

    a = np.ones((n, n))
    u_star = np.sin(np.pi * (X - 1)) * np.sin(2 * np.pi * (Y - 1))
    f = 5 * np.pi**2 * u_star

    # u_sol, _ = sor_no_matrix(f=f, n=n, w=1.75, a=a, h=h, u0=np.zeros_like(f),
    #                          max_iterations=2000, tolerance=1e-10,
    #                          show_final_stats=False)
    u_sol, _ = sor_no_matrix(
    f=f, n=n, w=1.0, a=a, h=h, u0=np.zeros_like(f),
    max_iterations=10000, tolerance=1e-12, show_final_stats=False)


    error = np.max(np.abs(u_sol - u_star))
    error_list.append(error)
    h_list.append(h)

fig = plt.figure(figsize=(12, 5))

# 1️⃣ Numerical solution
ax1 = fig.add_subplot(1, 2, 1, projection='3d')
surf1 = ax1.plot_surface(X, Y, u_sol-u_star, cmap='viridis', edgecolor='none')
ax1.set_title("Numerical solution $u_{sol}$")
ax1.set_xlabel("x")
ax1.set_ylabel("y")
ax1.set_zlabel("u-u*")
fig.colorbar(surf1, ax=ax1, shrink=0.6, aspect=10)

# 2️⃣ Exact solution
ax2 = fig.add_subplot(1, 2, 2, projection='3d')
surf2 = ax2.plot_surface(X, Y, u_star, cmap='viridis', edgecolor='none')
ax2.set_title("Exact solution $u_{*}$")
ax2.set_xlabel("x")
ax2.set_ylabel("y")
ax2.set_zlabel("u")
fig.colorbar(surf2, ax=ax2, shrink=0.6, aspect=10)

# Nice viewing angle
for ax in [ax1, ax2]:
    ax.view_init(elev=30, azim=-60)

plt.tight_layout()
plt.show()