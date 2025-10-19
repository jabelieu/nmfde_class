import numpy as np
import matplotlib.pyplot as plt
import time
from math import pi
import scipy.sparse as sp
import scipy.sparse.linalg as spla

def sor_no_matrix_9pt_const(f, n=32, w=1.5, h=None, u0=None,
                           max_iterations=20000, tolerance=1e-8, show_final_stats=True):
    """
    SOR for -Delta u = f on [-1,1]^2 with zero Dirichlet BC using a 9-point 4th-order Laplacian.
    Constant coefficient a = 1.
    """
    if h is None:
        h = 2.0 / (n - 1)
    if u0 is None:
        u = np.zeros((n, n), dtype=float)
    else:
        u = u0.copy().astype(float)

    denom = 20.0
    coeff = 6.0 * h * h
    residual_history = []
    iteration = 0
    max_iter = int(max_iterations)
    tol = float(tolerance)

    def compute_residual(u_arr, f_arr):
        uloc = u_arr
        nloc = uloc.shape[0]
        lap = np.zeros_like(uloc)
        for i in range(1, nloc-1):
            for j in range(1, nloc-1):
                lap_ij = (4*(uloc[i-1,j] + uloc[i+1,j] + uloc[i,j-1] + uloc[i,j+1]) +
                          (uloc[i-1,j-1] + uloc[i-1,j+1] + uloc[i+1,j-1] + uloc[i+1,j+1]) -
                          20.0*uloc[i,j]) / (6.0 * h * h)
                lap[i,j] = lap_ij
        r = -lap - f_arr
        r_interior = r[1:-1, 1:-1].ravel()
        return np.sqrt(np.sum(r_interior**2) * (h*h))

    residual_history.append(compute_residual(u, f))

    while iteration < max_iter and residual_history[-1] > tol:
        for i in range(1, n-1):
            for j in range(1, n-1):
                u_center = u[i,j]
                u_left = u[i-1,j]; u_right = u[i+1,j]
                u_down = u[i,j-1]; u_up = u[i,j+1]
                u_ll = u[i-1,j-1]; u_lr = u[i+1,j-1]
                u_ul = u[i-1,j+1]; u_ur = u[i+1,j+1]

                numerator = (4*(u_left + u_right + u_down + u_up) +
                             (u_ll + u_lr + u_ul + u_ur) + coeff * f[i,j])
                u_new_unrelaxed = numerator / denom
                u_new = (1.0 - w) * u_center + w * u_new_unrelaxed
                u[i,j] = u_new
        iteration += 1
        residual_history.append(compute_residual(u, f))
        if not np.isfinite(residual_history[-1]) or residual_history[-1] > 1e6:
            print("SOR diverged or became unstable. Stopping early.")
            break

    if show_final_stats:
        print(f"SOR finished: iterations={iteration}, final residual (L2)={residual_history[-1]:.3e}")
    return u, iteration, np.array(residual_history)


def manufactured_problem_10(n):
    x = np.linspace(-1.0, 1.0, n)
    y = np.linspace(-1.0, 1.0, n)
    h = 2.0 / (n - 1)
    X, Y = np.meshgrid(x, y, indexing='ij')
    u_exact = np.sin(pi*(X - 1.0)) * np.sin(2.0*pi*(Y - 1.0))
    f = 5.0 * (pi**2) * np.sin(pi*(X - 1.0)) * np.sin(2.0*pi*(Y - 1.0))
    return f, u_exact, x, y, h


# Small driver: run n = 16, 32, 64 with SOR
ns = [16, 32, 64]
errors_L2 = []
hs = []
sweep_info = []
max_iter = 20000

for n in ns:
    print(f"\nRunning n = {n} ...")
    f, u_exact, x, y, h = manufactured_problem_10(n)
    u0 = np.zeros((n,n))
    t0 = time.perf_counter()
    u_num, iters, res_hist = sor_no_matrix_9pt_const(f=f, n=n, w=1.5, h=h, u0=u0,
                                                     max_iterations=max_iter, tolerance=1e-8, show_final_stats=True)
    t1 = time.perf_counter()
    err = u_num - u_exact
    err_interior = err[1:-1,1:-1].ravel()
    L2_err = np.sqrt(np.sum(err_interior**2) * (h*h))
    errors_L2.append(L2_err)
    hs.append(h)
    sweep_info.append({'n': n, 'iters': iters, 'time_s': t1-t0, 'final_res': res_hist[-1]})
    print(f"n={n}, h={h:.4e}, L2 error = {L2_err:.3e}, time = {t1-t0:.3f}s, iterations = {iters}")

# plot errors
plt.figure(figsize=(6,5))
plt.loglog(hs, errors_L2, 'o-')
plt.xlabel('h'); plt.ylabel('L2 error')
slope = np.polyfit(np.log(hs), np.log(errors_L2), 1)[0]
plt.title(f'Problem 10: error vs h (estimated slope = {slope:.2f})')
plt.grid(True, which="both", ls="--", alpha=0.5)
plt.gca().invert_xaxis()
plt.show()

print("\nSummary of runs:")
for info in sweep_info:
    print(info)
print(f"\nEstimated convergence order ~ {abs(slope):.2f} (expect ~4.0)")
