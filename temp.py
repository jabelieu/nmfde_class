import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt
import math

def idx(i, j, N):
    return i*(N+1) + j

def build_f(N):
    h = 1.0/N
    xs = np.linspace(0,1,N+1)
    ys = np.linspace(0,1,N+1)
    F = np.zeros((N+1, N+1), dtype=float)
    for i,x in enumerate(xs):
        for j,y in enumerate(ys):
            if (1/3 <= x <= 1/2) and (1/2 <= y <= 2/3):
                F[i,j] = 1.0
    return F

def compute_ghost_coeffs(h):
    K = np.arange(0,5)
    V = np.zeros((5,5))
    for r,k in enumerate(K):
        for m in range(5):
            V[r,m] = ((-k*h)**m) / math.factorial(m)
    r = np.array([(h**m) / math.factorial(m) for m in range(5)])
    cols_rest = [0,2,3,4]
    M = V[:, cols_rest]
    v1 = V[:,1].reshape(5,1)
    r_rest = r[cols_rest]
    Minv = np.linalg.pinv(M)
    factor = r_rest @ Minv
    alpha = np.zeros(5)
    for e in range(5):
        Ue = np.zeros(5)
        Ue[e] = 1.0
        D_full = np.linalg.pinv(V) @ Ue
        alpha[e] = r @ D_full
    D_rest_q = - Minv @ v1.flatten()
    u_from_q = r_rest @ D_rest_q + r[1]*1.0
    beta = u_from_q
    return alpha, beta

def assemble_system(N, f_array, g_dirichlet=None, q_neumann=None, use_4th_neumann=True):
    if g_dirichlet is None:
        g_dirichlet = lambda x,y: 0.0
    if q_neumann is None:
        q_neumann = lambda y: 0.0
    h = 1.0/N
    size = (N+1)*(N+1)
    rows = []
    cols = []
    data = []
    b = np.zeros(size)
    alpha, beta = compute_ghost_coeffs(h) if use_4th_neumann else (None, None)
    lap_f = np.zeros_like(f_array)
    for i in range(1, N):
        for j in range(1, N):
            lap_f[i,j] = (f_array[i+1,j] + f_array[i-1,j] + f_array[i,j+1] + f_array[i,j-1] - 4*f_array[i,j]) / (h*h)
    for i in range(0, N+1):
        x = i*h
        for j in range(0, N+1):
            y = j*h
            k = idx(i,j,N)
            if (i == 0) or (j == 0) or (j == N):
                rows.append(k); cols.append(k); data.append(1.0)
                b[k] = g_dirichlet(x,y)
                continue
            coeff_factor = 1.0/(6.0*h*h)
            center_coeff = -20.0 * coeff_factor
            rows.append(k); cols.append(k); data.append(center_coeff)
            adj = [(-1,0),(1,0),(0,-1),(0,1)]
            for di,dj in adj:
                ii = i+di; jj = j+dj
                if 0 <= ii <= N and 0 <= jj <= N:
                    rows.append(k); cols.append(idx(ii,jj,N)); data.append(4.0*coeff_factor)
            corners = [(-1,-1),(-1,1),(1,-1),(1,1)]
            for di,dj in corners:
                ii = i+di; jj = j+dj
                if 0 <= ii <= N and 0 <= jj <= N:
                    rows.append(k); cols.append(idx(ii,jj,N)); data.append(1.0*coeff_factor)
            b[k] = f_array[i,j] + (h*h/12.0)*lap_f[i,j]
    A = sparse.coo_matrix((data,(rows,cols)), shape=(size,size)).tocsr()
    for j in range(1, N):
        i = N
        k = idx(i,j,N)
        A.data[A.indptr[k]:A.indptr[k+1]] = 0.0
        row_dict = {}
        def add_to_row(col_index, val):
            row_dict[col_index] = row_dict.get(col_index, 0.0) + val
        coeff_factor = 1.0/(6.0*h*h)
        center = -20.0 * coeff_factor
        add_to_row(k, center)
        neighbors = [
            (N-1,j-1,1.0), (N-1,j+1,1.0), (N+1,j-1,1.0), (N+1,j+1,1.0),
            (N-1,j,4.0), (N+1,j,4.0), (N,j-1,4.0), (N,j+1,4.0)
        ]
        for ii,jj,wt in neighbors:
            if 0 <= ii <= N and 0 <= jj <= N:
                add_to_row(idx(ii,jj,N), wt*coeff_factor)
            else:
                if use_4th_neumann:
                    for k0 in range(5):
                        ii_local = N - k0
                        col = idx(ii_local, jj, N)
                        add_to_row(col, alpha[k0] * coeff_factor)
                    b[k] += (beta * h) * coeff_factor * q_neumann(jj*h)
                else:
                    col = idx(N-1, jj, N)
                    add_to_row(col, 1.0 * coeff_factor)
        A[k,:] = 0.0
        cols_list = np.fromiter(row_dict.keys(), dtype=int)
        vals_list = np.fromiter((row_dict[c] for c in cols_list), dtype=float)
        A[k, cols_list] = vals_list
    for i in range(0, N+1):
        for j in [0, N]:
            k = idx(i,j,N)
            A[k,:] = 0.0
            A[k,k] = 1.0
            b[k] = g_dirichlet(i*h, j*h)
    for j in range(0, N+1):
        i = 0
        k = idx(i,j,N)
        A[k,:] = 0.0
        A[k,k] = 1.0
        b[k] = g_dirichlet(0.0, j*h)
    return A.tocsr(), b

def solve_poisson(N, use_4th_neumann=True):
    f = build_f(N)
    gfun = lambda x,y: 0.0
    qfun = lambda y: 0.0
    A, b = assemble_system(N, f, g_dirichlet=gfun, q_neumann=qfun, use_4th_neumann=use_4th_neumann)
    plt.figure()
    plt.imshow(b.reshape((N+1,N+1)), cmap='viridis')
    plt.savefig('temp_b.png')
    plt.figure(figsize=(20,10))
    plt.imshow(A.toarray(), cmap='viridis')
    plt.colorbar()
    for i in range((N+1)*(N+1)):
        for j in range((N+1)*(N+1)):
            plt.text(j, i, f"{A[i,j]:.1f}",
                    ha="center", va="center",
                    color="w", fontsize=10)
    # plt.grid(True, color='black')
    # plt.xticks(np.arange(-0.5, A.shape[1], 1),[]) # Hide x-axis labels
    # plt.yticks(np.arange(-0.5, A.shape[0], 1),[]) # Hide y-axis labels
    plt.savefig('temp_matrix.png')
    uvec = spsolve(A, b)
    U = uvec.reshape((N+1, N+1))
    return U, f

def plot_solution(U):
    # N = U.shape[0]-1
    xs = np.linspace(0,1,N+1)
    ys = np.linspace(0,1,N+1)
    X, Y = np.meshgrid(xs, ys, indexing='ij')
    # plt.figure(figsize=(6,5))
    # plt.contourf(X, Y, U, levels=30, cmap='viridis')
    # plt.colorbar()
    # plt.title('Numerical solution u')
    # plt.xlabel('x'); plt.ylabel('y')
    fig = plt.figure ( figsize = ( 12 , 10 ) , constrained_layout=True )
    ax1 = fig.add_subplot ( 121 )
    ax2 = fig.add_subplot ( 122 , projection = '3d' )

    im1 = ax1.imshow(U, origin="lower", extent=[0,1,0,1])
    fig.colorbar(im1 , label="u(x,y)",ax=ax1 , shrink = 0.5 )
    ax1.set_title("u(x,y) Heat Map")

    im2 = ax2.plot_surface(X, Y, -1*U, cmap='viridis', edgecolor="none")
    fig.colorbar(im2 , shrink = 0.5 , aspect=20, label="u(x,y)")
    ax2.set_xlabel("x")
    ax2.set_ylabel("y")
    ax2.set_zlabel("u(x,y)")
    ax2.set_title("u(x,y) in 3D")
    plt.savefig('temp_solution.png')
    # plt.show()

N = 4
U, f = solve_poisson(N, use_4th_neumann=True)
plot_solution(U)