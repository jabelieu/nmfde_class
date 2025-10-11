# Inputs:
# N        : number of interior points in each direction
# h        : grid spacing
# a[i,j]   : variable coefficient on interior grid points
# f[i,j]   : right-hand side
# omega    : SOR relaxation parameter
# tol      : convergence tolerance
# max_iter : maximum number of iterations

# Initialize U (interior points) to zero
U = np.zeros((N, N))  

for iter in range(max_iter):
    max_change = 0.0
    
    # Sweep rowwise: j = 0..N-1 (y-direction), i = 0..N-1 (x-direction)
    for j in range(N):
        for i in range(N):
            # Fetch neighbors with Dirichlet BC (u=0 outside)
            u_center = U[i,j]
            u_ip1 = U[i+1,j] if i+1 < N else 0.0
            u_im1 = U[i-1,j] if i-1 >= 0 else 0.0
            u_jp1 = U[i,j+1] if j+1 < N else 0.0
            u_jm1 = U[i,j-1] if j-1 >= 0 else 0.0
            
            u_ip1_jp1 = U[i+1,j+1] if (i+1<N and j+1<N) else 0.0
            u_ip1_jm1 = U[i+1,j-1] if (i+1<N and j-1>=0) else 0.0
            u_im1_jp1 = U[i-1,j+1] if (i-1>=0 and j+1<N) else 0.0
            u_im1_jm1 = U[i-1,j-1] if (i-1>=0 and j-1>=0) else 0.0
            
            # Coefficients from 9-point stencil (constant a case for simplicity)
            a_c = a[i,j]
            
            laplacian = (4*(u_ip1 + u_im1 + u_jp1 + u_jm1) +
                         (u_ip1_jp1 + u_ip1_jm1 + u_im1_jp1 + u_im1_jm1) -
                         20 * u_center) / (6 * h**2)
            
            # Cross-derivative terms for variable a
            ax_ux = ((a[i+1,j] - a[i-1,j]) * (u_ip1 - u_im1)) / (4 * h**2) if 0<i<N-1 else 0.0
            ay_uy = ((a[i,j+1] - a[i,j-1]) * (u_jp1 - u_jm1)) / (4 * h**2) if 0<j<N-1 else 0.0
            
            # Residual for the PDE
            rhs = f[i,j] + a_c*laplacian + ax_ux + ay_uy
            
            # SOR update
            U_new = (1-omega)*u_center + omega*rhs
            max_change = max(max_change, abs(U_new - u_center))
            U[i,j] = U_new
    
    # Check convergence
    if max_change < tol:
        print(f"Converged in {iter+1} iterations")
        break
