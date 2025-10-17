import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.sparse import diags,csr_matrix
from scipy.integrate import simpson
from scipy.sparse.linalg import spsolve,eigsh
import scipy as scp
import numpy as np
import sympy as sp

def fd_weights ( offsets , order , show_system = False ) :

    """
    Name          : fd_weights -> finite difference weights
    Purpose       : calculate the set of coefficients for a finite difference
                    approximation of the order-th derivative using the given
                    offsets. We are essentually solving the linear system
                    A c = b, where A is the Vandermonde matrix of the offsets,
                    c is the vector of coefficients we want to find, and b is
                    a vector of zeros with a factorial(order) in the order-th
                    position.
    Author          : christlieb, augmented by Joshua Belieu.
    Date augmented  : 2025-08-26
    Last modified   : 2025-08-26
    Inputs :
        - offsets     : list. list of offsets (in units of grid stride h)
        - order       : int. order of derivative to approximate
        - show_system : bool. if True, print the system of equations being
                        solved. default is False.
    Outputs:
        - <ret>: <type> ... <meaning/units/shape>
    Dependencies:
        - None.
    """

    '''
    list of offsets (in units of grid stride h), nsimplify allows floats.
    '''

    s = [ sp.nsimplify ( offset ) for offset in offsets ]
    m = len ( s )

    '''
    the Vandermonde matrix of the offsets. the "A" in our system Ac = b
    '''

    A = sp.Matrix ([[ s[j] ** k for j in range ( m ) ] for k in range ( m )])

    '''
    a vector of zeros with a factorial(order) in the order-th position. the 
    "b" in our system Ac = b.
    '''

    b = sp.Matrix ( [ 0 ] * m )
    b [ order ] = sp.factorial ( order )

    '''
    the coefficients we want to find. the "c" in our system Ac = b. LUsolve is 
    a decomposition method that splits A into a lower and upper triangular matrix
    and solves the system (Ux=y and Ly=c) in two steps.
    '''

    c = A.LUsolve ( b )

    if show_system :

        system = sp.Eq( sp.MatMul ( A , c , evaluate = False ) , b )
        sp.pprint(system)

    return sp.simplify ( c )

def derivative_1d_stencil ( n = 5 , offsets = [-1,0,1] , order = 2 , h = None ) :

    """
    Name: derivative_1d_stencil
    Purpose: what this subroutine computes and why
    Author: Joshua Belieu | Fletch
    Date written: 2025-09-24
    Last modified: 2025-09-24
    Inputs:
    - arg1: type ... meaning/units/shape
    - arg2: type ... meaning/units/shape
    Outputs:
    - ret: type ... meaning/units/shape
    Dependencies:
    - SubroutineA, SubroutineB, ...
    """

    if h == None :
        print ( "h not specified, assuming h = 1.0" )
        h = 1.0

    weights = [ float(num) for num in np.array(fd_weights ( offsets , order )).flatten()]

    stencil = diags ( weights , offsets , shape = ( n , n ) ) / h**order

    return stencil.toarray()

def build_interpolation_matrix ( coarse_resolution ) :

    fine_resolution = 2 * coarse_resolution - 1
    # fine_resolution = 2 * coarse_resolution
    interpolation_matrix = np.zeros ( ( fine_resolution , coarse_resolution ) )
    for coarse_index in range ( coarse_resolution ) :
        fine_index = 2 * coarse_index
        interpolation_matrix [ fine_index , coarse_index ] = 1.
        if fine_index + 1 < fine_resolution :
            interpolation_matrix [ fine_index + 1 , coarse_index ] += 0.5
        if fine_index - 1 >= 0 :
            interpolation_matrix [ fine_index - 1 , coarse_index ] += 0.5

    return interpolation_matrix

def one_step_weighted_jacobi ( A , u , f , w ) :

    """
    Name: one_step_weighted_jacobi
    Purpose: Calculate one step in the weighted jacobi scheme. This is a single
    step towards the solution of Au=f.
    Author: Joshua Belieu | Fletch
    Date written: 2025-10-15
    Last modified: 2025-10-15
    Inputs:
    - A : ndarray. the derivative matrix of the system
    - u : ndarray. the n-th iteration of a solution
    - f : ndarray. the RHS of the differential equation.
    - w : float. the mixing parameter that controls how much of the update is
    combined with the current form of the solution.
    Outputs:
    - u_new : ndarray. the (n+1)-th iteration of a solution.
    Dependencies:
    - Numpy.
    """

    D = np.diag ( A )
    D_inv = 1/D

    # print(u.shape,f.shape,A.shape,D.shape)

    u_new = u + w * D_inv * ( f - A @ u )

    return u_new

def two_cycle_vgrid ( Ah , fh , interpolation_matrix , restriction_matrix , nu1 ,
                      nu2 , w , coarse_steps = 10 , u0 = None , cheeky_solve = False ) :
    
    if u0 is None :
        u = np.zeros_like ( fh )
    else :
        u = u0.copy()
    for iteration in range ( nu1 ) :
        u = one_step_weighted_jacobi ( Ah , u , fh , w )

    r = fh - Ah @ u

    # print(restriction_matrix.shape,r.shape)

    r2h = restriction_matrix @ r
    A2h = restriction_matrix @ Ah @ interpolation_matrix

    if cheeky_solve == True :
        error_coarse = scp.linalg.solve ( A2h , r2h )
    else :
        error_coarse = np.zeros_like(r2h)
        for i in range(100):
            error_coarse = one_step_weighted_jacobi(A=A2h, u=error_coarse, f=r2h, w=w)

    
    u += interpolation_matrix @ error_coarse

    for iteration in range ( nu2 ) :
        u = one_step_weighted_jacobi ( A=Ah , u=u, f=fh , w=w )

    return u

'''
unit test
'''

n_2h = 500
n_h = 2 * n_2h - 1

x = np.linspace(0, 1, n_h)
u_star = np.sin(np.pi * x)
fh = -np.pi**2 * u_star
Ah = derivative_1d_stencil(n=n_h, offsets=[-1,0,1], order=2, h=1/(n_h-1))

interp_mat = build_interpolation_matrix(n_2h)
restric_mat = 0.5 * interp_mat.T

u_sol = two_cycle_vgrid(
    Ah=Ah,fh=fh,interpolation_matrix=interp_mat,restriction_matrix=restric_mat,
    nu1=5,nu2=5,w=1/3,cheeky_solve=True)
# for _ in range(5):
#     u_sol = two_cycle_vgrid(
#         Ah=Ah, fh=fh, interpolation_matrix=interp_mat, restriction_matrix=restric_mat,
#         nu1=3, nu2=3, w=2/3, cheeky_solve=True, u0=u_sol
#     )


# plt.plot(x,u_sol,label='Multigrid')
# plt.plot(x,u_star,label='True')
# plt.legend()
# plt.tick_params(direction='in')
# plt.grid(ls='--',alpha=0.5)
# plt.savefig('bar.png')
# plt.show()


n_points = 20
k_list = np.linspace ( 0 , 10 , n_points )
n_coarse_list = np.linspace (5e1,1e3,n_points,dtype=int)
error_list = []
n_fine_list = 2 * n_coarse_list - 1
h_list = 1/(n_fine_list-1)

for i,h in enumerate(h_list) :
    for k in k_list :
        n_fine = n_fine_list [i]
        n_coarse = n_coarse_list[i]

        x = np.linspace(0, 1, n_fine)
        u_star = np.sin(np.pi * x)
        fh = (-np.pi**2 + k ** 2) * u_star
        Ah = -derivative_1d_stencil(n=n_fine, offsets=[-1,0,1], order=2, h=1/(n_fine-1))+np.diag ( k**2* np.ones(n_fine))

        interp_mat = build_interpolation_matrix(n_coarse)
        restric_mat = 0.5 * interp_mat.T

        u_sol = two_cycle_vgrid(
            Ah=Ah,fh=fh,interpolation_matrix=interp_mat,restriction_matrix=restric_mat,
            nu1=3,nu2=3,w=0.9,cheeky_solve=True)
# plt.plot(x,u_sol,label='Multigrid')
# plt.plot(x,u_star,label='True')
# plt.legend()
        error = np.linalg.norm ( u_sol - u_star )
        
        error_list.append ( np.linalg.norm ( u_sol - u_star ) )
error_array = np.array(error_list).reshape(len(h_list), len(k_list))

convergance_factor_wrt_k = error_array[:, 1:] / error_array[:, :-1]
convergance_factor_wrt_h = error_array[1:, :] / error_array[:-1, :]

K_k, H_k = np.meshgrid(k_list[1:], h_list)
K_h, H_h = np.meshgrid(k_list, h_list[1:])

fig = plt.figure(figsize=(18, 8))

ax1 = fig.add_subplot(121, projection='3d')
surf1 = ax1.plot_surface(K_k, H_k, convergance_factor_wrt_k,
                         cmap=cm.viridis, edgecolor='none', alpha=0.9)
ax1.set_xlabel(r"$k$",fontsize=14)
ax1.set_ylabel(r"fine step size, $h$",fontsize=14)
ax1.set_zlabel("Convergance Factor",fontsize=14)
ax1.set_title ( 'Convergance factor with varying k',fontsize=14 )
ax1.view_init(elev=30, azim=240)
# fig.colorbar(surf1, ax=ax1, shrink=0.6, aspect=10, label='Convergence factor')

ax2 = fig.add_subplot(122, projection='3d')
surf2 = ax2.plot_surface(K_h, H_h, convergance_factor_wrt_h,
                         cmap=cm.viridis, edgecolor='none', alpha=0.9)
ax2.set_xlabel(r"$k$",fontsize=14)
ax2.set_ylabel(r"fine step size, $h$",fontsize=14)
ax2.set_zlabel("Convergance Factor",fontsize=14)
ax2.set_title ( 'Convergance factor with varying h',fontsize=14 )
ax2.view_init(elev=30, azim=210)
# fig.colorbar(surf2, ax=ax2, shrink=0.6, aspect=10, label='Convergence factor')


plt.tight_layout()
# plt.savefig('hw2_p4_DONTDELETE_moreRES.png')
# plt.show()

# number_cycles = range ( 8 )
# error_list = []

# n_coarse = 100
# n_fine = 2 * n_coarse - 1
# k=1
# h = 1/(n_fine-1)
# u_sol = np.zeros(n_fine)
# interp_mat = build_interpolation_matrix(n_coarse)
# restric_mat = 0.5 * interp_mat.T
# x = np.linspace(0, 1, n_fine)
# u_star = np.sin(np.pi * x)
# fh = (-np.pi**2 + k ** 2) * u_star
# Ah = derivative_1d_stencil(n=n_fine, offsets=[-1,0,1], 
#                     order=2, h=1/(n_fine-1))+np.diag ( k**2* np.ones(n_fine))
# for cycle_number in number_cycles :

#     u_sol = two_cycle_vgrid(
#         Ah=Ah,fh=fh,interpolation_matrix=interp_mat,restriction_matrix=restric_mat,
#         nu1=3,nu2=3,w=0.9,u0=u_sol,cheeky_solve=True)
#     error = np.linalg.norm ( u_sol - u_star )
#     error_list.append ( np.linalg.norm ( u_sol - u_star ) )    
# error_array = np.array(error_list[1:])/np.array(error_list[:-1])

# plt.plot(x,u_sol,label='Multigrid')
# plt.plot(x,u_star,label='True')
# plt.legend()

# plt.figure()
# plt.plot(number_cycles[1:],error_array,marker='s')
# # plt.yscale('log')
# plt.tick_params(direction='in')
# plt.grid (ls='--',alpha=0.5 )
plt.show()