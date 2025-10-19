import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.sparse import diags,csr_matrix
from scipy.integrate import simpson
from scipy.sparse.linalg import spsolve,eigsh
import scipy as scp
import numpy as np
import sympy as sp

def sor_no_matrix ( f=None , n = 5 , w = 1. , a = None , h = None , u0 = None , 
                   max_iterations = 1000 , tolerance = 1e-5 , show_final_stats = True ) :
    
    """
    Name: sor_no_matrix_mk1
    Purpose: do SOR w/o a ,atrix
    Author: Joshua Belieu | Fletch
    Date written: 2025-10-07(ish)
    Last modified: 2024-10-11
    Inputs:
    - f : ndarray. rhs of PDE
    - n  : int. system resolution, lattice points
    - w : float. smoother/mixing parameter. default value : 1 (should perform 
          like GS)
    - a : ndarray. diffusion coefficient matrix. default value : None
    - h : float. system stride. default value : None
    - u0 : ndarray. initial guess
    - max_iterations : int. maximum number of iterates. default value : 1000
    - tolerance : float. desired difference between iterates. default value : 
                  1e-5
    show_final_stats : bool. print final error and iteratoin to terminal. 
                       default value : True
    Outputs:
    - u : ndarray. most converged solution to PDE
    - iteration : int. final iteration count
    Dependencies:
    - None
    """

    if h is None :
        # print('lattice spacing not specified, assuming h=1/(n-1).')
        h = 2/(n-1)
    if u0 is None : 
        u = np.zeros ( (n , n) )
    else :
        u = u0.copy()
    if f is None : 
        f = np.zeros(n,n)
    if a is None : 
        a = np.ones((n,n))

    iteration = 0
    iteration_list = []
    error = np.inf
    while ( error > tolerance and iteration < max_iterations ) :
        iteration_list.append(iteration)
        error = 0.

        for j in  range ( 1,n-1 ) :
            for i in  range ( 1,n-1 ) :
                u_center = u[i,j]
                if i + 1 < n :
                    u_right = u [ i+1 , j ]
                else :
                    u_right = 0.
                if i - 1 >= 0 :
                    u_left = u [ i-1 , j ]
                else :
                    u_left = 0.
                if j + 1 < n :
                    u_up = u [ i , j+1 ]
                else :
                    u_up = 0.
                if j - 1 >= 0 :
                    u_down = u [ i , j-1 ]
                else : 
                    u_down = 0.
                if ( i+1<n and j+1<n ) :
                    u_upper_right = u [ i+1 , j+1 ]
                else :
                    u_upper_right = 0.
                if ( i-1>=0 and j+1<n ) :
                    u_upper_left = u [ i-1 , j+1 ]
                else :
                    u_upper_left = 0.
                if ( i+1<n and j-1>=0 ) :
                    u_lower_right = u [ i+1 , j-1 ]
                else :
                    u_lower_right = 0.
                if ( i-1>=0 and j-1>=0 ) :
                    u_lower_left = u [ i-1 , j-1 ]
                else :
                    u_lower_left = 0. # this block of code makes me sad and ChatGPT 
                                      # reccomends a more succinct stlye. i.e.,
                                      # u_bottom = u[i,j-1] if j-1 >= 0 else 0.0
                
                a_center = a[i,j]
                a_right = a[i+1,j] if i+1 < n else a[i,j]
                a_left = a[i-1,j] if i-1 >= 0 else a[i,j]
                a_up = a[i,j+1] if j+1 < n else a[i,j]
                a_down = a[i,j-1] if j-1 >= 0 else a[i,j]
                # u_laplacian = ( 4 * ( u_left + u_right + u_down + u_up ) + \
                #     u_lower_left + u_lower_right + u_upper_left + u_upper_right - \
                #     20 * u_center ) / ( 6 * h ** 2 )
                u_laplacian = ( 4 * ( u_left + u_right + u_down + u_up ) + \
                    u_lower_left + u_lower_right + u_upper_left + u_upper_right)/( 6 * h ** 2) - \
                    20 * u_center  / ( 6 * h ** 2 )
                # u_laplacian = (u_left + u_right + u_up + u_down - 4*u_center)/(h**2)
                
                if 0 < i < n-1 :
                    dadx_dudx = ( a_right - a_left )*( u_right - u_left ) / (4 * h**2)
                else :
                    dadx_dudx = 0.
                if 0 < j < n-1 :
                    dady_dudy = ( a_up - a_down )*( u_up - u_down ) / (4 * h**2)
                else :
                    dady_dudy = 0.

                # rhs = -f[i,j] - a_center * u_laplacian - dadx_dudx - dady_dudy

                # u_new = (1-w) * u_center + w * rhs#/(-20 * a_center / (6*h**2)) # rhs/(.) is test
                u_new = (1 - w)*u_center + w * (
    4*(u_left + u_right + u_up + u_down)
    + (u_lower_left + u_lower_right + u_upper_left + u_upper_right)
    + (6*h**2/a_center)*f[i,j]
) / 20.0
                error = max(error,abs( u_new - u_center ))
                u[i,j] = u_new

        iteration += 1
    
    if show_final_stats is True :
        out_str = f'Final Stats : \n Iteration : {iteration} \n ||Error|| : {np.abs(error):.2e}'
        print(out_str)
    
    return u , iteration

"""
Name: order tester
Purpose: is the order of the sor_no_matrix 4th order for constant a?
Author: Joshua Belieu | Fletch
Date written: 2025-10-07(ish)
Last modified: 2025-10-11
Inputs:
- n_list : ndarray. list of lattice points.
Outputs:
- ret: plot : canvas. error/h^i vs n_list. (next cell)
Dependencies:
- sor_no_matrix
"""

n_list = np.arange ( 10 , 50 + 1 , 10 )
error_list = []
h_list = []

for n in n_list :
    h = 2/(n-1)
    x = np.linspace(-1,1,n)
    y = np.linspace(-1,1,n)
    X,Y = np.meshgrid(x,y)
    a = np.ones((n,n))
    u_star = np.sin(np.pi*(X-1))*np.sin(2*np.pi*(Y-1))
    f = 5 * np.pi ** 2 * u_star
    u0 = np.zeros((n,n))

    u_sol,_ = sor_no_matrix (f=f,n=n,w=1.75,a=a,h=h,u0=u0,
                           max_iterations=1e3,tolerance=1e-10,
                           show_final_stats=False)
    error = np.max(np.abs(u_sol-u_star))
    error_list.append(error)
    h_list.append(h)

for order in range ( 0 , 4 + 1 ) :
    plt.plot ( n_list , np.array(error_list)/np.array(h_list)**order ,label=order,marker='s')

plt.yscale('log')
plt.legend()
plt.show()