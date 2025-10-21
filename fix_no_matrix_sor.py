import matplotlib.pyplot as plt
import numpy as np

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

    if h is None : # if no lattice spacing given, assume below form
        # print('lattice spacing not specified, assuming h=1/(n-1).')
        h = 2/(n-1)
    if u0 is None :  # if no guess, zeros.
        u = np.zeros ( (n , n) )
    else : # copy init guess
        u = u0.copy()
    if f is None : # if no RHS, zeros
        f = np.zeros(n,n)
    if a is None : # if no diffusivity coeff, ones.
        a = np.ones((n,n))

    iteration = 0 # iteration counter.
    iteration_list = [] # bookkeeping
    error_list = [] # "
    error = np.inf # init error
    
    # loop until convergance or max iterations met.
    while ( error > tolerance and iteration < max_iterations ) :
        # iteration_list.append(iteration)
        error = 0. # re-init error

        # setup u w/ BCs.
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
                # setup a
                a_center = a[i,j]
                a_right = a[i+1,j] if i+1 < n else a[i,j]
                a_left = a[i-1,j] if i-1 >= 0 else a[i,j]
                a_up = a[i,j+1] if j+1 < n else a[i,j]
                a_down = a[i,j-1] if j-1 >= 0 else a[i,j]
                # u_laplacian = ( 4 * ( u_left + u_right + u_down + u_up ) + \
                #     u_lower_left + u_lower_right + u_upper_left + u_upper_right - \
                #     20 * u_center ) / ( 6 * h ** 2 )
                # laplacian term
                u_laplacian = ( 4 * ( u_left + u_right + u_down + u_up ) + \
                    u_lower_left + u_lower_right + u_upper_left + u_upper_right)/( 6 * h ** 2) - \
                    20 * u_center  / ( 6 * h ** 2 )
                # u_laplacian = (u_left + u_right + u_up + u_down - 4*u_center)/(h**2)
                
                # cross derivative terms
                if 0 < i < n-1 :
                    dadx_dudx = ( a_right - a_left )*( u_right - u_left ) / (4 * h**2)
                else :
                    dadx_dudx = 0.
                if 0 < j < n-1 :
                    dady_dudy = ( a_up - a_down )*( u_up - u_down ) / (4 * h**2)
                else :
                    dady_dudy = 0.
                
                # print(dadx_dudx,dady_dudy)

                rhs = -f[i,j] - a_center * u_laplacian - dadx_dudx - dady_dudy

                # u_new = (1-w) * u_center + w * rhs#/(-20 * a_center / (6*h**2)) # rhs/(.) is test
                # form u_new
                u_new = (1 - w)*u_center + w * (
    4*(u_left + u_right + u_up + u_down)
    + (u_lower_left + u_lower_right + u_upper_left + u_upper_right)
    + (6*h**2/a_center)*f[i,j]
) / 20.0
                # ger error matric
                error = max(error,abs( u_new - u_center ))
                # update u
                u[i,j] = u_new

        # bookkeeping
        iteration += 1
        error_list.append(error)
        iteration_list.append(iteration)
    
    # show final stats
    if show_final_stats is True :
        out_str = f'Final Stats : \n Iteration : {iteration} \n ||Error|| : {np.abs(error):.2e}'
        print(out_str)
    
    return u , iteration , iteration_list , error_list

'''
unit test
'''

unit_test_flag = 1

if unit_test_flag == 1 :

    n = 32
    b = np.zeros((n,n))
    x = np.linspace ( -1 , 1 , n )
    y = np.linspace ( -1 , 1 , n )
    X, Y = np.meshgrid(x, y)
    u_star = np.sin ( np.pi * X ) + np.sin ( np.pi * Y )
    b = -np.pi ** 2 * u_star

    # for i in range ( n ) :
    #     for j in range ( n ) :
    #         if 1/3 <= x[i] <= 1/2 and 1/2 <= y[j] <= 2/3 :
    #             b[i,j] = 1
    # a = 1+X
    u_sol , _,_,_ = sor_no_matrix(f=b,n=n,w=0.5,a=np.ones((n,n)),max_iterations=int(1e2),tolerance=1e-5)

    fig = plt.figure ( figsize = ( 12 , 10 ) , constrained_layout=True )
    ax1 = fig.add_subplot ( 131 , projection = '3d' )
    ax2 = fig.add_subplot ( 132 , projection = '3d' )
    ax3 = fig.add_subplot ( 133 , projection = '3d' )


    im1 = ax1.plot_surface(X, Y, u_sol, cmap='viridis', edgecolor="none")
    fig.colorbar(im1 , label="u(x,y)",ax=ax1 , shrink = 0.25)
    ax1.set_title("u(x,y)")
    im1 = ax1.plot_surface(X, Y, u_sol, cmap='viridis', edgecolor="none")

    im2 = ax2.plot_surface(X, Y, u_star, cmap='viridis', edgecolor="none")
    fig.colorbar(im2 , shrink = 0.25, label=r"$u^*(x,y)$")
    ax2.set_xlabel("x")
    ax2.set_ylabel("y")
    ax2.set_zlabel(r"$u^*(x,y)$")
    ax2.set_title(r"$u^*$")

    im3 = ax3.plot_surface(X, Y, u_sol-u_star, cmap='viridis', edgecolor="none")
    fig.colorbar(im3 , shrink = 0.25, label="u(x,y)")
    ax3.set_xlabel("x")
    ax3.set_ylabel("y")
    ax3.set_zlabel("u(x,y)")
    ax3.set_zlabel(r"$u(x,y)-u^*$")
    plt.show()