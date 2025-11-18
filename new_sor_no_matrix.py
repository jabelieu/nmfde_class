import matplotlib.pyplot as plt
import numpy as np

def sor_no_matrix ( f=None , n = 5 , w = 1. , a = None , h = None , u0 = None ,
                   max_iterations = 1000 , tolerance = 1e-5 , show_final_stats = True ) :
    
    """
    Name: sor_no_matrix_mk3
    Purpose: do SOR w/o a matrix
    Author: Joshua Belieu | Fletch
    Date written: 2025-10-20
    Last modified: 2024-10-21
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

        interiors_indices = range ( 2 , n - 2 )
        for j in  interiors_indices :
            for i in interiors_indices :
                u_old = u.copy()

                u_center = u [ i , j ] # central point, what we're updating
                u_1r = u [ i + 1 , j ] # 1 unit right of the central point
                u_2r = u [ i + 2 , j ] # 2 units right of the central point
                u_1l = u [ i - 1 , j ] # 1 unit left of the central point
                u_2l = u [ i - 2 , j ] # 2 units left of the central point
                u_1u = u [ i , j + 1 ] # 1 unit up of the central point
                u_2u = u [ i , j + 2 ] # 2 units up of the central point
                u_1d = u [ i , j - 1 ] # 1 unit down of the central point
                u_2d = u [ i , j - 2 ] # 2 units down of the central point

                # below is the same shit but now for a.

                a_center = a [ i , j ]
                a_1r = a [ i + 1 , j ]
                a_2r = a [ i + 2 , j ]
                a_1l = a [ i - 1 , j ]
                a_2l = a [ i - 2 , j ]
                a_1u = a [ i , j + 1 ]
                a_2u = a [ i , j + 2 ]
                a_1d = a [ i , j - 1 ]
                a_2d = a [ i , j - 2 ]

                a_p12_x = 0.5 * ( a_1r + a_center ) # a plus one half in x direction
                a_m12_x = 0.5 * ( a_1l + a_center ) # a minus one half in x direction
                a_p12_y = 0.5 * ( a_1u + a_center ) # a plus one half in y direction
                a_m12_y = 0.5 * ( a_1d + a_center ) # a minus one half in y direction

                one_component = a_m12_x * u_1l + a_m12_y * u_1d + \
                                a_p12_x * u_1r + a_p12_y * u_1u
                
                two_component = -( a_m12_x * u_2l + a_m12_y * u_2d ) + \
                                a_p12_x * u_2r + a_p12_y * u_2u

                numerator = h ** 2 * f [ i , j ] + 0.5 * two_component \
                                + 2 * one_component
                denominator = a_m12_x + a_p12_x + a_m12_y + a_p12_y

                u_update = (2/3) * (numerator / denominator)

                u_new = ( 1 - w ) * u_center + w * u_update

                error = abs ( u_new - u_center )
                u [ i , j ] = u_new

        iteration += 1
        error_list.append(error)
        iteration_list.append(iteration)

    if show_final_stats is True :
        out_str = f'Final Stats : \n Iteration : {iteration} \n ||Error|| : {np.abs(error):.2e}'
        print(out_str)
    
    return u , iteration , iteration_list , error_list

unit_test_flag = 1

if unit_test_flag == 1 :

    n = 32
    b = np.zeros((n,n))
    x = np.linspace ( -1 , 1 , n )
    y = np.linspace ( -1 , 1 , n )
    X, Y = np.meshgrid(x, y)
    u_star = np.sin ( np.pi * X ) + np.sin ( np.pi * Y )
    a=np.ones((n,n))
    b = -a*(1-np.pi ** 2 ) * u_star

    u_sol , _,_,_ = sor_no_matrix(f=b,n=n,w=0.5,a=a,max_iterations=int(1e2),tolerance=1e-5)

    fig = plt.figure ( figsize = ( 12 , 10 ) , constrained_layout=True )
    ax1 = fig.add_subplot ( 131 , projection = '3d' )
    ax2 = fig.add_subplot ( 132 , projection = '3d' )
    ax3 = fig.add_subplot ( 133 , projection = '3d' )


    im1 = ax1.plot_surface(X, Y, u_sol, cmap='viridis', edgecolor="none")
    fig.colorbar(im1 , label="u",ax=ax1 , shrink = 0.25)
    ax1.set_title("u")
    im1 = ax1.plot_surface(X, Y, u_sol, cmap='viridis', edgecolor="none")

    im2 = ax2.plot_surface(X, Y, u_star, cmap='viridis', edgecolor="none")
    fig.colorbar(im2 , shrink = 0.25, label=r"$u^*$")
    ax2.set_xlabel("x")
    ax2.set_ylabel("y")
    ax2.set_zlabel(r"$u^*$")
    ax2.set_title(r"$u^*$")

    im3 = ax3.plot_surface(X, Y, u_sol-u_star, cmap='viridis', edgecolor="none")
    fig.colorbar(im3 , shrink = 0.25, label=r"$u-u^*$")
    ax3.set_xlabel("x")
    ax3.set_ylabel("y")
    ax3.set_zlabel("u")
    ax3.set_title(r"$u-u^*$")
    plt.show()