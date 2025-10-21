import matplotlib.pyplot as plt
import numpy as np

def sor_no_matrix ( f=None , n = 5 , w = 1. , a = None , h = None , u0 = None ,
                   max_iterations = 1000 , tolerance = 1e-5 , show_final_stats = True ) :
    
    """
    Name: sor_no_matrix_mk3
    Purpose: do SOR w/o a matrix
    Author: Joshua Belieu | Fletch
    Date written: 2025-10-20
    Last modified: 2024-10-20
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

                # TODO : double check these relations with notes in 'scratch'

                a_p12_x = 0.5 * ( a_1r + a_2r ) # a plus one half in x direction
                a_m12_x = 0.5 * ( a_1l + a_2l ) # a minus one half in x direction
                a_p12_y = 0.5 * ( a_1u + a_2u ) # a plus one half in y direction
                a_m12_y = 0.5 * ( a_1d + a_2d ) # a minus one half in y direction

                # TODO : keep on truckin'