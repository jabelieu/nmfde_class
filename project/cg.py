#===============================================================================
# Name    : cg -> Conjugate Gradient
# Purpose : Solve linear systems using the Conjugate Gradient method
# Author  : Joshua Belieu | Fletch
# Date    : November 5, 2025
#===============================================================================
'                                 BEGIN PROGRAM                                '
#-------------------------------------------------------------------------------

"""
import libraries
"""

import matplotlib.pyplot as plt
import numpy as np

"""
functions
"""

def conjugate_gradient ( f , A , u0 = None ,
                        tolerance = 1e-3 ,
                        max_iterations = 100 ,
                        show_final_stats = False ) :
    
    """
    Name: conjugate_gradient
    Purpose: solve a given system using conjugate gradient
    Author: Joshua Belieu | Fletch
    Date written: 2025-11-05
    Last modified: 2025-11-05
    Inputs:
    - f: ndarray. rhs of diff. eq.
    - A: ndarray. matrix of the system.
    - u0: ndarray. initial guess.
    - tolerance: float. convergence tolerance.
    - max_iterations: int. maximum number of iterations.
    Outputs:
    - ui: ndarray. solution of the system.
    Dependencies:
    - numpy.
    """

    if u0 is None : # if no init guess, make one.
        uim1 = np.zeros_like ( f )
    else : # make a copy of the initial guess
        uim1 = u0.copy()

    # init residuals and directions
    rim1 = f - A @ uim1
    pim1 = rim1.copy()

    # bookkeeping
    iteration = 0
    convergence = np.inf

    while convergence > tolerance and iteration < max_iterations :

        # w is A * p so that when we calc the next p, the result is orthogonal
        # to A. a is the step size in the p direction.
        wim1 = A @ pim1
        aim1 = (rim1.T @ rim1) / ( pim1.T @ wim1 )

        # update solution and residuals
        ui = uim1 + aim1 * pim1
        ri = rim1 - aim1 * wim1

        # convergence metric
        convergence = np.linalg.norm ( ri )

        # step size for p update.
        bim1 = (ri.T @ ri) / (rim1.T @ rim1)
        # update p
        pi = ri + bim1 * pim1

        # prepare for next iteration
        uim1 = ui.copy()
        rim1 = ri.copy()
        pim1 = pi.copy()

        iteration += 1

    if show_final_stats == True :

        s = f" \n Final Convergence : {convergence:.2e} \n Iterations \
        : {iteration} / {max_iterations}"
        print ( s )

    return ui

"""
unit test
"""

test_flag = 1

if test_flag == 1 :

    """
    system : u''=-u, u(0)=0, u(2pi)=0
    exact solution : u=sin(x)
    """

    n = int ( 1e3 )
    x = np.linspace ( 0 , 2 * np.pi , n )
    u_star = np.sin ( x )
    h = x[1] - x[0]
    
    main_diagonal = -2 * np.ones ( n )
    off_diagonal = 1 * np.ones ( n - 1 )

    A = ( np.diag ( main_diagonal ) + np.diag ( off_diagonal , k = 1 ) + \
        np.diag ( off_diagonal , k = -1 ) ) / ( h ** 2 )
    
    # apply boundary conditions
    A[0,:] = 0
    A[0,0] = 1
    A[-1,:] = 0
    A[-1,-1] = 1
    
    f = -u_star

    u_sol = conjugate_gradient ( f , A , tolerance = 1e-10 ,
                                max_iterations = int ( 1e2 ) ,
                                 show_final_stats=True )

    fig,axs = plt.subplots(1,2, figsize=(12,5))

    axs[0].plot(x, u_star, label='Exact Solution')
    axs[0].plot(x, u_sol, label='CG Solution')
    axs[0].legend()
    axs[0].set_xlabel('x')
    axs[0].set_ylabel('u(x)')

    axs[1].plot( np.abs ( u_sol - u_star ) , label='Error' )
    axs[1].legend()
    axs[1].set_ylabel(r'$|u_{sol} - u^*|$')

    for ax in axs :
        ax.grid(ls='--', alpha=0.5)
        ax.tick_params(direction='in', which='both')

    plt.show()

#-------------------------------------------------------------------------------
'                                  END PROGRAM                                 '
#===============================================================================