import numpy as np
from scipy.integrate import solve_ivp

import matplotlib.pyplot as plt

import sympy as sp
import scipy.sparse as sparse


rm = 1
rc = 5
Diff = 1.e-4
km = 1.
kc = 1.
ngrid = 25
Vratio = 0.01

def SDeriv(t, x):
    """
    Computes the spatial derivative of the system at time point, t
    :param t: time
    :param x: state variables
    :param diffeq_params: differential equation parameters
    :param integration_params: integration parameters
    :return: d: a list of values of the spatial derivative at time point, t
    """


    ###################################################################################
    ################################# Initialization ##################################
    ###################################################################################


    d = np.zeros(len(x)).tolist()
    # coefficients for the diffusion equation

    rmax = rc/rm
    rmin = 1.
    Deltar = (rmax - rmin)/(ngrid - 1)

    ###################################################################################
    ################################## MCP reactions ##################################
    ###################################################################################

    d[0] = (3*km/rm) * Diff * (x[1] - x[0])

    ####################################################################################
    ##################################### boundary of MCP ##############################
    ####################################################################################

    r = rmin

    first_coef = (Diff/rm**2) * ((r + 0.5*Deltar)**2)/(Deltar**2)
    second_coef = (Diff/rm) * km * ((r - 0.5*Deltar)**2)/Deltar
    d[1] = first_coef * (x[2] - x[1]) - second_coef * (x[1] - x[0])

    ####################################################################################
    ##################################### interior of cell #############################
    ####################################################################################

    for k in range(2, ngrid):

        # cell equations for ith compound in the cell
        r += Deltar

        d[k] = (Diff/((Deltar*rm*r)**2)) * (((r + 0.5 * Deltar) ** 2.) * (x[k+1] - x[k])
                                              - ((r - 0.5 * Deltar) ** 2.) * (x[k] - x[k-1]))


    ####################################################################################
    ###################### boundary of cell with external volume #######################
    ####################################################################################

    r = rmax

    first_coef = (Diff/rm) * kc* ((r + 0.5*Deltar)**2)/((r**2)*Deltar)
    second_coef = (Diff/(rm**2)) * ((r - 0.5*Deltar)**2)/((r**2)*(Deltar**2))

    d[-2] = first_coef * (x[-1] - x[-2]) - second_coef * (x[-2] - x[-3])

    #####################################################################################
    ######################### external volume equations #################################
    #####################################################################################
    d[-1] = 3*Vratio *Diff* (kc / rc) * (x[-2] - x[-1])  # external equation for concentration
    return d

def main():
    """
    plots of internal and external concentration

    """
    #################################################
    # Define spatial derivative and jacobian
    #################################################


    # spatial derivative
    x_list_sp = np.array(sp.symbols('x:' + str(ngrid+2)))

    #jacobian
    SDerivSymbolic = SDeriv(0,x_list_sp)
    SDerivGrad = sp.Matrix(SDerivSymbolic).jacobian(x_list_sp)
    SDerivGradFun = sp.lambdify(x_list_sp, SDerivGrad, 'numpy')
    SDerivGradFunSparse = lambda t,x: sparse.csr_matrix(SDerivGradFun(*x))

    #################################################
    # Integrate with BDF
    #################################################

    # initial conditions
    y0 = np.zeros(ngrid+2)
    y0[-1] = 10

    # time samples
    fintime = 5.e7

    tol = 1e-12

    #terminal event
    event = lambda t,y: np.absolute(SDeriv(t,y)[-1]) - tol
    event.terminal = True

    rmax = rc / rm
    rmin = 1.
    Deltar = (rmax - rmin) / (ngrid - 1)
    alpha = (Diff/(rm**2))



    sol = solve_ivp(SDeriv,[0, fintime], y0, method="BDF",jac=SDerivGradFunSparse,
                    atol=1.0e-10,rtol=1.0e-10)#max_step)

    #plt.plot(sol.t,sol.y[-1,:].T)
    #plt.show()
    M_cell = rc  # why?
    M_mcp = 1.
    Mgrid = np.linspace(M_mcp, M_cell, ngrid)
    DeltaM = np.divide((M_cell - M_mcp), ngrid)
    Mgridfull = np.concatenate(([M_mcp - DeltaM],Mgrid, [M_cell + DeltaM]))

    #for i in range(0,sol.y.shape[1],sol.y.shape[1]//10):
    #    plt.plot(Mgridfull/rm, sol.y[:,i].T)
    #plt.show()

    plt.plot(Mgridfull, sol.y[:, -1].T)
    plt.ylim([0,12])
    plt.show()

    #check mass balance
    volcell = (4*np.pi*((rc/rm)**3))/3
    volmcp = (4 * np.pi * ((rm/rm) ** 3)) / 3

    ext_masses_org = y0[-1] * (volcell/Vratio)

    print('starting external mass: ' + str(ext_masses_org))

    print('final external conc: ' + str(sol.y[-1, -1]))
    print('final cell conc: ' + str(sol.y[3,-1]))

    print(sol.y[0, -1])
    print(sol.y[-1,-1])
    ext_masses_fin = sol.y[-1, -1] * (volcell/Vratio)
    cell_masses_fin =sol.y[3,-1]* (volcell - volmcp)
    print('final external mass: ' + str(ext_masses_fin))
    print('final cell mass: ' + str(cell_masses_fin))
    mcp_masses_fin = sol.y[0, -1] * volmcp
    #print(ext_masses_org.sum())# + cell_masses_org.sum() )#+ mcp_masses_org.sum())
    print(ext_masses_fin.sum() + cell_masses_fin.sum() + mcp_masses_fin.sum())

if __name__ == '__main__':
    main()
