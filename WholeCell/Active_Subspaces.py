import numpy as np
from scipy.integrate import solve_ivp

import matplotlib.pyplot as plt
import warnings
import sympy as sp
import scipy.sparse as sparse
from Whole_Cell_Engineered_System_IcdE import *
from Whole_Cell_Engineered_System_IcdE_LocalSensitivity_Analysis import *


def cost_fun_unif(jac,M,names,bounds,maxN=10):
    """
    Monte Carlo integration estimate of the cost function
    :param jac: jacobian of the function
    :param M: number of parameters that the function depends on
    :param bounds: bounds on parameter space
    :param maxN: maximum number iterations for Monte Carlo integration
    :return (1/N)*C: Monte Carlo estimate of the cost function
    """
    N = 0
    C = np.zeros((M,M))
    bound_diff = bounds[:,1] - bounds[:,0]
    while 1:
        # updates
        x = bound_diff*np.random.uniform(0,1,size=M) + bounds[:,0]
        dict_vals = {name:val for name,val in zip(names,x)}
        j = jac(dict_vals)
        outer_prod = np.prod(bound_diff)*np.outer(j,j)  # TODO: Check if needed to multiply by some probability function
        C +=outer_prod
        N +=1
        # break statements
        if N > maxN:
            warnings.warn("The cost matrix may not converged. The maximum N needs to be increased.")
            break

        fro_outer = np.linalg.norm(outer_prod, ord='fro')
        fro_C = np.linalg.norm(C, ord='fro')
        print(fro_outer/fro_C)
        if (N > 0) and (fro_outer / fro_C < 1.e-4):
            break

    return (1/N)*C

def jac(dict_vals,integration_params, SDerivSymbolicJacParamsLambFun,
        SDerivSymbolicJacConcLambFun, dSensSymJacSparseMatLamFun,
        tol, fintime = 3.e6):

    # get integration parameters
    params_sens_dict = integration_params['Sensitivity Params']
    nSensitivityEqs = integration_params['nSensitivityEqs']
    nParams = integration_params['nParams']
    nVars = integration_params['nVars']

    # put differential equation parameters in the dictionary format
    diffeq_params = integration_params['diffeq_params'].copy()
    for key, value in dict_vals.items():
        diffeq_params[key] = value
    dimscalings = integration_params['dimscalings']


    # initial conditions

    y0 = np.zeros(nVars)     # initial conditions -- state variable
    y0[-5] = diffeq_params['GInit'] / dimscalings['G0']  # y0[-5] gives the initial state of the external substrate.
    y0[-1] = diffeq_params['IInit'] / dimscalings['I0']  # y0[-1] gives the initial state of the external substrate.
    y0[0] = diffeq_params['NInit'] / dimscalings['N0']  # y0[5] gives the initial state of the external substrate.
    y0[1] = diffeq_params['DInit'] / dimscalings['D0']  # y0[6] gives the initial state of the external substrate.

    sens0 = np.zeros(nSensitivityEqs)  #initial conditions -- sensitivity equation
    for i,param in enumerate(params_sens_dict):
        if param in ['GInit', 'IInit', 'NInit', 'DInit']:
            sens0[i:nSensitivityEqs:nParams] = 1/diffeq_params[param]
    xs0 = np.concatenate([y0,sens0])

    # solve differential equation
    dSensParams = lambda t,xs: dSens(t, xs, diffeq_params, integration_params,
                                     SDerivSymbolicJacParamsLambFun, SDerivSymbolicJacConcLambFun) #senstivity equation

    dSensSymJacSparseMatLamFunTXS = lambda t,xs: dSensSymJacSparseMatLamFun(t,xs,list(dict_vals.values())) #jacobian


    event = lambda t,xs: np.absolute(dSensParams(t,xs)[nVars-1]) - tol #terminal event
    event.terminal = True

    sol = solve_ivp(dSensParams,[0, fintime], xs0, method="BDF", jac = dSensSymJacSparseMatLamFunTXS,
                    atol=1.0e-2, rtol=1.0e-2)

    # scalings = list(dimscalings.values())
    #
    # for i in range(7):
    #     plt.plot(sol.t,sol.y[i,:].T*scalings[i])
    # plt.legend(['N','D','G','H','P','A','I'],loc='upper right')
    # plt.show()


    # for i in range(5):
    #     plt.plot(sol.t,sol.y[(nVars-i-1):(nVars-i),:].T*scalings[-i-1])
    # plt.legend(['G','H','P','A','I'],loc='upper right')
    # plt.title('Plot of External concentrations')
    # plt.show()

    return sol.y[nVars:,-1].T

def create_jac_sens_param(x_sp,sensitivity_sp,param_sp, integration_params,
                    SDerivSymbolicJacParamsLambFun,SDerivSymbolicJacConcLambFun):
    """
    Computes the jacobian matrix of the dSens that depends on parameters
    :param x_sp: symbols of the state variables
    :param sensitivity_sp: symbols of the senstivity equation
    :param param_sp: parameter symbols
    :param integration_params: dictionary of integration parameters
    :param SDerivSymbolicJacParamsLambFun: jacobian of spatial derivative wrt params
    :param SDerivSymbolicJacConcLambFun: jacobian of spatial derivative wrt state variables
    :return dSensSymJacSparseMatLamFun: sparse jacobian of dSens wrt the concentrations
    """

    # create state variables
    allVars = np.concatenate((x_sp,sensitivity_sp))
    diffeq_params = integration_params['diffeq_params'].copy()
    for key, value in param_sp.items():
        diffeq_params[key] = value

    #create RHS
    dSensSym = sp.Matrix(dSens(0,allVars,diffeq_params, integration_params,
          SDerivSymbolicJacParamsLambFun, SDerivSymbolicJacConcLambFun))
    dSensSymJac = dSensSym.jacobian(allVars)

    # generate jacobian
    dSensSymJacDenseMatLam = sp.lambdify((allVars,list(param_sp.values())),dSensSymJac)
    dSensSymJacSparseMatLamFun = lambda t,xs,param: sparse.csr_matrix(dSensSymJacDenseMatLam(xs,param))

    return dSensSymJacSparseMatLamFun


def main():
    # create dictionary of differential equation parameters symbols

    params = {'KmDhaTH': 0.77,
              'KmDhaTN': 0.03,
              'KiDhaTD': 0.23,
              'KiDhaTP': 7.4,
              'VfDhaT' : 86.2,
              'VfDhaB' : 10.,
              'KmDhaBG' : 0.01,
              'KiDhaBH' : 5.,
              'VfIcdE' : 1.,
              'KmIcdED' : 1.,
              'KmIcdEI' : 1.,
              'KiIcdEN' : 1.,
              'KiIcdEA' : 1.,
              'NInit':20,
              'DInit':20}

    # compute non-dimensional scaling
    dimscalings = initialize_dim_scaling(**params)

    param_sp = create_param_symbols('km','kc','GInit','IInit')





    # create a dictionary of integration parameters
    ngrid = 2
    nParams = len(param_sp)
    integration_params = initialize_integration_params(ngrid=ngrid)
    nVars = 5 * (2 + (integration_params['ngrid'])) + 2

    integration_params['nVars'] = nVars
    integration_params['nParams'] = nParams
    nSensitivityEqs = integration_params['nParams'] * integration_params['nVars']
    integration_params['nSensitivityEqs'] = nSensitivityEqs
    integration_params['n_compounds_cell'] = 5
    integration_params['diffeq_params'] = params
    integration_params['dimscalings'] = dimscalings

    # initialize parameter values
    x_sp, sensitivity_sp = create_state_symbols(nVars, nParams)
    integration_params['x_sp'] = x_sp
    integration_params['sensitivity_sp'] = sensitivity_sp
    integration_params['Sensitivity Params'] = param_sp

    # compute set up senstivity equations
    SDerivSymbolicJacParamsLambFun, SDerivSymbolicJacConcLambFun = compute_jacs(x_sp, param_sp,
                                                                                integration_params,
                                                                                diffeq_params=params)


    dSensSymJacSparseMatLamFunTXSParam = create_jac_sens_param(x_sp, sensitivity_sp, param_sp, integration_params,
                                                               SDerivSymbolicJacParamsLambFun, SDerivSymbolicJacConcLambFun)

    tol = 1e-5
    jac_F = lambda dict_vals: jac(dict_vals,integration_params, SDerivSymbolicJacParamsLambFun,
                             SDerivSymbolicJacConcLambFun, dSensSymJacSparseMatLamFunTXSParam, tol)

    # compute jacobian of P
    ind_P_ext = 3 # index of P_ext at steady state
    jac_f = lambda vars: jac_F(vars)[-ind_P_ext*nParams:-(ind_P_ext-1)*nParams]

    # set bounds

    bounds = np.array([[0.01,10],[0.01,10],[1,10],[1,10]])
    cost_mat = cost_fun_unif(jac_f,nParams,list(param_sp.keys()),bounds,maxN=200)
    w,v = np.linalg.eig(cost_mat)
    print('eigenvalues: ' )
    print(w)

    print('\n')
    print('eigenvectors: ')
    print(v)



if __name__ == '__main__':
    main()


