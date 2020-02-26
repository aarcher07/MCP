import numpy as np
from scipy.integrate import solve_ivp

import matplotlib.pyplot as plt
import warnings
import sympy as sp
import scipy.sparse as sparse
from Whole_Cell_Engineered_System_IcdE import *
from Whole_Cell_Engineered_System_IcdE_LocalSensitivityAnalysis import *


def cost_fun_unif(jac,M,bounds,maxN=1000):
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
    while 1:
        if N > maxN:
            warnings.warn("The cost matrix maybe not be full rank. The maximum N needs to be increased.")
            break

        if M > np.linalg.matrix_rank((1/N)*C):
            break
        x = (bounds[:,1] - bounds[:,0])*np.random.uniform(0,1,size=M) + bounds[:,0]
        j = jac(x)
        C += np.dot(j,j.T)

    return (1/N)*C

def jac(vars,integration_params, SDerivSymbolicJacParamsLambFun, SDerivSymbolicJacConcLambFun, dSensSymJacSparseMatLamFun,
        tol, fintime = 2.e8):

    # get integration parameters
    param_list = integration_params['param_list']
    params_sens_dict = integration_params['Sensitivity Params']
    nSensitivityEqs = integration_params['nSensitivityEqs']
    nParams = integration_params['nParams']
    nVars = integration_params['nParams']

    # put differential equation parameters in the dictionary format
    diffeq_params = {param: arg for param,arg in zip(param_list[:18],vars[:18])}
    dimscalings = {param: arg for param, arg in zip(param_list[18:], vars[18:])}


    # initial conditions

    y0 = np.zeros(nVars)     # initial conditions -- state variable
    y0[-5] = diffeq_params['GInit'] / dimscalings['G0']  # y0[-5] gives the initial state of the external substrate.
    y0[-1] = diffeq_params['IInit'] / dimscalings['I0']  # y0[-1] gives the initial state of the external substrate.
    y0[0] = diffeq_params['NInit'] / dimscalings['N0']  # y0[5] gives the initial state of the external substrate.
    y0[1] = diffeq_params['DInit'] / dimscalings['D0']  # y0[6] gives the initial state of the external substrate.

    sens0 = np.zeros(nSensitivityEqs)  #initial conditions -- sensitivity equation
    for i,param in enumerate(param_list):
        if param in ['GInit', 'IInit', 'NInit', 'PInit']:
            sens0[i:nSensitivityEqs:nParams] = 1
    xs0 = np.concatenate([y0,sens0])

    # solve differential equation

    dSensParams = lambda t,xs: dSens(t, xs, diffeq_params, integration_params, SDerivSymbolicJacParamsLambFun,
                                     SDerivSymbolicJacConcLambFun) #senstivity equation
    dSensSymJacSparseMatLamFunTXS = lambda t,xs: dSensSymJacSparseMatLamFun(xs,vars) #jacobian


    event = lambda t,xs: np.absolute(dSensParams(t,xs)[nVars-1]) - tol #terminal event
    event.terminal = True

    sol = solve_ivp(dSensParams,[0, fintime], xs0, method="BDF", jac = dSensSymJacSparseMatLamFunTXS,
                    atol=1.0e-6, rtol=1.0e-6)

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

    #create RHS
    dSensSym = sp.Matrix(dSens(0,allVars,param_sp, integration_params,
          SDerivSymbolicJacParamsLambFun, SDerivSymbolicJacConcLambFun))
    dSensSymJac = dSensSym.jacobian(allVars)

    # generate jacobian
    dSensSymJacDenseMatLam = sp.lambdify((allVars,param_sp),dSensSymJac)
    dSensSymJacSparseMatLamFun = lambda t,xs,param: sparse.csr_matrix(dSensSymJacDenseMatLam(*xs,*param))

    return dSensSymJacSparseMatLamFun


def main():
    # create a dictionary of integration parameters
    ngrid = 25
    nParams = 22

    integration_params = initialize_integration_params(ngrid=ngrid)
    nVars = 5 * (2 + (integration_params['ngrid'])) + 2
    integration_params['nVars'] = nVars

    integration_params['nParams'] = nParams


    nSensitivityEqs = integration_params['nParams'] * integration_params['nVars']
    integration_params['nSensitivityEqs'] = nSensitivityEqs


    # create dictionary of differential equation parameters symbols
    param_list = ['alpha' + str(i) for i in range(10)]
    param_list.extend(['beta' + str(i) for i in range(4)])
    param_list.extend(['gamma' + str(i) for i in range(4)])
    param_list.extend(['kc', 'km', 'GInit', 'IInit', 'NInit', 'DInit'])
    integration_params['param_list'] = param_list
    params_sens_dict = create_param_symbols(param_list)
    param_sp = list(params_sens_dict.values())
    integration_params['params_sens_dict'] = params_sens_dict
    integration_params['n_compounds_cell'] = 5

    # compute jacobians SDev wrt params and state variables
    x_sp, sensitivity_sp = create_state_symbols(nVars, nParams)
    integration_params['x_sp'] = x_sp
    integration_params['sensitivity_sp'] = sensitivity_sp
    SDerivSymbolicJacParamsLambFun, SDerivSymbolicJacConcLambFun = compute_jacs(x_sp, params_sens_dict,
                                                                                integration_params)
    dSensSymJacSparseMatLamFunTXSParam = create_jac_sens_param(x_sp, sensitivity_sp, param_sp, integration_params,
                                                               SDerivSymbolicJacParamsLambFun, SDerivSymbolicJacConcLambFun)

    tol = 1e-5
    jac_F = lambda vars: jac(vars,integration_params, SDerivSymbolicJacParamsLambFun, SDerivSymbolicJacConcLambFun,
                            dSensSymJacSparseMatLamFunTXSParam, tol)
    k = 7 + 5*ngrid + 3 + - 1 # index of P_ext at steady state, -1 since python indexes at 0
    # get jacobian of P ONLY  -- store in jacf
    # compute cost_fun = cost_fun_unif(jacf, M, bounds, maxN=1000)





