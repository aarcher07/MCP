import numpy as np
from scipy.integrate import solve_ivp
from scipy.constants import *
import sys
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib
import math
import sympy as sp
import scipy.sparse as sparse
import time
from numpy.linalg import LinAlgError
from Whole_Cell_Engineered_System_IcdE import *

def create_param_symbols(*args):
    """
    :param args: list of parameters to analyze using sensitivity analysis
    :return: dictionary of sympy symbols
    """

    params_sens_dict = {arg:sp.symbols(arg) for arg in args}

    return params_sens_dict


def create_state_symbols(nVars,nParams):
    """
    creates a list of sympy symbols for state vectors and derivative of each state vector
    wrt to parameters from create_param_symbols
    :param ngrid: size of grid
    :param nParams: number of parameters
    :return x_sp: list of state variables
    :return sensitivity_sp: list of sensitivity variables
    """

    nSensitivityEqs = nVars * nParams

    #state variables
    x_sp = np.array(sp.symbols('x0:' + str(nVars)))

    #sensitivity variables
    sensitivity_sp = np.array(list(sp.symbols('s0:' + str(nSensitivityEqs))))

    return [x_sp,sensitivity_sp]


def compute_jacs(x_sp,params_sens_dict,integration_params,**kwargs):
    """
    Computes the jacobian of the spatial derivative wrt concentrations (state variables)
    and parameters from create_param_symbols

    :param x_sp: state variables
    :param params_sens_dict: params to compute sensitivity wrt
    :param diffeq_params: dictionary of all paramaters and their values
    :return SDerivSymbolicJacParamsLambFun: jacobian of spatial derivative wrt params
    :return SDerivSymbolicJacConcLambFun: jacobian of spatial derivative wrt concentration
    """

    # check if sensitivity to all params
    if kwargs['diff_params'] is None:
        param_list = list(params_sens_dict.values())
        params_sensitivity_sp = list(params_sens_dict.values())

    else:
        diffeq_params = kwargs['diff_params']

        params_sensitivity = list(params_sens_dict.keys())

        params_sensitivity_sp = list(params_sens_dict.values())
        param_list = [params_sens_dict[key] if key in params_sensitivity else value
                      for key,value in diffeq_params.items()]


    SDerivSymbolic = sp.Matrix(SDeriv(0,x_sp,param_list,integration_params))

    # derivative of rhs wrt params
    SDerivSymbolicJacParams = SDerivSymbolic.jacobian(params_sensitivity_sp)
    SDerivSymbolicJacParamsLamb = sp.lambdify((x_sp,params_sensitivity_sp), SDerivSymbolicJacParams,'numpy')
    SDerivSymbolicJacParamsLambFun = lambda t,x,params: SDerivSymbolicJacParamsLamb(x,params)

    # derivative of rhs wrt Conc
    SDerivSymbolicJacConc = SDerivSymbolic.jacobian(x_sp)
    SDerivSymbolicJacConcLamb = sp.lambdify((x_sp,params_sensitivity_sp),SDerivSymbolicJacConc,'numpy')
    SDerivSymbolicJacConcLambFun = lambda t,x,params: SDerivSymbolicJacConcLamb(x,params)

    return [SDerivSymbolicJacParamsLambFun,SDerivSymbolicJacConcLambFun]


def dSens(t,xs,diffeq_params, integration_params,
          SDerivSymbolicJacParamsLambFun, SDerivSymbolicJacConcLambFun):
    """
    Compute RHS of the sensitivity equation

    :param t: time
    :param xs: state variables and sensitivity variables
    :param diffeq_params: dictionary of differential equation parameters
    :param integration_params: dictionary of differential equation parameters
    :param SDerivSymbolicJacParamsLambFun:
    :param SDerivSymbolicJacConcLambFun:
    :return dxs: RHS of the sensitivity system
    """
    # initialization
    nVars = integration_params['nVars']
    x = xs[:nVars]
    s = xs[nVars:]
    dxs = []
    params_sens_dict = integration_params['Sensitivity Params']
    nParams = integration_params['nParams']
    nSensitivityEqs = integration_params['nSensitivityEqs']

    # get rhs of x
    dxs.extend(SDeriv(0, x, list(diffeq_params.values()), integration_params))

    # get values of sensitivity params
    param_sens_vals = [diffeq_params[key] for key in params_sens_dict.keys()]

    # compute rhs of sensitivity equations
    SDerivSymbolicJacParamsMat = SDerivSymbolicJacParamsLambFun(t,x,param_sens_vals)
    SDerivSymbolicJacConcMat = SDerivSymbolicJacConcLambFun(t,x,param_sens_vals)
    for i in range(nVars):
        for j in range(nParams):
            dxs.append(np.dot(SDerivSymbolicJacConcMat[i,:], s[range(j,nSensitivityEqs,nParams)])
                       + SDerivSymbolicJacParamsMat[i,j])
    return dxs



def create_jac_sens(x_sp,sensitivity_sp,diffeq_params, integration_params,
                    SDerivSymbolicJacParamsLambFun,SDerivSymbolicJacConcLambFun):
    """
    Computes the jacobian matrix of the dSens
    :param x_sp: symbols of the state variables
    :param sensitivity_sp: symbols of the senstivity equation
    :param diffeq_params: dictionary of the parameter values
    :param integration_params: dictionary of integration parameters
    :param SDerivSymbolicJacParamsLambFun: jacobian of spatial derivative wrt params
    :param SDerivSymbolicJacConcLambFun: jacobian of spatial derivative wrt state variables
    :return dSensSymJacSparseMatLamFun: sparse jacobian of dSens wrt the concentrations
    """

    # create state variables
    allVars = np.concatenate((x_sp,sensitivity_sp))

    #create RHS
    dSensSym = sp.Matrix(dSens(0,allVars,diffeq_params, integration_params,
          SDerivSymbolicJacParamsLambFun, SDerivSymbolicJacConcLambFun))
    dSensSymJac = dSensSym.jacobian(allVars)

    # generate jacobian
    dSensSymJacDenseMatLam = sp.lambdify(allVars,dSensSymJac)
    dSensSymJacSparseMatLamFun = lambda t,xs: sparse.csr_matrix(dSensSymJacDenseMatLam(*xs))

    return dSensSymJacSparseMatLamFun


def main():

    #initialize differential equation variables
    [diffeq_params, dimscalings] = initialize(VfDhaB=40.0, KmDhaBG=1.0, KiDhaBH=1., VfIcdE=30., KmIcdED=1.,
                                           KmIcdEI=1., KiIcdEN=1., KiIcdEA=1.0, perm_mcp=1.0, perm_cell=1.0,
                                           GInit=100, IInit=100, NInit=50, DInit=50)

    # create dictionary of integration parameters
    params_sens_dict = create_param_symbols('alpha0','alpha1','kc', 'km', 'GInit', 'IInit')
    ngrid = 25
    integration_params = initialize_integration_params(ngrid=ngrid)
    nVars = 5 * (2 + (integration_params['ngrid'])) + 2
    integration_params['nVars'] = nVars

    nParams = len(params_sens_dict)
    integration_params['nParams'] = nParams
    integration_params['Sensitivity Params'] = params_sens_dict
    nSensitivityEqs = integration_params['nParams']*integration_params['nVars']
    integration_params['nSensitivityEqs'] = nSensitivityEqs

    # initial conditions -- state variable
    n_compounds_cell = 5
    y0 = np.zeros((integration_params['ngrid'] + 2) * n_compounds_cell + 2)
    y0[-5] = diffeq_params['GInit']/ dimscalings['G0']  # y0[-5] gives the initial state of the external substrate.
    y0[-1] = diffeq_params['IInit'] / dimscalings['I0']  # y0[-1] gives the initial state of the external substrate.
    y0[0] = diffeq_params['NInit'] / dimscalings['N0']  # y0[5] gives the initial state of the external substrate.
    y0[1] = diffeq_params['DInit'] / dimscalings['D0']  # y0[6] gives the initial state of the external substrate.


    # initial conditions -- sensitivity equation
    sens0 = np.zeros(nSensitivityEqs)
    for i,param in enumerate(params_sens_dict):
        if param in ['GInit', 'IInit', 'NInit', 'PInit']:
            sens0[i:nSensitivityEqs:nParams] = 1
    xs0 = np.concatenate([y0,sens0])

    # setup differential eq
    x_sp, sensitivity_sp = create_state_symbols(integration_params['nVars'], integration_params['nParams'])
    SDerivSymbolicJacParamsLambFun, SDerivSymbolicJacConcLambFun = compute_jacs(x_sp, params_sens_dict,integration_params, diff_params=diffeq_params)
    dSensParams = lambda t,xs: dSens(t, xs, diffeq_params, integration_params, SDerivSymbolicJacParamsLambFun,
                                     SDerivSymbolicJacConcLambFun)

    #create jacobian of dSensParams
    dSensSymJacSparseMatLamFun = create_jac_sens(x_sp, sensitivity_sp, diffeq_params, integration_params,
                                                 SDerivSymbolicJacParamsLambFun, SDerivSymbolicJacConcLambFun)


    # solution params
    fintime = 2.e8
    tol = 1e-5
    nsamples = 100
    timeorig = np.linspace(0,fintime,nsamples)

    # terminal event
    event = lambda t,xs: np.absolute(dSensParams(t,xs)[nVars-1]) - tol
    event.terminal = True

    sol = solve_ivp(dSensParams,[0, fintime], xs0, method="BDF", jac = dSensSymJacSparseMatLamFun,
                    t_eval=timeorig, atol=1.0e-6, rtol=1.0e-6)

    # plot state variables solution
    print(sol.message)
    plt.plot(sol.t,sol.y[(nVars-5):nVars,:].T)
    plt.legend(['G','H','P','A','I'],loc='upper right')
    #plt.savefig('/Users/aarcher/PycharmProjects/MCP/WholeCell/plots/ScipyCode_MCPDynamics.png')
    plt.show()

    plt.plot(sol.t,sol.y[:7,:].T)
    plt.legend(['N','D','G','H','P','A','I'],loc='upper right')
    #filename = "VfDhaB_"+str(VfDhaB)+"_KmDhaBG_" + str(KmDhaBG) + "_KiDhaBH_" + str(KiDhaBH) + "_VfIcdE_"  + str(VfIcdE) + "_KmIcdEA_" + str(KmIcdEA) + "_KmIcdEN_" + str(KmIcdEN) + "_KiIcdED_" + str(KiIcdED) + "_KiIcdEI_" + str(KiIcdEI) + "_GInit_" + str(GInit) + "_NInit_" + "_GInit_" + str(GInit) + "_NInit_" + str(NInit) + "_DInit_" + str(DInit) + "_AInit_" + str(AInit)
    #plt.savefig('/Users/aarcher/PycharmProjects/MCP/WholeCell/plots/ScipyCode_MCPDynamics.png')
    plt.show()

    # plot sensitivity variable solutions for external variables
    namesExt = ['G','H','P','A','I']
    for i in range(0,len(namesExt)):
        if i == 0:
            plt.plot(sol.t,sol.y[-nParams:,:].T)
        else:
            plt.plot(sol.t, sol.y[-(i+1)*nParams:-(i*nParams), :].T)
        plt.title(r'Sensitivity, $\partial ' + namesExt[i]+'/\partial p_i$, of the external concentration of '
                  + namesExt[i] + ' wrt $p_i =  \alpha_0, \alpha_1, k_m, k_c, G_0, I_0$')
        plt.xlabel('time')
        plt.ylabel(r'$\partial ' + namesExt[i]+'/\partial p_i$')
        plt.legend([r'$\alpha_0$',r'$\alpha_1$', r'$k_m$',r'$k_c$',r'$G_0$',r'$I_0$'],loc='upper right')
        #plt.savefig('/Users/aarcher/PycharmProjects/MCP/WholeCell/plots/Perm_SensitivityExternal_'+ namesExt[i]
        #            + '_ngrid' + str(ngrid) +'.png')
        plt.show()

    # plot sensitivity variable solutions for MCP variables
    namesMCP = ['N','D','G','H','P','A','I']
    for i in range(0,len(namesMCP)):
        plt.plot(sol.t,sol.y[(nVars + i*nParams):(nVars+(i+1)*nParams),:].T)
        plt.title(r'Sensitivity, $\partial ' + namesMCP[i]+'/\partial p_i$, of the internal concentration of '
                  + namesMCP[i] +' wrt $p_i = \alpha_0, \alpha_1, k_m, k_c, G_0, I_0$')
        plt.xlabel('time')
        plt.ylabel(r'$\partial ' + namesMCP[i]+'/\partial p_i$')
        plt.legend([r'$\alpha_0$',r'$\alpha_1$', r'$k_m$',r'$k_c$',r'$G_0$',r'$I_0$'],loc='upper right')
        #plt.savefig('/Users/aarcher/PycharmProjects/MCP/WholeCell/plots/Perm_SensitivityInternal_'+ namesMCP[i]
        #            + '_ngrid' + str(ngrid) +'.png')
        plt.show()


if __name__ == '__main__':
    main()


