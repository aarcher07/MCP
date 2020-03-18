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
    :param params_sens_syms: params to compute sensitivity wrt
    :param diffeq_params: dictionary of all paramaters and their values
    :return SDerivSymbolicJacParamsLambFun: jacobian of spatial derivative wrt params
    :return SDerivSymbolicJacConcLambFun: jacobian of spatial derivative wrt concentration
    """

    # check if sensitivity to all params
    if kwargs['diffeq_params'] is None:
        diffeq_params = params_sens_dict
        params_sensitivity_sp = list(params_sens_dict.values())

    else:
        diffeq_params = kwargs['diffeq_params'].copy()
        params_sensitivity_sp = list(params_sens_dict.values())
        for key,value in params_sens_dict.items():
            diffeq_params[key] = value

    SDerivSymbolic = sp.Matrix(SDeriv(0,x_sp,integration_params,**diffeq_params))

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
    dxs.extend(SDeriv(0, x, integration_params, **diffeq_params))
    # get values of params
    param_vals = [diffeq_params[key] for key in params_sens_dict.keys()]
    # compute rhs of sensitivity equations
    SDerivSymbolicJacParamsMat = SDerivSymbolicJacParamsLambFun(t,x,param_vals)
    SDerivSymbolicJacConcMat = SDerivSymbolicJacConcLambFun(t,x,param_vals)
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
    # get parameters
    ngrid = 25
    integration_params = initialize_integration_params(ngrid=ngrid)
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
              'km' : 0.1,
              'kc': 1.,
              'GInit':10,
              'IInit':10,
              'NInit':20,
              'DInit':20}

    # create dictionary of integration parameters
    params_sens_dict = create_param_symbols('VfDhaT',
                                            'VfDhaB',
                                            'VfIcdE',
                                            'km',
                                            'kc',
                                            'GInit',
                                            'IInit')

    # compute non-dimensional scaling
    dimscalings = initialize_dim_scaling(**params)

    # store info about state variables
    n_compounds_cell = 5
    nVars = n_compounds_cell * (2 + (integration_params['ngrid'])) + 2
    integration_params['nVars'] = nVars

    # store info about parameters
    nParams = len(params_sens_dict)
    integration_params['nParams'] = nParams
    integration_params['Sensitivity Params'] = params_sens_dict
    nSensitivityEqs = integration_params['nParams']*integration_params['nVars']
    integration_params['nSensitivityEqs'] = nSensitivityEqs

    # initial conditions -- state variable
    n_compounds_cell = 5
    y0 = np.zeros((integration_params['ngrid'] + 2) * n_compounds_cell + 2)
    y0[-5] = params['GInit']/ dimscalings['G0']  # y0[-5] gives the initial state of the external substrate.
    y0[-1] = params['IInit'] / dimscalings['I0']  # y0[-1] gives the initial state of the external substrate.
    y0[0] = params['NInit'] / dimscalings['N0']  # y0[5] gives the initial state of the external substrate.
    y0[1] = params['DInit'] / dimscalings['D0']  # y0[6] gives the initial state of the external substrate.


    # initial conditions -- sensitivity equation
    sens0 = np.zeros(nSensitivityEqs)
    for i,param in enumerate(params_sens_dict):
        if param in ['GInit', 'IInit', 'NInit', 'DInit']:
            sens0[i:nSensitivityEqs:nParams] = 1/params[param]
    xs0 = np.concatenate([y0,sens0])

    # setup differential eq
    x_sp, sensitivity_sp = create_state_symbols(integration_params['nVars'], integration_params['nParams'])
    SDerivSymbolicJacParamsLambFun, SDerivSymbolicJacConcLambFun = compute_jacs(x_sp, params_sens_dict,integration_params, diffeq_params=params)
    dSensParams = lambda t,xs: dSens(t, xs, params, integration_params, SDerivSymbolicJacParamsLambFun,
                                     SDerivSymbolicJacConcLambFun)

    #create jacobian of dSensParams
    dSensSymJacSparseMatLamFun = create_jac_sens(x_sp, sensitivity_sp, params, integration_params,
                                                 SDerivSymbolicJacParamsLambFun, SDerivSymbolicJacConcLambFun)

    # solution params
    fintime = 3.e5
    tol = 1e-10
    #nsamples = 100
    #timeorig = np.linspace(0,fintime,nsamples)

    # terminal event
    event = lambda t,xs: np.absolute(dSensParams(t,xs)[nVars-1]) - tol
    event.terminal = True

    sol = solve_ivp(dSensParams,[0, fintime], xs0, method="BDF", jac = dSensSymJacSparseMatLamFun,
                    atol=1.0e-3, rtol=1.0e-3)

    # plot state variables solution
    print(sol.message)
    scalings = list(dimscalings.values())
    # external solution
    for i in range(5):
        plt.plot(sol.t,sol.y[(nVars-i-1):(nVars-i),:].T*scalings[-i-1])
    plt.legend(['G','H','P','A','I'],loc='upper right')
    plt.title('Plot of external concentrations')
    plt.xlabel('time')
    plt.ylabel('concentration')
    #plt.savefig('/Users/aarcher/PycharmProjects/MCP/WholeCell/plots/ScipyCode_MCPDynamics.png')
    plt.show()

    for i in range(7):
        plt.plot(sol.t,sol.y[i:(i+1),:].T*scalings[i])
    plt.legend(['N','D','G','H','P','A','I'],loc='upper right')
    plt.title('Plot of internal concentrations')
    plt.xlabel('time')
    plt.ylabel('concentration')
    #filename = "VfDhaB_"+str(VfDhaB)+"_KmDhaBG_" + str(KmDhaBG) + "_KiDhaBH_" + str(KiDhaBH) + "_VfIcdE_"  + str(VfIcdE) + "_KmIcdEA_" + str(KmIcdEA) + "_KmIcdEN_" + str(KmIcdEN) + "_KiIcdED_" + str(KiIcdED) + "_KiIcdEI_" + str(KiIcdEI) + "_GInit_" + str(GInit) + "_NInit_" + "_GInit_" + str(GInit) + "_NInit_" + str(NInit) + "_DInit_" + str(DInit) + "_AInit_" + str(AInit)
    #plt.savefig('/Users/aarcher/PycharmProjects/MCP/WholeCell/plots/ScipyCode_MCPDynamics.png')
    plt.show()

    # plot sensitivity variable solutions for external variables
    namesExt = ['I','A','P','H','G']
    sens_vars_names = [r'$V_f^{DhaT}$', r'$V_f^{DhaB}$', r'$V_f^{IcdE}$', r'$k_m$', r'$k_c$', r'$G_0$', r'$I_0$']

    #
    # for i in range(len(namesExt)):
    #     figure, axes = plt.subplots(nrows=4, ncols=2,figsize=(20,20))
    #     if i == 0:
    #         yub = np.max(sol.y[-(nParams):,:])
    #         lub = np.min(sol.y[-(nParams):,:])
    #         for j in range(nParams):
    #             axes[j//2,j % 2].plot(sol.t,sol.y[-(nParams-j),:].T)
    #             axes[j//2,j % 2].set_xlabel('time')
    #             axes[j // 2, j % 2].set_ylabel(r'$\partial ' + namesExt[i]+'/\partial ' + sens_vars_names[j][1:])
    #             axes[j // 2, j % 2].set_title(sens_vars_names[j])
    #             axes[j // 2, j % 2].set_ylim([lub,yub])
    #     else:
    #         yub = np.max(sol.y[-(i+1)*nParams:-i*nParams,:])
    #         lub = np.min(sol.y[-(i+1)*nParams:-i*nParams,:])
    #         for j in range(nParams):
    #             axes[j//2,j % 2].plot(sol.t,sol.y[-((i+1)*nParams-j),:].T)
    #             axes[j//2,j % 2].set_xlabel('time')
    #             axes[j // 2, j % 2].set_ylabel(r'$\partial ' + namesExt[i]+'/\partial ' + sens_vars_names[j][1:])
    #             axes[j // 2, j % 2].set_title(sens_vars_names[j])
    #             axes[j // 2, j % 2].set_ylim([1.1*lub,1.1*yub])
    #
    #     figure.suptitle(r'Sensitivity, $\partial ' + namesExt[i]+'/\partial p_i$, of the external concentration of '
    #                     + namesExt[i] + ' wrt $p_i$')
    #     plt.savefig('/Users/aarcher/PycharmProjects/MCP/WholeCell/plots/Perm_SensitivityExternal_'+ namesExt[i] + '_ngrid' + str(ngrid) +'.png')
    #     plt.show()
    #
    # # plot sensitivity variable solutions for MCP variables
    # namesMCP = ['N','D','G','H','P','A','I']
    # for i in range(0,len(namesMCP)):
    #     figure, axes = plt.subplots(nrows=4, ncols=2, figsize=(20,20))
    #     yub = np.max(sol.y[(nVars + i*nParams):(nVars + (i+1)*nParams), :])
    #     lub = np.min(sol.y[(nVars + i*nParams):(nVars + (i+1)*nParams), :])
    #     for j in range(nParams):
    #         axes[j // 2, j % 2].plot(sol.t, sol.y[nVars + i*nParams+j, :].T)
    #         axes[j // 2, j % 2].set_xlabel('time')
    #         axes[j // 2, j % 2].set_ylabel(r'$\partial ' + namesMCP[i] + '/\partial ' + sens_vars_names[j][1:])
    #         axes[j // 2, j % 2].set_title(r'$p_i = ' + sens_vars_names[j][1:])
    #         axes[j // 2, j % 2].set_ylim([1.1*lub, 1.1*yub])
    #
    #     figure.suptitle(r'Sensitivity, $\partial ' + namesMCP[i]+'/\partial p_i$, of the MCP concentration of '
    #                     + namesMCP[i] + ' wrt $p_i$')
    #     plt.savefig('/Users/aarcher/PycharmProjects/MCP/WholeCell/plots/Perm_SensitivityInternal_'+ namesMCP[i] + '_ngrid' + str(ngrid) +'.png')
    #     plt.show()


    #check conservation of mass
    scalings = list(dimscalings.values())

    volcell = 4*np.pi*(integration_params['Rc']**3)/3
    volmcp = 4 * np.pi * (integration_params['Rm'] ** 3) / 3
    volratio = integration_params['Vratio']

    ext_masses_org = np.multiply(y0[-5:],scalings[2:]) * (volcell/volratio)
    cell_masses_org = np.multiply(y0[13:18],scalings[2:]) * (volcell - volmcp)
    mcp_masses_org = np.multiply(y0[:7],scalings) * volmcp

    ext_masses_fin = np.multiply(sol.y[(nVars-5):nVars, -1],scalings[2:]) * (volcell/volratio)
    cell_masses_fin = np.multiply(sol.y[13:18,-1],scalings[2:]) * (volcell - volmcp)
    mcp_masses_fin = np.multiply(sol.y[:7, -1],scalings) * volmcp
    print(ext_masses_org.sum() + cell_masses_org.sum() + mcp_masses_org.sum())
    print(ext_masses_fin.sum() + cell_masses_fin.sum() + mcp_masses_fin.sum())


if __name__ == '__main__':
    main()


