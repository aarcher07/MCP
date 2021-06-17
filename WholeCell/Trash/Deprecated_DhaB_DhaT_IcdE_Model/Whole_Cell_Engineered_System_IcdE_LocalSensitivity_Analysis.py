'''
Local Sensitivity Analysis of IcdE Model with functions.
This module gives the user control over the parameters for 
which they would like to do sensitivity analysis.

Programme written by aarcher07

Editing History:
- 26/10/20
'''

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
import math

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

    SDerivSymbolic = sp.Matrix(SDeriv(0,x_sp,integration_params,diffeq_params))

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
    dxs.extend(SDeriv(0, x, integration_params, diffeq_params))
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


def main(nsamples = 100):

    #initialize differential equation variables
    # get parameters
    ngrid = 100
    integration_params = initialize_integration_params(ngrid=ngrid)
    params = {'KmDhaTH': 0.77,
              'KmDhaTN': 0.03,
              'KiDhaTD': 0.23,
              'KiDhaTP': 7.4,
              'VfDhaT' : 86.2,
              'VfDhaB' : 100.,
              'KmDhaBG' : 0.01,
              'KiDhaBH' : 5.,
              'VfIcdE' : 30.,
              'KmIcdED' : 0.1,
              'KmIcdEI' : 0.02,
              'KiIcdEN' : 3.,
              'KiIcdEA' : 10.,
              'km' : 10,
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
                                            'kc')

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
    fintime = 1e8
    mintime = 1
    tol = 1e-10
    timeorig = np.logspace(np.log10(mintime),np.log10(fintime),nsamples)

    # terminal event
    event = lambda t,xs: np.absolute(dSensParams(t,xs)[nVars-1]) - tol
    event.terminal = True
    starttime = time.time()
    sol = solve_ivp(dSensParams,[0, fintime], xs0, method="BDF", jac = dSensSymJacSparseMatLamFun, t_eval=timeorig,
                    atol=1.0e-2, rtol=1.0e-2)
    endtime = time.time()

    print('code time: ' + str(endtime-starttime))
    # plot state variables solution
    print(sol.message)

    #create grid
    M_cell = (integration_params['Rc'] / integration_params['Rm']) ** 3  # why?
    M_mcp = 1.
    Mgrid = np.linspace(M_mcp, M_cell, integration_params['ngrid']-1)
    DeltaM = np.divide((M_cell - M_mcp), (integration_params['ngrid']-1))

    # rescaling
    scalings = list(dimscalings.values())
    volcell = 4 * np.pi * (integration_params['Rc'] ** 3) / 3
    volmcp = 4 * np.pi * (integration_params['Rm'] ** 3) / 3
    volratio = integration_params['Vratio']

    # rescale the solutions
    numeachcompound = 2 + integration_params['ngrid']
    ncompounds = 5


    xvalslogtimeticks = list(range(int(np.log10(fintime))+1))
    xtexlogtimeticks = [r'$10^{' + str(i) + '}$' for i in range(int(np.log10(fintime))+1)]
    sol.y[:2, :] = (np.multiply(sol.y[:2, :].T, scalings[:2])).T
    for i in range(numeachcompound):
        j = range(2 + i * ncompounds, 2 + (i + 1) * ncompounds)
        sol.y[j, :] = (np.multiply(sol.y[j, :].T, scalings[2:])).T

    #plot sensitivity variable solutions for external variables
    namesExt = ['I','A','P','H','G']
    sens_vars_names = [r'$V_f^{DhaT}$', r'$V_f^{DhaB}$', r'$V_f^{IcdE}$', r'$k_m$', r'$k_c$']#, r'$G_0$', r'$I_0$']


    for i in range(len(namesExt)):
        figure, axes = plt.subplots(nrows=int(math.ceil(len(params_sens_dict)/2)), ncols=2, figsize=(10,15))
        if i == 0:
            soly = sol.y[-(nParams):,:]*scalings[-1]
            yub = np.max(soly) +1
            lub = np.min(soly)  - 1
            for j in range(nParams):
                axes[j//2,j % 2].plot(np.log10(timeorig),soly[j,:].T)
                axes[j//2,j % 2].set_xlabel('log(time)')
                axes[j // 2, j % 2].set_ylabel(r'$\log\partial ' + namesExt[i]+'/\partial ' + sens_vars_names[j][1:])
                axes[j // 2, j % 2].set_title(sens_vars_names[j])
                axes[j // 2, j % 2].set_ylim([lub, yub])
                axes[j // 2, j % 2].set_xticks(xvalslogtimeticks, xtexlogtimeticks)
        else:
            soly = sol.y[-(i+1)*nParams:-i*nParams, :]*scalings[-i - 1]
            yub = np.max(soly) + 1
            lub = np.min(soly) - 1
            for j in range(nParams):
                axes[j//2,j % 2].plot(np.log10(timeorig),soly[j,:].T)
                axes[j//2,j % 2].set_xlabel('log(time)')
                axes[j // 2, j % 2].set_ylabel(r'$\log\partial ' + namesExt[i]+'/\partial ' + sens_vars_names[j][1:])
                axes[j // 2, j % 2].set_title(sens_vars_names[j])
                axes[j // 2, j % 2].set_ylim([lub, yub])
                axes[j // 2, j % 2].set_xticks(xvalslogtimeticks, xtexlogtimeticks)

        figure.suptitle(r'Sensitivity, $\partial ' + namesExt[i]+'/\partial p_i$, of the external concentration of '
                        + namesExt[i] + ' wrt $p_i$', y = 0.92)
        plt.savefig('/Users/aarcher/PycharmProjects/MCP/WholeCell/plots/Perm_SensitivityExternal_'+ namesExt[i] + '_ngrid' + str(ngrid) +'.png',
                    bbox_inches='tight')
        plt.show()

    # plot sensitivity variable solutions for MCP variables
    namesMCP = ['N','D','G','H','P','A','I']
    for i in range(0,len(namesMCP)):
        figure, axes = plt.subplots(nrows=int(math.ceil(len(params_sens_dict)/2)), ncols=2, figsize=(10,15))
        soly = sol.y[(nVars + i*nParams):(nVars + (i+1)*nParams), :]*scalings[i]
        yub = np.max(soly)  + 1
        lub = np.min(soly) - 1
        for j in range(nParams):
            axes[j // 2, j % 2].plot(np.log(timeorig), soly[j,:].T)
            axes[j // 2, j % 2].set_xlabel('log(time)')
            axes[j // 2, j % 2].set_ylabel(r'$\log\partial ' + namesMCP[i] + '/\partial ' + sens_vars_names[j][1:])
            axes[j // 2, j % 2].set_title(r'$p_i = ' + sens_vars_names[j][1:])
            axes[j // 2, j % 2].set_ylim([lub, yub])
            axes[j // 2, j % 2].set_xticks(xvalslogtimeticks, xtexlogtimeticks)

        figure.suptitle(r'Sensitivity, $\partial ' + namesMCP[i]+'/\partial p_i$, of the MCP concentration of '
                        + namesMCP[i] + ' wrt $p_i$', y = 0.92)
        plt.savefig('/Users/aarcher/PycharmProjects/MCP/WholeCell/plots/Perm_SensitivityInternal_'+ namesMCP[i] + '_ngrid' + str(ngrid) +'.png',
                    bbox_inches='tight')
        plt.show()


    #check mass balance
    volcell = 4*np.pi*(integration_params['Rc']**3)/3
    volmcp = 4 * np.pi * (integration_params['Rm'] ** 3) / 3
    volratio = integration_params['Vratio']

    scalings = list(dimscalings.values())
    ext_masses_org = y0[-5:]* (volcell/volratio)
    cell_masses_org = y0[12:17] * (volcell - volmcp)
    mcp_masses_org = y0[:7] * volmcp


    ext_masses_fin = sol.y[-5:, -1] * (volcell/volratio)
    cell_masses_fin = sol.y[12:17,-1] * (volcell - volmcp)
    mcp_masses_fin = sol.y[:7, -1] * volmcp
    print(ext_masses_org.sum() + cell_masses_org.sum() + mcp_masses_org.sum())
    print(ext_masses_fin.sum() + cell_masses_fin.sum() + mcp_masses_fin.sum())
    print((sol.y[-5:, -1]).sum()*(volcell/volratio+volmcp+volcell))


if __name__ == '__main__':
    main()


