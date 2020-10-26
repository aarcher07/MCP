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
import Whole_Cell_Engineered_System_WellMixed_MCPs_DhaB_DhaT_Taylor
eps = 10**(-5)

# override ComputeEnzymeConcentrations in the original documentation
def _new_ComputeEnzymeConcentrations(ratio, dPacking):
    '''
    Computes the enzyme concentrations from the relative enzyme expressions and
    the packing efficiency
    '''
    rDhaB = 8/2.
    rDhaT = 5/2.
    NDhaT= sp.Symbol('NDhaT')
    AvogadroConstant = constants.Avogadro
    rMCP =140/2.
    Vol = lambda r: 4*np.pi*(r**3)/3;  
    SigmaDhaTExpression = lambda NDhaT: (NDhaT*Vol(rDhaT) + ratio*NDhaT*Vol(rDhaB))/Vol(rMCP) - dPacking
    SigmaDhaT = sp.solve(SigmaDhaTExpression(NDhaT),NDhaT)[0]/(AvogadroConstant*Vol(rMCP*(1e-9)))
    SigmaDhaB = ratio*SigmaDhaT
    return [SigmaDhaB, SigmaDhaT]

Whole_Cell_Engineered_System_WellMixed_MCPs_DhaB_DhaT_Taylor.ComputeEnzymeConcentrations = _new_ComputeEnzymeConcentrations

from Whole_Cell_Engineered_System_WellMixed_MCPs_DhaB_DhaT_Taylor import *

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


def compute_jacs(x_sp,params_sens_dict,integration_params,dS = SDeriv, **kwargs):
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
    if kwargs['diffeq_params'] is None:
        diffeq_params = params_sens_dict.copy()
        params_sensitivity_sp = list(params_sens_dict.values())
        # # log transform
        # diffeq_params = {key:10**(paramsyms) for key,paramsyms in params_sens_dict.items()}
        # params_sensitivity_sp = list(params_sens_dict.values())

    else:
        diffeq_params = kwargs['diffeq_params'].copy()
        params_sensitivity_sp = list(params_sens_dict.values())
        for key,value in params_sens_dict.items():
            # log transform variable
            # diffeq_params[key] = 10**(value)
            diffeq_params[key] = value


    SDerivSymbolic = sp.Matrix(dS(0,x_sp,integration_params,diffeq_params))

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
          SDerivSymbolicJacParamsLambFun, SDerivSymbolicJacConcLambFun,dS = SDeriv):
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
    # log transform variables
    # diffeq_params_copied = diffeq_params.copy()
    # for key in diffeq_params_copied.keys():
    #     diffeq_params_copied[key] = 10**(diffeq_params_copied[key])

    # get rhs of x
    dxs.extend(dS(0, x, integration_params, diffeq_params))
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


def SDerivLog10Param(*args):
    diffeq_params = args[3]
    diffeq_params_log10 = {key: 10**value for key,value in diffeq_params.items()}
    return SDeriv(args[0],args[1],args[2],diffeq_params_log10)

def SDerivLog2Param(*args):
    diffeq_params = args[3]
    diffeq_params_log2 = {key: 2**value for key,value in diffeq_params.items()}
    return SDeriv(args[0],args[1],args[2],diffeq_params_log2)

def SDerivLog10StateVariableParam(*args):
    x = args[1]
    diffeq_params = args[3]
    x_shiftedlog10 = [10**xi - eps for xi in x]
    diffeq_params_log10 = {key: 10**value for key,value in diffeq_params.items()}
    return SDeriv(args[0],x_shiftedlog10,args[2],diffeq_params_log10)

def create_jac_sens(x_sp,sensitivity_sp,diffeq_params, integration_params,
                    SDerivSymbolicJacParamsLambFun,SDerivSymbolicJacConcLambFun, dS = SDeriv):
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
          SDerivSymbolicJacParamsLambFun, SDerivSymbolicJacConcLambFun, dS = dS))
    dSensSymJac = dSensSym.jacobian(allVars)

    # generate jacobian
    dSensSymJacDenseMatLam = sp.lambdify(allVars,dSensSymJac)
    dSensSymJacSparseMatLamFun = lambda t,xs: sparse.csr_matrix(dSensSymJacDenseMatLam(*xs))

    return dSensSymJacSparseMatLamFun


def main(nsamples = 500):

    external_volume =  9e-6
    NcellsPerMCubed = 8e14 # 7e13-8e14 cells per m^3
    Ncells = external_volume*NcellsPerMCubed   
    mintime = 10**(-15)
    secstohrs = 60*60
    fintime = 72*60*60
    integration_params = initialize_integration_params(external_volume = external_volume, 
                                                       Ncells =Ncells,cellular_geometry="rod",
                                                       Rc = 0.375e-6, Lc = 2.47e-6)
    params = {'KmDhaTH': 0.77, # mM
          'KmDhaTN': 0.03, # mM
          'kcatfDhaT': 59.4, # /seconds
          'kcatfDhaB':400, # /seconds Input
          'KmDhaBG': 0.6, # mM Input
          'km': 10**-7, 
          'kc': 10.**-5,
          'dPacking': 0.64,
          'Nmcps': 10,
          'enz_ratio': 1/1.33}
    init_conditions = { 'GInit': 200, #  2 * 10^(-4) mol/cm3 = 200 mM. 
                      'NInit': 1., # mM
                      'DInit': 1. # mM
                      }
    Nmcps = params['Nmcps']
    tolG = 0.5*init_conditions['GInit']

    def event_Gmin(t,y):
        return y[-3] - tolG
    def event_Pmax(t,y):
        return y[-1] - tolG

    params_sens_dict = create_param_symbols('kcatfDhaB',
                                            'KmDhaBG',
                                            'km',
                                            'kc',
                                            'dPacking',
                                            'Nmcps')

    # log transform parameters in params_sens_dict
    for key in params.keys():
        params[key] = np.log10(params[key])
    dS = SDerivLog10Param
    # store info about parameters
    nParams = len(params_sens_dict)
    integration_params['nParams'] = nParams
    integration_params['Sensitivity Params'] = params_sens_dict
    nSensitivityEqs = integration_params['nParams']*integration_params['nVars']
    integration_params['nSensitivityEqs'] = nSensitivityEqs

    #################################################
    # Integrate with BDF
    #################################################

    # initial conditions
    n_compounds_cell = 3
    nVars = integration_params['nVars']
    y0 = np.zeros(nVars) 
    y0[-3] = init_conditions['GInit']  # y0[-5] gives the initial state of the external substrate.
    y0[0] = init_conditions['NInit']  # y0[5] gives the initial state of the external substrate.
    y0[1] = init_conditions['DInit']  # y0[6] gives the initial state of the external substrate.
    # time samples
    # initial conditions -- sensitivity equation
    sens0 = np.zeros(nSensitivityEqs)
    for i,param in enumerate(params_sens_dict):
        if param in ['GInit', 'IInit', 'NInit', 'DInit']:
            sens0[i:nSensitivityEqs:nParams] = 1
    xs0 = np.concatenate([y0,sens0])
    # setup differential eq
    x_sp, sensitivity_sp = create_state_symbols(integration_params['nVars'], integration_params['nParams'])
    SDerivSymbolicJacParamsLambFun, SDerivSymbolicJacConcLambFun = compute_jacs(x_sp, params_sens_dict,integration_params, diffeq_params=params, dS=dS)
    dSensParams = lambda t,xs: dSens(t, xs, params, integration_params, SDerivSymbolicJacParamsLambFun,
                                     SDerivSymbolicJacConcLambFun, dS=dS)
    #create jacobian of dSensParams
    dSensSymJacSparseMatLamFun = create_jac_sens(x_sp, sensitivity_sp, params, integration_params,
                                                 SDerivSymbolicJacParamsLambFun, SDerivSymbolicJacConcLambFun, dS = dS)

    # solution params
    tol = 1e-7
    timeorig = np.logspace(np.log10(mintime),np.log10(fintime),nsamples)

    # terminal event
    starttime = time.time()
    sol = solve_ivp(dSensParams,[0, fintime+10], xs0, method="BDF", jac = dSensSymJacSparseMatLamFun, t_eval=timeorig,
                     atol=tol,rtol=tol, events=[event_Gmin,event_Pmax])
    endtime = time.time()


    #################################################
    # Plot solution
    #################################################
    volcell = integration_params['cell volume']
    volmcp = 4 * np.pi * (integration_params['Rm'] ** 3) / 3
    external_volume = integration_params['external_volume']
    colour = ['b','r','y','c','m']


    print('code time: ' + str(endtime-starttime))
    # plot state variables solution
    print(sol.message)


    # rescale the solutions
    ncompounds = 3
    timeorighours = timeorig/secstohrs


    #plot parameters
    namesvars = ['Glycerol', '3-HPA', '1,3-PDO']
    sens_vars_names = [r'$kcat_f^{DhaB}$', r'$K_M^{DhaB}$', r'$k_{m}$', r'$k_{c}$', r'$dPacking$', r'$MCP$']#, r'$G_0$', r'$I_0$']
    colour = ['b','r','y','c','m']

    # cellular solutions
    for i in range(0,ncompounds):
        ycell = sol.y[5+i, :]
        plt.plot(sol.t/secstohrs,ycell, colour[i])
    plt.title('Plot of cellular concentration')
    plt.legend(['Glycerol', '3-HPA', '1,3-PDO'], loc='upper right')
    plt.xlabel('time (hr)')
    plt.ylabel('concentration (mM)')
    plt.grid() 
    plt.show()


    # plot sensitivity variable solutions for MCP variables
    for i in range(2,2+len(namesvars)):
        figure, axes = plt.subplots(nrows=int(math.ceil(len(params_sens_dict)/2)), ncols=2, figsize=(10,10), sharex=True, sharey=True)
        soly = sol.y[(nVars + i*nParams):(nVars + (i+1)*nParams), :]
        maxy = np.max(soly)
        miny =np.min(soly)
        yub = 1.15*maxy if maxy > 0 else 0.85*maxy
        lub = 0.85*miny if miny > 0 else 1.15*miny
        for j in range(nParams):
            axes[j // 2, j % 2].plot(timeorighours, soly[j,:].T)
            axes[j // 2, j % 2].set_ylabel(r'$\log\partial (' + namesvars[i-2] + ')/\partial ' + sens_vars_names[j][1:])
            axes[j // 2, j % 2].set_title(sens_vars_names[j])
            axes[j // 2, j % 2].set_ylim([lub, yub])
            axes[j // 2, j % 2].grid()
            if j >= (nParams-2):
                axes[(nParams-1) // 2, j % 2].set_xlabel('time/hrs')

        figure.suptitle(r'Sensitivity, $\partial (' + namesvars[i-2]+')/\partial p_i$, of the MCP concentration of '
                        + namesvars[i-2] + ' wrt $p_i$', y = 0.92)
        # plt.savefig('/Users/aarcher/PycharmProjects/MCP/WholeCell/plots/Perm_SensitivityInternal_'+ namesvars[i-2] +'.png',
        #             bbox_inches='tight')
        plt.show()


   # plot sensitivity variable solutions for cellular variables
    for i in range(2+len(namesvars),2+2*len(namesvars)):
        figure, axes = plt.subplots(nrows=int(math.ceil(len(params_sens_dict)/2)), ncols=2, figsize=(10,10), sharex=True, sharey=True)
        soly = sol.y[(nVars + i*nParams):(nVars + (i+1)*nParams), :]
        maxy = np.max(soly)
        miny = np.min(soly)
        yub = 1.15*maxy if maxy > 0 else 0.85*maxy
        lub = 0.85*miny if miny > 0 else 1.15*miny
        for j in range(nParams):
            axes[j // 2, j % 2].plot(timeorighours, soly[j,:].T)
            axes[j // 2, j % 2].set_ylabel(r'$\partial (' + namesvars[i-2-len(namesvars)] + ')/\partial ' + sens_vars_names[j][1:])
            axes[j // 2, j % 2].set_title(sens_vars_names[j])
            axes[j // 2, j % 2].grid()
            axes[j // 2, j % 2].set_ylim([lub, yub])
            if j >= (nParams-2):
                axes[(nParams-1) // 2, j % 2].set_xlabel('time/hrs')


        figure.suptitle(r'Sensitivity, $\partial (' + namesvars[i-2-len(namesvars)]+')/\partial p_i$, of the cellular concentration of '
                        + namesvars[i-2-len(namesvars)] + ' wrt $p_i$', y = 0.92)
        # plt.savefig('/Users/aarcher/PycharmProjects/MCP/WholeCell/plots/Perm_SensitivityInternal_'+ namesvars[i-2] +'.png',
        #             bbox_inches='tight')
        plt.show()


    # sensitivity variables
    for i in range(-len(namesvars),0):
        figure, axes = plt.subplots(nrows=int(math.ceil(len(params_sens_dict)/2)), ncols=2, figsize=(10,10), sharex=True,sharey=True)
        if i == -3:
            soly = sol.y[-(nParams):,:]
        else:
            soly = sol.y[-(i+1)*nParams:-i*nParams, :]
        maxy = np.max(soly)
        miny = np.min(soly)
        yub = 1.15*maxy if maxy > 0 else 0.85*maxy
        lub = 0.85*miny if miny > 0 else 1.15*miny
        for j in range(nParams):
            axes[j // 2, j % 2].plot(timeorighours, soly[j,:].T)
            axes[j // 2, j % 2].set_ylabel(r'$\partial (' + namesvars[i+3] + ')/\partial ' + sens_vars_names[j][1:])
            axes[j // 2, j % 2].set_ylim([lub, yub])
            axes[j // 2, j % 2].set_title(sens_vars_names[j])
            axes[j // 2, j % 2].grid()
            if j >= (nParams-2):
                axes[(nParams-1) // 2, j % 2].set_xlabel('time/hrs')

        figure.suptitle(r'Sensitivity, $\partial (' + namesvars[i + 3]+')/\partial p_i$, of the external concentration of '
                        + namesvars[i + 3] + ' wrt $p_i$', y = 0.92)
        # plt.savefig('/Users/aarcher/PycharmProjects/MCP/WholeCell/plots/Perm_SensitivityExternal_'+ namesvars[i+3]  +'.png',
        #             bbox_inches='tight')
        plt.show()


    #check mass balance
    ext_masses_org = y0[(nVars-3):nVars]* external_volume
    cell_masses_org = y0[5:8] * volcell 
    mcp_masses_org = y0[:5] * volmcp
    ext_masses_fin = sol.y[(nVars-3):nVars, -1] * external_volume
    cell_masses_fin = sol.y[5:8,-1] * volcell
    mcp_masses_fin = sol.y[:5, -1] * volmcp
    print(ext_masses_org.sum() + Ncells*cell_masses_org.sum() + Ncells*Nmcps*mcp_masses_org.sum())
    print(ext_masses_fin.sum() + Ncells*cell_masses_fin.sum() + Ncells*Nmcps*mcp_masses_fin.sum())


if __name__ == '__main__':
    main()


