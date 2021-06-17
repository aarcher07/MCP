"""
Parallelizes the Active_Subspaces.py code. This code generates the
average parameter directions that most affects the model in a bounded region
of parameter space.

Programme written by aarcher07
Editing History:
- 9/11/20
"""

import numpy as np
from scipy.integrate import solve_ivp
from mpi4py import MPI
import matplotlib as mpl
mpl.rc('text', usetex = True)
import matplotlib.pyplot as plt
import warnings
import sympy as sp
import scipy.sparse as sparse
import os
import sys

from DhaB_DhaT_Model import *
from DhaB_DhaT_Model_LocalSensAnalysis import *

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()


def cost_fun_unif(jac,M,names,bounds,maxN=10,output_level=1,nfuns = 100,maxValError=100,tol=1e-5):
    """
    Monte Carlo integration estimate of the cost function
    :param jac: jacobian of the function
    :param M: number of parameters that the function depends on
    :param bounds: bounds on parameter space
    :param maxN: maximum number iterations for Monte Carlo integration
    :param output_level: 1 - No output, 2 - end result, 3 - iteration output
    :return (1/N)*C: Monte Carlo estimate of the cost function
    """
    localcountValError = 0
    fro_ratio = np.Inf
    N = 0
    C = np.zeros((nfuns,M,M))
    log10bounds = np.log10(bounds)
    bound_difflog10 = log10bounds[:,1] - log10bounds[:,0]
    param_samples = []

    if rank == 0:
        output_header = '%12s %16s %30s %18s' % (' iteration ',  ' max ||Delta C||_F ',
                                                 ' max ||Delta C||_F/||C_{i}||_F ', ' Num of Value Error ')
        print(output_header)
        status = -99

    while 1:
        # updates
        x = (bound_difflog10*np.random.uniform(0,1,size=M) + log10bounds[:,0])
        dict_vals = {name:np.log2(10**val) for name,val in zip(names,x)}
        j = jac(dict_vals)
        outer_prod = np.array([(1/np.prod(bound_difflog10))*np.outer(j[i, :],j[i, :]) for i in range(j.shape[0])])
        if outer_prod.shape[0] == nfuns:
            C += comm.allreduce(outer_prod,MPI.SUM)
            N += size

            # compute norms and ratios
            fro_outer = np.array([np.linalg.norm(outer_prod[i, :, :], ord='fro') for i in range(nfuns)])
            fro_C = np.array([np.linalg.norm(C[i, :, :], ord='fro') for i in range(nfuns)])
            max_fro_outer = np.max(fro_outer)
            fro_ratio = np.max(fro_outer / fro_C)

            # compute max norms and ratios across each process
            max_fro_ratio = comm.allreduce(fro_ratio,MPI.MAX)
            max_max_fro_outer = comm.allreduce(max_fro_outer,MPI.MAX)
            param_samples.append(dict_vals)
            globalcountValError = comm.allreduce(localcountValError,MPI.SUM)

            # print statement for rank 0
            if (rank == 0) and (N % (5*size) == 0) and (output_level == 3):
                output = '%10i %18.4e %22.4e %22i' % (N, max_max_fro_outer, max_fro_ratio, globalcountValError)
                print(output)

            if (rank == 0) and (N % (100*size) == 0) and (output_level == 3):
                print(output_header)

            # break statements
            if max_max_fro_outer < tol:
                status = 0
                break

            if N >= maxN:
                status = -1
                break

            if globalcountValError >= maxValError:
                status = -2
                break

        else:
            localcountValError += 1

    if rank == 0:
        # Final output message
        if output_level >= 1:
            print('')
            print('max ||Delta C||_F/||C_{i}||_F ......................: %20.4e' % fro_ratio)
            print('total number of iterations .........................: %d' % N)
            print('total number of solutions with ValuerError .........: %d' % globalcountValError)
            print('')
            if status == 0:
                print('Exit: Converged within tolerance.')
            elif status == -1:
                print('Exit: Maximum number of iterations, (%d), exceeded.' %
                      maxN)
            elif status == -2:
                print('Exit: Maximum number of solutions with ValueError, (%d), exceeded. '
                      'You may need to increase atol and rtol.' %
                      maxValError)
            else:
                print('ERROR: Unknown status value: %d\n' % status)

    return [param_samples,(1/N)*C]

def jac(dict_vals,integration_params, diffeq_params, init_conditions, SDerivSymbolicJacParamsLambFun,
        SDerivSymbolicJacConcLambFun, dSensSymJacSparseMatLamFun, dS = SDeriv,integrationtol=1e-3, 
        fintime = 72*60*60, mintime = 10**(-15),nsamples = 100):
    """
    :param dict_vals:
    :param integration_params:
    :param diffeq_params:
    :param init_conditions:
    :param SDerivSymbolicJacParamsLambFun:
    :param SDerivSymbolicJacConcLambFun:
    :param dSensSymJacSparseMatLamFun:
    :param dS:
    :param integrationtol:
    :param fintime:
    :return:
    """

    # get integration parameters
    params_sens_dict = integration_params['Sensitivity Params']
    nSensitivityEqs = integration_params['nSensitivityEqs']
    nParams = integration_params['nParams']
    nVars = integration_params['nVars']

    # put differential equation parameters in the dictionary format
    for key, value in dict_vals.items():
        diffeq_params[key] = value

    # initial conditions
    y0 = np.zeros(nVars)    
    y0[-3] = init_conditions['GInit'] 
    y0[0] = init_conditions['NInit'] 
    y0[1] = init_conditions['DInit'] 

    sens0 = np.zeros(nSensitivityEqs)  #initial conditions -- sensitivity equation
    for i,param in enumerate(params_sens_dict):
        if param in ['GInit', 'IInit', 'NInit', 'DInit']:
            sens0[i:nSensitivityEqs:nParams] = 1
    xs0 = np.concatenate([y0,sens0])
    dSensParams = lambda t,xs: dSens(t, xs, diffeq_params, integration_params,
                                     SDerivSymbolicJacParamsLambFun, SDerivSymbolicJacConcLambFun,
                                     dS = dS) #senstivity equation

    dSensSymJacSparseMatLamFunTXS = lambda t,xs: dSensSymJacSparseMatLamFun(t,xs,list(dict_vals.values())) #jacobian
    timeorig = np.logspace(np.log10(mintime),np.log10(fintime),nsamples)

    tolsolve = 10**-4
    def event_stop(t,xs):
        dSsample = sum(np.abs(dSensParams(t,xs)))
        return dSsample - tolsolve 
    event_stop.terminal = True
    sol = solve_ivp(dSensParams,[0, fintime+1], xs0, method="BDF", jac = dSensSymJacSparseMatLamFunTXS,t_eval=timeorig,
                    atol=integrationtol, rtol=integrationtol,events=event_stop)

    return sol.y.T

def create_jac_sens_param(x_sp,sensitivity_sp,param_sp, diffeq_params,integration_params,
                    SDerivSymbolicJacParamsLambFun,SDerivSymbolicJacConcLambFun, dS=SDeriv):
    """
    Computes the jacobian matrix of the dSens that depends on parameters
    :param x_sp: symbols of the state variables
    :param sensitivity_sp: symbols of the senstivity equation
    :param param_sp: parameter symbols
    :param diffeq_params: dictionary of parameter values. param_sp is contained in diffeq_params.
    :param integration_params: dictionary of integration parameters
    :param SDerivSymbolicJacParamsLambFun: jacobian of spatial derivative wrt params
    :param SDerivSymbolicJacConcLambFun: jacobian of spatial derivative wrt state variables
    :return dSensSymJacSparseMatLamFun: sparse jacobian of dSens wrt the concentrations
    """

    # create state variables
    allVars = np.concatenate((x_sp,sensitivity_sp))
    diffeq_params = diffeq_params.copy()
    for key, value in param_sp.items():
        diffeq_params[key] = value


    #create RHS
    dSensSym = sp.Matrix(dSens(0,allVars, diffeq_params, integration_params,
          SDerivSymbolicJacParamsLambFun, SDerivSymbolicJacConcLambFun,dS=dS))
    dSensSymJac = dSensSym.jacobian(allVars)

    # generate jacobian
    dSensSymJacDenseMatLam = sp.lambdify((allVars,list(param_sp.values())),dSensSymJac)
    dSensSymJacSparseMatLamFun = lambda t,xs,param: sparse.csr_matrix(dSensSymJacDenseMatLam(xs,param))

    return dSensSymJacSparseMatLamFun


def main(maxN = 100):
    maxN = int(maxN)
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

    # log transform parameters in params_sens_dict
    for key in params.keys():
        params[key] = np.log2(params[key])
    # compute non-dimensional scaling
    params_sens_dict = create_param_symbols('km','kc')

    # create a dictionary of integration parameters
    nParams = len(params_sens_dict)
    nsamples = 500
    external_volume =  9e-6
    NcellsPerMCubed = 8e14 # 7e13-8e14 cells per m^3
    Ncells = external_volume*NcellsPerMCubed   
    mintime = 10**(-15)
    secstohrs = 60*60
    fintime = 72*60*60
    timeorig = np.logspace(np.log10(mintime),np.log10(fintime),nsamples)
    integration_params = initialize_integration_params(external_volume = external_volume, 
                                                       Ncells =Ncells,cellular_geometry="rod",
                                                       Rc = 0.375e-6, Lc = 2.47e-6)


    dS = SDerivLog2Param

    # store info about parameters
    nParams = len(params_sens_dict)
    integration_params['nParams'] = nParams
    integration_params['Sensitivity Params'] = params_sens_dict
    nSensitivityEqs = integration_params['nParams']*integration_params['nVars']
    integration_params['nSensitivityEqs'] = nSensitivityEqs


    #################################################
    # Integration arguments of BDF
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

    SDerivSymbolicJacParamsLambFun, SDerivSymbolicJacConcLambFun = compute_jacs(x_sp, params_sens_dict, integration_params, 
                                                                                diffeq_params=params, dS=dS)

    dSensSymJacSparseMatLamFunTXSParam = create_jac_sens_param(x_sp, sensitivity_sp, params_sens_dict, params, integration_params,
                                                               SDerivSymbolicJacParamsLambFun, SDerivSymbolicJacConcLambFun, dS=dS)
    #############################################
    ##### Active Subspaces Initialization
    #############################################

    integrationtol = 1e-10
    tol  = 1e-5
    maxValError = int(2e2)
    jac_F = lambda dict_vals: jac(dict_vals,integration_params,params,init_conditions,SDerivSymbolicJacParamsLambFun,
                                           SDerivSymbolicJacConcLambFun, dSensSymJacSparseMatLamFunTXSParam, dS= dS,
                                           integrationtol=integrationtol,fintime=fintime,nsamples=nsamples,mintime=mintime)

    # compute jacobian of G,P after 5 hours
    timeorighours = timeorig/secstohrs

    # sensitivity variables external
    timecheck = 5.
    ind_first_close_enough =  np.argmin(np.abs(timeorighours-timecheck)) # index of closest time after 5hrs
    
    ind_P_ext = -3 # index of P_ext 
    range_P_ext = range(ind_P_ext*nParams,(ind_P_ext+1)*nParams)
    ind_sens_P_ext = np.array([(ind_first_close_enough,i) for i in range_P_ext])
    
    ind_G_ext = -1 # index of G_ext,P_ext 
    range_G_ext = range(ind_G_ext*nParams,(ind_G_ext+1)*nParams)
    ind_sens_G_ext =  np.array([(ind_first_close_enough,i) for i in range_G_ext])
    
    ind_H_cys = 4 + 2 # index of H_cystosol
    range_H_cys = range(nVars+ind_H_cys*nParams,nVars+(ind_H_cys+1)*nParams)
    nfuns = 3

    # equilibium before timecheck hrs
    ind_sens_P_ext_after_timecheck = np.array([(-1,i) for i in range_P_ext])
    ind_sens_G_ext_after_timecheck = np.array([(-1,i) for i in range_G_ext])

    def jac_f(vars):
        jacobian_sample = jac_F(vars)
        ind_HPA_max = np.argmax(jacobian_sample[:,ind_H_cys]) 
        ind_sens_H_cys =  np.array([[ind_HPA_max,i] for i in range_H_cys])
        jac_HPA_max = jacobian_sample[tuple(ind_sens_H_cys.T)]

        if jacobian_sample.shape[0] >= ind_first_close_enough:
            jac_P_ext = jacobian_sample[tuple(ind_sens_P_ext.T)]
            jac_G_ext = jacobian_sample[tuple(ind_sens_G_ext.T)]
        else:
            jac_P_ext = jacobian_sample[tuple(ind_sens_P_ext_after_timecheck.T)]
            jac_G_ext = jacobian_sample[tuple(ind_sens_G_ext_after_timecheck.T)]
        return np.array([jac_HPA_max,jac_P_ext,jac_G_ext])

    # set bounds
    bounds = np.array([[10**(-7),10**(-4)],[10**(-7),10**(-4)]])
    # dict_vals = {name:val for name,val in zip(params_sens_dict.keys(),[np.log2(10**-7),np.log2(10**-5)])}
    # j = jac_f(dict_vals,nsamples)
    # print(j)
    # out = np.stack([np.outer(j[i, :],j[i, :]) for i in range(j.shape[0])])
    # print(out)
    # print(out.shape)
    param_samples, cost_mat = cost_fun_unif(jac_f,nParams,list(params_sens_dict.keys()),bounds,
                                            maxN=maxN,output_level=3,nfuns=nfuns,
                                            maxValError=maxValError, tol=tol)
    w,v = np.linalg.eigh(cost_mat)

    ########################### Solving with parameter set sample #######################
    if rank == 0:
        parent_folder_name = ''.join(name + '_' for name in params_sens_dict.keys())[:-1]
        child_folder_name = 'maxN_' + str(maxN)
        folder_path = parent_folder_name + '/' + child_folder_name

        # create folders
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        if not os.path.exists(folder_path + "/cost_mat"):
            os.makedirs(folder_path + "/cost_mat")
        if not os.path.exists(folder_path + "/eigenvectors"):
            os.makedirs(folder_path + "/eigenvectors")
        if not os.path.exists(folder_path + "/eigenvalues"):
            os.makedirs(folder_path + "/eigenvalues")

        # save files
        bounds_names = ''.join(name + "_" + "{:.2e}".format(bd[0]) + "_" + "{:.2e}".format(bd[1]) + "_" for name,bd in zip(params_sens_dict.keys(),bounds))[:-1]
        with open(folder_path + "/cost_mat/" + bounds_names + ".txt", 'w') as outfile:
            for slice_2d in cost_mat:
                np.savetxt(outfile, slice_2d)
        with open(folder_path + "/eigenvectors/" + bounds_names + ".txt", 'w') as outfile:
            for slice_2d in v:
                np.savetxt(outfile, slice_2d)
        np.savetxt(folder_path + "/eigenvalues/" + bounds_names + ".txt", w, delimiter=",")

if __name__ == '__main__':
    main(*sys.argv[1:])


