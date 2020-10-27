"""
Parallelizes the Active_Subspaces.py code. This code generates the
average parameter directions that most affects the model in a bounded region
of parameter space.

Programme written by aarcher07
Editing History:
- 27/10/20
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
from Whole_Cell_Engineered_System_IcdE import *
from Whole_Cell_Engineered_System_IcdE_LocalSensitivity_Analysis import *

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()


def cost_fun_unif(jac,M,names,bounds,maxN=10,output_level=1,nsamples = 100,maxValError=100,tol=1e-5):
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
    C = np.zeros((nsamples,M,M))
    bound_diff = bounds[:,1] - bounds[:,0]
    param_samples = []

    if rank == 0:
        output_header = '%12s %16s %30s %18s' % (' iteration ',  ' max ||Delta C||_F ',
                                                 ' max ||Delta C||_F/||C_{i}||_F ', ' Num of Value Error ')
        print(output_header)
        status = -99

    while 1:
        # updates
        x = bound_diff*np.random.uniform(0,1,size=M) + bounds[:,0]
        dict_vals = {name:val for name,val in zip(names,x)}
        j = jac(dict_vals, nsamples)
        outer_prod = np.array([np.prod(bound_diff)*np.outer(j[i, :],j[i, :]) for i in range(j.shape[0])])
        print(outer_prod.shape)
        print(M)
        if outer_prod.shape[0] == M:
            print('hi')
            C += comm.allreduce(outer_prod,MPI.SUM)
            N += size

            # compute norms and ratios
            fro_outer = np.array([np.linalg.norm(outer_prod[i, :, :], ord='fro') for i in range(nsamples)])
            fro_C = np.array([np.linalg.norm(C[i, :, :], ord='fro') for i in range(nsamples)])
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
            print('total number of solutions with ValuerError .........: %d' % countValError)
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

def jac(dict_vals,integration_params, SDerivSymbolicJacParamsLambFun,
        SDerivSymbolicJacConcLambFun, dSensSymJacSparseMatLamFun,
        integrationtol=1e-3, fintime = 3.e6, mintime = 1,nsamples = 100):
    """

    :param dict_vals:
    :param integration_params:
    :param SDerivSymbolicJacParamsLambFun:
    :param SDerivSymbolicJacConcLambFun:
    :param dSensSymJacSparseMatLamFun:
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

    # TODO: this can be computed outside to speed up code
    dSensParams = lambda t,xs: dSens(t, xs, diffeq_params, integration_params,
                                     SDerivSymbolicJacParamsLambFun, SDerivSymbolicJacConcLambFun) #senstivity equation

    dSensSymJacSparseMatLamFunTXS = lambda t,xs: dSensSymJacSparseMatLamFun(t,xs,list(dict_vals.values())) #jacobian


    #timeorig = np.logspace(np.log10(mintime), np.log10(fintime), nsamples)
    timeorig1 = np.logspace(np.log10(mintime), np.log10(5e3), int(0.15*nsamples), endpoint=False)
    timeorig2 = np.logspace(np.log10(5e3), np.log10(1e5), int(0.8*nsamples), endpoint=False)
    timeorig3 = np.logspace(np.log10(1e5), np.log10(fintime), int(0.05*nsamples))
    timeorig = np.concatenate([timeorig1,timeorig2,timeorig3])

    sol = solve_ivp(dSensParams,[0, fintime+1], xs0, method="BDF", jac = dSensSymJacSparseMatLamFunTXS,t_eval=timeorig,
                    atol=integrationtol, rtol=integrationtol)


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

    return sol.y[nVars:,:].T

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
              'VfIcdE' : 30.,
              'KmIcdED' : 0.1,
              'KmIcdEI' : 0.02,
              'KiIcdEN' : 3.,
              'KiIcdEA' : 10.,
              'GInit':10.,
              'IInit': 10,
              'NInit':20,
              'DInit':20}


    # compute non-dimensional scaling
    dimscalings = initialize_dim_scaling(**params)

    param_sp = create_param_symbols('km','kc')


    # create a dictionary of integration parameters
    ngrid = 10
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
    print('hi1')
    # compute set up senstivity equations
    SDerivSymbolicJacParamsLambFun, SDerivSymbolicJacConcLambFun = compute_jacs(x_sp, param_sp,
                                                                                integration_params,
                                                                                diffeq_params=params)


    dSensSymJacSparseMatLamFunTXSParam = create_jac_sens_param(x_sp, sensitivity_sp, param_sp, integration_params,
                                                               SDerivSymbolicJacParamsLambFun, SDerivSymbolicJacConcLambFun)
    print('hi0')
    mintime = 1
    fintime = 1e7
    integrationtol = 1e-2
    tol  = 1e-5
    nsamples = 500
    maxN = int(1e3)
    maxValError = int(2e2)
    jac_F = lambda dict_vals,nsamples: jac(dict_vals,integration_params, SDerivSymbolicJacParamsLambFun,
                                           SDerivSymbolicJacConcLambFun, dSensSymJacSparseMatLamFunTXSParam,
                                           integrationtol=integrationtol,fintime=fintime,nsamples=nsamples,mintime=mintime)

    # compute jacobian of P
    ind_P_ext = 3 # index of P_ext at steady state
    jac_f = lambda vars, nsamples: jac_F(vars,nsamples)[:,-ind_P_ext*nParams:-(ind_P_ext-1)*nParams]

    # set bounds
    bounds = np.array([[0.01,20],[0.01,20]])
    param_samples, cost_mat = cost_fun_unif(jac_f,nParams,list(param_sp.keys()),bounds,
                                            maxN=maxN,output_level=3,nsamples=nsamples,
                                            maxValError=maxValError, tol=tol)
    w,v = np.linalg.eigh(cost_mat)


    ########################### Solving with parameter set sample #######################
    if rank == 0:
        param_dict = params.copy()
        param_subset = param_samples[0]

        for key in param_sp.keys():
            param_dict[key] = param_subset[key]

        dimscalings = integration_params['dimscalings']


        # initial conditions
        y0 = np.zeros(nVars)     # initial conditions -- state variable
        y0[-5] = param_dict['GInit'] / dimscalings['G0']  # y0[-5] gives the initial state of the external substrate.
        y0[-1] = param_dict['IInit'] / dimscalings['I0']  # y0[-1] gives the initial state of the external substrate.
        y0[0] = param_dict['NInit'] / dimscalings['N0']  # y0[5] gives the initial state of the external substrate.
        y0[1] = param_dict['DInit'] / dimscalings['D0']  # y0[6] gives the initial state of the external substrate.

        # TODO: this can be computed outside of loop to speed up code
        SDerivParameterized = lambda t, x: SDeriv(t, x, integration_params, param_dict)
        SDerivSymbolicJacConcLambFunParamterized = lambda t,x: SDerivSymbolicJacConcLambFun(t,x,param_subset.values())

        timeorig1 = np.logspace(np.log10(mintime), np.log10(5e3), int(0.15*nsamples), endpoint=False)
        timeorig2 = np.logspace(np.log10(5e3), np.log10(1e5), int(0.80*nsamples), endpoint=False)
        timeorig3 = np.logspace(np.log10(1e5), np.log10(fintime), int(0.05*nsamples))
        timeorig = np.concatenate([timeorig1,timeorig2,timeorig3])

        sol = solve_ivp(SDerivParameterized,[0, fintime+1], y0, method="BDF", jac = SDerivSymbolicJacConcLambFunParamterized
                        ,t_eval=timeorig, atol=1e-5, rtol=1e-5)

        xvalslogtimeticks = list(range(int(np.log10(fintime))+1))

        mineig = int(np.log10(np.min(w)) - 2)
        maxeig = int(np.log10(np.max(w)) + 2)
        yvalslogtimeticks = list(range(mineig,maxeig))

        xtexlogtimeticks = [r'$10^{' + str(i) + '}$' for i in range(int(np.log10(fintime))+1)]
        ytexlogtimeticks = [r'$10^{' + str(i) + '}$' for i in range(mineig,maxeig)]
        print(sol.message)
        print(sol.y.shape)


        # TODO: overlay total mass in cell
        # TODO: overlay total concentration in external and internal

        #plot eigenvalues
        plt.scatter(np.log10(timeorig), np.log10(w[:,0]))
        plt.scatter(np.log10(timeorig), np.log10(w[:,1]))
        plt.xticks(xvalslogtimeticks, xtexlogtimeticks)
        plt.yticks(yvalslogtimeticks, ytexlogtimeticks)
        plt.xlabel('time')
        plt.ylabel('eigenvalue')
        plt.legend([r'$\lambda_1$',r'$\lambda_2$'])
        plt.show()


        logtimeorig = np.log10(timeorig)
        timeseries_eigfirst = v[:, :, 0]
        timeseries_eigsecond = v[:, :, 1]

        plt.scatter(logtimeorig,timeseries_eigfirst[:,0])
        plt.scatter(logtimeorig,timeseries_eigfirst[:,1])
        plt.title('Scatter plot of first eigenvector')
        plt.xticks(xvalslogtimeticks, xtexlogtimeticks)
        plt.legend([r'$v_{1}$',r'$v_{2}$'])
        plt.ylabel(r'$v_{\cdot,\cdot}$')
        plt.xlabel('time')
        plt.show()

        plt.scatter(logtimeorig,timeseries_eigsecond[:,0])
        plt.scatter(logtimeorig,timeseries_eigsecond[:,1])
        plt.legend([r'$v_{1}$',r'$v_{2}$'])
        plt.title('Scatter plot of second eigenvector')
        plt.xticks(xvalslogtimeticks, xtexlogtimeticks)
        plt.xlabel('time')
        plt.ylabel(r'$v_{\cdot,\cdot}$')
        plt.show()


if __name__ == '__main__':
    main()


