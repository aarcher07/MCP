"""
Generates plots of the steady state of the DhaB12-DhaT-IcdE model
as at most 2 parameters are varied.

Editing History:
- 26/10/20
"""

import numpy as np
from scipy.integrate import solve_ivp
import sympy as sp
import scipy.sparse as sparse
import pdb
from mpi4py import MPI
import time
import matplotlib.pyplot as plt
from Whole_Cell_Engineered_System_IcdE.py import *

def plot_steady_state_param(param_sens = ['km'], param_sens_bounds = np.array([[1.,10.]]), nparamsvals=10,
                            inds=[-3], params=None):
    """
    Plots of steady state concentration with index, inds, as function of parameters,
    param_sens. param_sens has bounds, param_sens_bounds. The non-varying parameters 
    are params.  

    :param param_sens: list of strings denoting varying parameters
    :param param_sens_bound: 2D numpy array of bounds for varying parameters
    :param nparamsvals: number of logarithmic spaced points with param_sens_bound bounds
	:param inds: index of concentration which is being studied
	:param params: values of non-varying parameters 
	
	Note: only pairs of parameters can be visualized at a time. 
    """
    #################################################
    # Define spatial derivative and jacobian
    #################################################

    # get parameters
    integration_params = initialize_integration_params(ngrid=25,Vratio=0.9)
    n_compounds_cell = 5
    # time samples
    fintime = 5.e4
    tol = 1e-12

    if params is None:
        params = {'KmDhaTH': 0.77,
                  'KmDhaTN': 0.03,
                  'KiDhaTD': 0.23,
                  'KiDhaTP': 7.4,
                  'VfDhaT': 86.2,
                  'VfDhaB': 10.,
                  'KmDhaBG': 0.01,
                  'KiDhaBH': 5.,
                  'VfIcdE' : 30.,
                  'KmIcdED' : 0.1,
                  'KmIcdEI' : 0.02,
                  'KiIcdEN' : 3.,
                  'KiIcdEA' : 10.,
                  'km': 0.1,
                  'kc': 1.,
                  'k1': 10.,
                  'k-1': 2.,
                  'DhaB2Exp': 1,
                  'iDhaB1Exp':2,
                  'SigmaDhaB': 10**-1,
                  'SigmaDhaT': 10**-1,
                  'SigmaIcdE': 10**-1,
                  'GInit': 10.,
                  'IInit': 10.,
                  'NInit': 20.,
                  'DInit': 20.}

    N = len(param_sens)
    assert N == len(param_sens_bounds)

    param_sens_sp = [sp.Symbol(p) for p in param_sens]

    for key,val in zip(param_sens,param_sens_sp):
        params[key] = val

    # spatial derivative
    x_list_sp = np.array(sp.symbols('x:' + str(5 * (2 + (integration_params['ngrid'])) + 2)))

    SDerivParameterizedLambdify = sp.lambdify((x_list_sp,param_sens_sp),
                                              SDeriv(0, x_list_sp, integration_params, params))

    SDerivParameterized = lambda t,x, p: SDerivParameterizedLambdify(x,p)


    # jacobian
    SDerivSymbolic = SDeriv(0, x_list_sp, integration_params, params)
    SDerivGrad = sp.Matrix(SDerivSymbolic).jacobian(x_list_sp)
    SDerivGradFun = sp.lambdify((x_list_sp,param_sens_sp), SDerivGrad, 'numpy')
    SDerivGradFunSparse = lambda t, x, p: sparse.csr_matrix(SDerivGradFun(x,p))

    # list to store steady state values
    steadystatevals = []

    # sample parameter space
    if N == 1:
        meshedparamvals = np.logspace(param_sens_bounds[:,0], param_sens_bounds[:,1], num=nparamsvals)
        meshedparamvals = meshedparamvals.reshape(1,-1)
        M = nparamsvals

    else:
        # parameter samples
        paramvals = np.logspace(param_sens_bounds[:,0],param_sens_bounds[:,1],num=nparamsvals)
        mesh = np.meshgrid(*list(paramvals.T))
        meshedparamvals = mesh.copy()
        #################################################
        # Integrate with BDF
        #################################################

        # initial conditions
        for i in range(N):
            meshedparamvals[i] = meshedparamvals[i].flatten()
        meshedparamvals = np.array(meshedparamvals)
        M = meshedparamvals.shape[1]

    #solve the system of ODE across each parameter set
    for j in range(M):
        params_copy = params.copy()
        param_sample = meshedparamvals[:,j]
        dimscalings = initialize_dim_scaling(**params_copy)

        for key, val in zip(param_sens, param_sample):
            params_copy[key] = val

        x0 = np.zeros((integration_params['ngrid'] + 2) * n_compounds_cell + 2)
        x0[-5] = params_copy['GInit'] / dimscalings['G0']  # y0[-5] gives the initial state of the external substrate.
        x0[-1] = params_copy['IInit'] / dimscalings['I0']  # y0[-1] gives the initial state of the external substrate.
        x0[0] = params_copy['NInit'] / dimscalings['N0']  # y0[5] gives the initial state of the external substrate.
        x0[1] = params_copy['DInit'] / dimscalings['D0']  # y0[6] gives the initial state of the external substrate.

        # define dS as function of t and x and jacobian as a function of t and x
        dS = lambda t,x: SDerivParameterized(t,x,param_sample)
        dSdx = lambda t,x: SDerivGradFunSparse(t, x, param_sample)

        # terminal event
        event = lambda t, x: np.absolute(dS(t, x)[-1]) - tol
        event.terminal = True

        sol = solve_ivp(dS, [0, fintime], x0, method="BDF", jac=dSdx, events=event,
                        atol=1.0e-5, rtol=1.0e-5)

        print(sol.message)
        steadystatevals.append(sol.y[inds,-1])

    steadystatevals = np.array(steadystatevals)
    if N == 1:
        plt.plot(np.log10(meshedparamvals[0,:]),steadystatevals.T)
        plt.ylim([steadystatevals.min() - steadystatevals.min()/2.,steadystatevals.max() + steadystatevals.max()/2.])
        plt.legend(inds)
        plt.show()

    if N == 2:
        for i in range(len(inds)):
            reshapedvals = steadystatevals[:,i].reshape(nparamsvals,-1)
            plt.imshow(reshapedvals,extent=param_sens_bounds.flatten(),origin='lower',vmin=reshapedvals.min(),
                       vmax=reshapedvals.max(),cmap='jet')
            plt.colorbar()
            plt.title("Heat map of " + str(inds[i]))
            plt.show()

    if N > 3:
        print("TODO: generate a gif of the function")


if __name__ == '__main__':
    params = {'KmDhaTH': 0.77,
              'KmDhaTN': 0.03,
              'KiDhaTD': 0.23,
              'KiDhaTP': 7.4,
              'VfDhaT': 56.4,
              'VfDhaB': 600.,
              'KmDhaBG': 0.8,
              'KiDhaBH': 5.,
              'VfIcdE' : 30.,
              'KmIcdED' : 0.1,
              'KmIcdEI' : 0.02,
              'KiIcdEN' : 3.,
              'KiIcdEA' : 10.,
              'km': 10.**-4,
              'kc': 10.**-4,
              'k1': 10.,
              'k-1': 2.,
              'DhaB2Exp': 100.,
              'iDhaB1Exp': 1.,
              'SigmaDhaB': 10**-1,
              'SigmaDhaT': 10**-1,
              'SigmaIcdE': 10**-1,
              'GInit': 10.,
              'IInit': 10.,
              'NInit': 20.,
              'DInit': 20.}

    plot_steady_state_param(param_sens=['km','IInit'],
                           param_sens_bounds = np.array([[1, 10.],[1,10.]]))

    # plot_steady_state_param(param_sens=['VfDhaT','VfDhaB'],
    #                        param_sens_bounds = np.array([[0.,1.],[1.,2.]]), inds = [-3,-2])

