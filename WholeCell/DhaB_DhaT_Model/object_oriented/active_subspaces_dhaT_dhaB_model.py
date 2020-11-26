"""
This code generates the average senstivity matrix that 
most affects the model in a bounded region of parameter space.

Programme written by aarcher07
Editing History:
- 25/11/20
"""

import numpy as np
from numpy.linalg import LinAlgError
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

from dhaB_dhaT_model_jac import *

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

class Active_Subspaces_DhaT_DhaB_Model(Active_Subspaces):
    def __init__(self,jac, nfuns, nparams, max_iters, output_level,
                 max_exceptions, param_bounds, param_trans = "", tol=1e-5):
        """
        Initializes a class that computes and ranks the average sensitivity matrix  
        of each function used to compute jac, the sensitivity matrix of the functions. 

        The average sensitivity matrix is computed using Monte Carlo Integration. 

        :params jac            : jacobian of the problem at hand. jac returns an
                                 an array of dimensions, (nfuncs, nparams, nparams).
        :params nparams        : number of parameters whose senstivities being studied                         
        :params nfuns          : number of functions whose jacobians are being evaluated
        :params max_iters      : maximum number of iterations
        :params output_level   : 1 - only summary output of the end of the code
                                 2 - detailed output of each iteration 
        :params max_exceptions : maximum number of evaluations exceptions
        :params param_bounds   : numpy array of the parameter bounds
        :params param_trans    : "log2" - log2 transform of the parameters
                                 "log10" - log10 transform of the parameters
                                  Otherwise, identity transform of the parameters
        :params tol            : tolerance of the Monte Carlo integration.
        """
        self.param_bounds = param_bounds

        runif = lambda M: np.random.uniform(0,1,size=M)
        if params_trans == 'log2':
            dist = lambda M: self._transform_variables_log2(runif(M))
        elif params_trans == 'log10'
            dist = lambda M: self._transform_variables_log10(runif(M))
        else:
            dist = lambda M: self._transform_variables_id(runif(M))

        super().__init__(self, jac, nfuns, nparams, max_iters, output_level,
                         max_exceptions, dist = dist, tol=1e-5)

    def _transform_variables_log10(self,runif):
        """
        Transforms uniform random numbers on [0,1] to the corresponding 
        log10 parameter distribution 
        :params runif: vector of random numbers
        :return param_estimates: dictionary of random parameter samples
        """
        param_estimates = {}
        for runif_single_val, (param,(bound_a,bound_b)) in zip(runif,self.param_bounds.items()):
            if param in ['km','kc']:
                # permeabilities
                param_estimates[param] = (bound_b - bound_a)*runif_single_val + bound_a
            elif param in PARAMETER_LIST:
                # parameter except for initial conditions and permeability
                param_estimates[param] = np.log10((bound_b - bound_a)*runif_single_val + bound_a)
            else:
                # initial conditions
                param_estimates[param] = (bound_b - bound_a)*runif_single_val + bound_a
        return param_estimates

    def _transform_variables_log2(self,runif):
        """
        Transforms uniform random numbers on [0,1] to the corresponding 
        log2 parameter distribution 

        :params runif: vector of random numbers
        :return param_estimates: dictionary of random parameter samples
        """
        param_estimates = {}
        for runif_single_val, (param,(bound_a,bound_b)) in zip(runif,self.param_bounds.items()):
            if param in ['km','kc']:
                # permeabilities
                param_estimates[param] = np.log2(10**((bound_b - bound_a)*runif_single_val + bound_a))
            elif param in PARAMETER_LIST:
                # parameter except for initial conditions and permeability
                param_estimates[param] = np.log2((bound_b - bound_a)*runif_single_val + bound_a)
            else:
                # initial conditions
                param_estimates[param] = (bound_b - bound_a)*runif_single_val + bound_a
        return param_estimates

    def _transform_variables_id(self,runif):
        """
        Transforms uniform random numbers on [0,1] to the corresponding 
        parameter distribution 

        :params runif: vector of random numbers
        :return param_estimates: dictionary of random parameter samples
        """
        param_estimates = {}
        for runif_single_val, (param,(bound_a,bound_b)) in zip(runif,self.param_bounds.items()):
            if param in ['km','kc']:
                # permeabilities
                param_estimates[param] = 10**((bound_b - bound_a)*runif_single_val + bound_a)
            elif param in PARAMETER_LIST:
                # parameter except for initial conditions and permeability
                param_estimates[param] = (bound_b - bound_a)*runif_single_val + bound_a
            else:
                # initial conditions
                param_estimates[param] = (bound_b - bound_a)*runif_single_val + bound_a
        return param_estimates