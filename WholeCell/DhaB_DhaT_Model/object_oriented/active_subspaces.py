"""
Parallelized the Active_Subspaces.py code.

Programme written by aarcher07
Editing History:
- 9/11/20
"""

import numpy as np
from numpy.linalg import LinAlgError
from scipy.integrate import solve_ivp
from pathos.multiprocessing import ProcessingPool as Pool
import matplotlib as mpl
from mpi4py import MPI
mpl.rc('text', usetex = True)
import matplotlib.pyplot as plt
import warnings
import sympy as sp
import scipy.sparse as sparse
import os
import sys

from dhaB_dhaT_model_jac import *
from active_subspaces_dhaT_dhaB_model import *

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

class ActiveSubspaces:
    def __init__(self,jac, nfuncs, nparams, niters=10**3, rdist = 'unif'):
        """
        Initializes a class that computes and ranks the average sensitivity matrix  
        of each function used to compute jac, the sensitivity matrix of the functions. 

        The average sensitivity matrix is computed using Monte Carlo Integration. 

        :params jac            : jacobian of the problem at hand. jac returns an
                                 an array of dimensions, (nfuncs, nparams, nparams).
        :params nparams        : number of parameters whose senstivities being studied                         
        :params nfuncs          : number of functions whose jacobians are being evaluated
        :params niters         : maximum number of iterations
        :params dist           : distribution of the parameters
        """

        self.jac = jac
        self.nfuncs = nfuncs
        self.nparams = nparams
        self.niters = niters
        if rdist == 'unif':
            dist = lambda M: 2*np.random.uniform(0,1,size=M)-1
        self.dist = dist
        self.param_samples = []

    def compute_cost_matrix(self):
        """
        Monte Carlo integration estimate of the cost function matrix
        """

        if rank == 0:
            niters_rank = self.niters//size + self.niters % size
        else:
            niters_rank = self.niters//size
        # generate data
        param_samples_rank = []
        jac_list_rank = []
        for _ in range(niters_rank):  
            sample = self.dist(self.nparams) # sample transformed unit form distribution
            param_samples_rank.append(sample)
            jac_list_rank.append(self.jac(sample))

        # gather data
        jac_list = None
        param_samples = None
        jac_list = comm.gather(jac_list_rank, root=0)
        param_samples = comm.gather(param_samples_rank, root=0)

        if rank == 0:
            #flatten data
            jac_list_flattened = [item for sublist in jac_list for item in sublist]
            param_samples_flattened = [item for sublist in param_samples for item in sublist]

            # remove unsuccessful integrations 
            jac_list_cleaned_reordered = [[] for _ in range(self.nfuncs)]
            for jac_sample in jac_list_flattened:
                for i in range(self.nfuncs):
                    if len(jac_sample[i]) != 0:
                        jac_list_cleaned_reordered[i].append(jac_sample[i])

            # count successful integrations
            nfuncs_successes = []
            for i in range(self.nfuncs):
                nfuncs_successes.append(len(jac_list_cleaned_reordered[i])*1.0)

            # compute cost matrix
            cost_matrix = [np.zeros((self.nparams,self.nparams)) for _ in range(self.nfuncs)]
            for i in range(self.nfuncs):
                for jac_est in jac_list_cleaned_reordered[0]:
                    cost_matrix[i] += np.outer(jac_est,jac_est)/nfuncs_successes[i]

            # compute cost matrix
            variance_matrix = [np.zeros((self.nparams,self.nparams)) for _ in range(self.nfuncs)]
            for i in range(self.nfuncs):
                for jac_est in jac_list_cleaned_reordered[0]:
                    variance_matrix[i] += (np.outer(jac_est,jac_est)-cost_matrix[i])**2/(nfuncs_successes[i]-1)            
            results = [nfuncs_successes, jac_list_cleaned_reordered, variance, cost_matrix]

            return results



def test():
    f = lambda x: np.exp(0.7*x[0] + 0.3*x[1])
    jac = lambda x: [np.array([0.7*f(x),0.3*f(x)])]
    as_test = ActiveSubspaces(jac, 1, 2,niters=10)
    results = as_test.compute_cost_matrix()
    if rank == 0:
        print(np.linalg.eig(results[-1]))

def dhaB_dhaT_model(argv, arc):
    # get inputs
    enz_ratio_name = argv[1]
    
    # initialize variables
    ds = ''
    start_time = (10**(-15))
    final_time = 100*HRS_TO_SECS
    integration_tol = 1e-5
    nsamples = 500
    enz_ratio_name_split =  enz_ratio_name.split(":")
    enz_ratio = float(enz_ratio_name_split[0])/float(enz_ratio_name_split[1])
    params_values_fixed = {'NAD_MCP_INIT': 0.1,
                          'enz_ratio': enz_ratio,
                          'G_MCP_INIT': 0,
                          'H_MCP_INIT': 0,
                          'P_MCP_INIT': 0,
                          'G_CYTO_INIT': 0,
                          'H_CYTO_INIT': 0,
                          'P_CYTO,INIT': 0 ,
                          'G_EXT_INIT': 200,
                          'H_EXT_INIT': 0,
                          'P_EXT_INIT': 0}

    param_sens_bounds = {'kcatfDhaB': [400, 860], # /seconds Input
                        'KmDhaBG': [0.6,1.1], # mM Input
                        'kcatfDhaT': [40.,100.], # /seconds
                        'KmDhaTH': [0.1,1.], # mM
                        'KmDhaTN': [0.0116,0.48], # mM
                        'NADH_MCP_INIT': [0.12,0.60],
                        'km': np.log10([10**-8,10**-6]), 
                        'kc': np.log10([10**-8,10**-4]),
                        'dPacking': [0.3,0.64],
                        'nmcps': [3.,30.]}

    dhaB_dhaT_model_jacobian_as = DhaBDhaTModelJacAS(start_time, final_time, integration_tol, nsamples,
                                                params_values_fixed,param_sens_bounds, ds = ds)
    def dhaB_dhaT_jac(runif):
        param_sens_dict = {param_name: val for param_name,val in zip(param_sens_bounds.keys(),runif)}
        return dhaB_dhaT_model_jacobian_as.jac_subset(param_sens_dict) 

    as_dhaB_dhaT_mod = ActiveSubspaces(dhaB_dhaT_jac, 3, len(param_sens_bounds),niters=10**6)
    results = as_dhaB_dhaT_mod.compute_cost_matrix()
    if rank == 0:
        print('number of functions evaluations')
        print(results[0])
        print('\n variance')
        print(results[-2])
        print('\n cost matrix')
        print(results[-1])
        print('\n eigenvalues, eigenvectors 3-HPA max')
        print(np.linalg.eig(results[-1][0]))

if __name__ == '__main__':
    dhaB_dhaT_model(sys.argv, len(sys.argv))

