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
mpl.rc('text', usetex = True)
import matplotlib.pyplot as plt
import warnings
import sympy as sp
import scipy.sparse as sparse
import os
import sys

from dhaB_dhaT_model_jac import *

class Active_Subspaces:
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

    def compute_cost_matrix(self):
        """
        Monte Carlo integration estimate of the cost function matrix
        """
        pool = Pool(processes=4)

        param_samples = pool.map(self.dist,[self.nparams]*self.niters) # sample transformed unit form distribution
        jac_list = pool.map(self.jac,param_samples) 
        pool.close()
        pool.join()
        # remove unsuccessful integrations 
        jac_list_cleaned_reordered = [[]]*self.nfuncs
        for jac_sample in jac_list:
            for i in range(self.nfuncs):
                if len(jac_sample[i]) != 0:
                    jac_list_cleaned_reordered[i].append(jac_sample[i])

        # count successful integrations
        nfuncs_successes = []
        for i in range(self.nfuncs):
            nfuncs_successes.append(len(jac_list_cleaned_reordered[i])*1.0)

        # compute cost matrix
        cost_matrix = [np.zeros((self.nparams,self.nparams))]*self.nfuncs
        for i in range(self.nfuncs):
            for jac_est in jac_list_cleaned_reordered[0]:
                cost_matrix[i] += np.outer(jac_est,jac_est)/nfuncs_successes[i]

        results = [nfuncs_successes, jac_list_cleaned_reordered, cost_matrix]

        return results



def main():
    f = lambda x: np.exp(0.7*x[0] + 0.3*x[1])
    jac = lambda x: [np.array([0.7*f(x),0.3*f(x)])]
    as_test = Active_Subspaces(jac, 1, 2,niters=10**5)
    results = as_test.compute_cost_matrix()
    print(np.linalg.eig(results[-1]))

if __name__ == '__main__':
    main(*sys.argv[1:])


