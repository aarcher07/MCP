"""
Parallelized the Active_Subspaces.py code.

Programme written by aarcher07
Editing History:
- 9/11/20
"""

import numpy as np
from numpy.linalg import LinAlgError
from scipy.integrate import solve_ivp
import time
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpi4py import MPI
mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}' #for \text command
import warnings
import sympy as sp
import scipy.sparse as sparse
import os
import sys
import pickle
from skopt.space import Space
from dhaB_dhaT_model_jac import *
from active_subspaces_dhaT_dhaB_model import *
from misc import *

from skopt.sampler import Lhs, Sobol
from skopt.space import Space

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()



class ActiveSubspaces:
    def __init__(self,jac, nfuncs, nparams, niters=10**3, sampling = 'rsampling'):
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
        self.sampling = sampling
        self.sample_space = Space([(-1.,1.) for _ in range(self.nparams)])
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
        param_samples_diff_int_rank = []
        jac_list_rank = []
        
        if self.sampling == "LHS":
            lhs = Lhs(lhs_type="classic", criterion=None)
            param_samples_unorganized = lhs.generate(self.sample_space, niters_rank)
        elif self.sampling == "rsampling":
            param_samples_unorganized = self.sample_space.rvs(niters_rank)
        elif self.sampling == "Sobol":
            sobol = Sobol()
            param_samples_unorganized = sobol.generate(self.sample_space.dimensions, niters_rank)

        for sample in param_samples_unorganized:  
            jac_sample = self.jac(sample)
            if jac_sample:
                param_samples_rank.append(sample)
                jac_list_rank.append(jac_sample)
            else:
                param_samples_diff_int_rank.append(sample)


        # gather data
        jac_list = None
        param_samples = None
        param_samples_diff_int = None
        jac_list = comm.gather(jac_list_rank, root=0)
        param_samples = comm.gather(param_samples_rank, root=0)
        param_samples_diff_int = comm.gather(param_samples_diff_int_rank, root=0)

        if rank == 0:
            #flatten data
            jac_list_flattened = [item for sublist in jac_list for item in sublist]
            param_samples_flattened = [item for sublist in param_samples for item in sublist]
            param_samples_diff_int_flattened = [item for sublist in param_samples_diff_int for item in sublist]

            # remove unsuccessful integrations 
            jac_list_cleaned_reordered = [[] for _ in range(self.nfuncs)] # store list of outer product of jac for each QoI
            nfuncs_successes = [0. for _ in range(self.nfuncs)] # count successful integration

            for jac_sample in jac_list_flattened:
                for i in range(self.nfuncs):
                    if len(jac_sample[i]) != 0:
                        jac_list_cleaned_reordered[i].append(np.outer(jac_sample[i],jac_sample[i]))
                        nfuncs_successes[i] += 1.

            # compute cost matrix and norm convergence
            cost_matrix = []
            cost_matrix_cumul = []
            norm_convergence = []

            for i in range(self.nfuncs):
                cost_cumsum = np.cumsum(jac_list_cleaned_reordered[i],axis=0)/np.arange(1,nfuncs_successes[i]+1)[:,None,None]
                cost_matrix_cumul.append(cost_cumsum)
                cost_matrix.append(cost_cumsum[-1,:,:])
                norm_convergence.append(np.linalg.norm(cost_cumsum,ord='fro',axis=(1,2))) 

            # compute variance matrix
            variance_matrix = []
            for i in range(self.nfuncs):
                variance_mat = np.sum((jac_list_cleaned_reordered[i]-cost_matrix[i])**2/(nfuncs_successes[i]-1),axis=0)            
                variance_matrix.append(variance_mat)
            param_results = {"PARAM_SAMPLES": param_samples_flattened,
                             "DIFFICULT_PARAM_SAMPLES": param_samples_diff_int_flattened}

            fun_results = {"NUMBER_OF_FUNCTION_SUCCESS": nfuncs_successes,
                           "NORM_OF_SEQ_OF_CUMUL_SUMS": norm_convergence,
                           "SEQ_OF_CUMUL_SUMS": cost_matrix_cumul, 
                           "VARIANCE_OF_ENTRIES": variance_matrix,
                           "FINAL_COST_MATRIX":cost_matrix}

            return {'PARAMETER_RESULTS': param_results, 'FUNCTION_RESULTS': fun_results}


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
    niters = int(float(argv[2]))
    sampling =  argv[3]
    threshold = float(argv[4])
    # initialize variables
    ds = ''
    start_time = (10**(-15))
    final_time = 100*HRS_TO_SECS
    integration_tol = 1e-3
    nsamples = 500
    tolsolve = 10**-10
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
                        'PermMCPPolar': np.log10([10**-4, 10**-2]),
                        'NonPolarBias': np.log10([10**-2, 10**-1]),
                        'PermCell': np.log10([10**-9,10**-4]),
                        'dPacking': [0.3,0.64],
                        'nmcps': [3.,30.]}
    
    # create object to generate jac
    dhaB_dhaT_model_jacobian_as = DhaBDhaTModelJacAS(start_time, final_time, integration_tol,
                                                     nsamples, tolsolve, params_values_fixed,
                                                     param_sens_bounds, ds = ds)
    def dhaB_dhaT_jac(runif):
        param_sens_dict = {param_name: val for param_name,val in zip(param_sens_bounds.keys(),runif)}
        return dhaB_dhaT_model_jacobian_as.jac_subset(param_sens_dict) 

    # create object to run active subspaces
    as_dhaB_dhaT_mod = ActiveSubspaces(dhaB_dhaT_jac, 3, len(param_sens_bounds),niters=niters, sampling = sampling)

    # run integration
    start_time = time.time()
    results = as_dhaB_dhaT_mod.compute_cost_matrix()
    end_time = time.time()

    # gather results and output
    if rank == 0:
        param_results = results["PARAMETER_RESULTS"]
        fun_results = results["FUNCTION_RESULTS"]

        date_string = time.strftime("%Y_%m_%d_%H:%M")

        # create folder 
        params_names = param_sens_bounds.keys()
        folder = generate_folder_name(param_sens_bounds)
        folder_name = os.path.abspath(os.getcwd()) + '/data/' + enz_ratio_name+ '/'+  folder
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)

        # store results
        file_name_pickle = folder_name + '/sampling_' + sampling + '_N_' + str(niters) + '_enzratio_' + enz_ratio_name+ '_'+ date_string + '.pkl'
        with open(file_name_pickle, 'wb') as f:
            pickle.dump(results, f)
        
        # save text output
        generate_txt_output(fun_results["FINAL_COST_MATRIX"], fun_results["NUMBER_OF_FUNCTION_SUCCESS"],
                            fun_results["VARIANCE_OF_ENTRIES"], param_results["DIFFICULT_PARAM_SAMPLES"],
                            folder_name, param_sens_bounds, size, sampling,enz_ratio_name,niters,
                            date_string, start_time,end_time)

        # save eigenvalue plots
        generate_eig_plots_QoI(fun_results["FINAL_COST_MATRIX"],param_sens_bounds,sampling,
                               enz_ratio_name, niters,date_string,threshold, save=True)

if __name__ == '__main__':
    dhaB_dhaT_model(sys.argv, len(sys.argv))

