"""
Parallelized the Active_Subspaces.py code.

Programme written by aarcher07
Editing History:
- 9/11/20
"""

import numpy as np
from numpy.linalg import LinAlgError
from scipy.integrate import solve_ivp
import multiprocessing as mp
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

class Active_Subspaces:
    def __init__(self,jac, nfuns, nparams, niters, 
                 output_level, max_exceptions,
                 dist = lambda M: np.random.uniform(0,1,size=M),
                 tol=1e-5):
        """
        Initializes a class that computes and ranks the average sensitivity matrix  
        of each function used to compute jac, the sensitivity matrix of the functions. 

        The average sensitivity matrix is computed using Monte Carlo Integration. 

        :params jac            : jacobian of the problem at hand. jac returns an
                                 an array of dimensions, (nfuncs, nparams, nparams).
        :params nparams        : number of parameters whose senstivities being studied                         
        :params nfuns          : number of functions whose jacobians are being evaluated
        :params niters         : maximum number of iterations
        :params output_level   : 1 - only summary output of the end of the code
                                 2 - detailed output of each iteration 
        :params max_exceptions : maximum number of evaluations exceptions
        :params dist           : distribution of the parameters
        :params tol            : tolerance of the Monte Carlo integration.
        """

        self.jac = jac
        self.nfuns = nfuns
        self.nparams = nparams
        self.names = names
        self.niters = niters
        self.dist = dist
        self.output_level = output_level
        self.max_exceptions = max_exceptions
        self.tol= tol

    def compute_cost_matrix():
        """
        Monte Carlo integration estimate of the cost function matrix
        """
        pool = mp.Pool(processes=4)

        param_samples = [pool.apply_async(self.dist(M)).get() for _ in range(niters)] # sample transformed unit form distribution
        jac_list = pool.map(self.jac,param_samples) 
        
        # remove unsuccessful integrations
        jac_list_cleaned_reordered = [[]]*self.nfuncs

        for jac_sample in jac_list:
            for i in self.nfuncs:
                if jac_sample[i] != -1:
                    jac_list_cleaned_reordered[i].append(jac_sample[i])

        # count successful integrations
        nfuncs_successes = []
        for i in self.nfuncs:
            nfuncs_successes.append(len(jac_list_cleaned_reordered[i])*1.0)


        # compute cost matrix
        C_HPA_max = np.zeros(self.nparams,self.nparams)
        C_HPA_max = np.zeros(self.nparams,self.nparams)
        C_HPA_max = np.zeros(self.nparams,self.nparams)

        for i in self.nfuncs:
            C_HPA_max = np.outer(j,j)/nHPAMax
        for j in jac_P_ext:
            C_P_ext = np.outer(j,j)/nPext5hrs
        for j in jac_G_ext:
            C_G_ext = np.outer(j,j)/nGext5hrs
        np.sum(outer_prod_jacs)

        return results



def main(maxN = 100, params_sens_list = ['km','kc'],
         params_values_fixed = {'KmDhaTH': 0.77, # mM
                                'KmDhaTN': 0.03, # mM
                                'kcatfDhaT': 59.4, # /seconds
                                'kcatfDhaB':400, # /seconds Input
                                'KmDhaBG': 0.6, # mM Input
                                'dPacking': 0.64,
                                'nmcps': 10,
                                'enz_ratio': 1/1.33},
        bounds = [[10**-7,10**-5],[10**-7,10**-5]], ds = "log10"):

    maxN = int(maxN)
    bounds = np.array(bounds)

    # log transform parameters in params_values_fixed
    for key in params_values_fixed.keys():
        if ds == "log2":
            params_values_fixed[key] = np.log2(params_values_fixed[key])
        elif ds == "log10":
            params_values_fixed[key] = np.log10(params_values_fixed[key])

    # senstivity parameters & transformed parameter bounds
    params_sens_list = ['km','kc']

    transformed_bounds = {}
    for param,bound in param_bounds.items():
        if ds == "log10":
            if param in ['km','kc']:
                transformed_bounds[param] = np.log10(bound)
            transformed_bounds[param] = np.log10(bound)

            elif ds == "log2":
                #TODO: implement this transformation
                print("TODO")
                return






    # init_conditions = { 'GInit': 200, #  2 * 10^(-4) mol/cm3 = 200 mM. 
    #                   'NInit': 1., # mM
    #                   'DInit': 1. # mM
    #                   }


    # # compute non-dimensional scaling
    # params_sens_dict = create_param_symbols('km','kc')

    # #create lambda functions for distributions that returns a list



    # w,v = np.linalg.eigh(cost_mat)

    # ########################### Solving with parameter set sample #######################
    # if rank == 0:
    #     parent_folder_name = ''.join(name + '_' for name in params_sens_dict.keys())[:-1]
    #     child_folder_name = 'maxN_' + str(maxN)
    #     folder_path = parent_folder_name + '/' + child_folder_name

    #     # create folders
    #     if not os.path.exists(folder_path):
    #         os.makedirs(folder_path)
    #     if not os.path.exists(folder_path + "/cost_mat"):
    #         os.makedirs(folder_path + "/cost_mat")
    #     if not os.path.exists(folder_path + "/eigenvectors"):
    #         os.makedirs(folder_path + "/eigenvectors")
    #     if not os.path.exists(folder_path + "/eigenvalues"):
    #         os.makedirs(folder_path + "/eigenvalues")

    #     # save files
    #     bounds_names = ''.join(name + "_" + "{:.2e}".format(bd[0]) + "_" + "{:.2e}".format(bd[1]) + "_" for name,bd in zip(params_sens_dict.keys(),bounds))[:-1]
    #     with open(folder_path + "/cost_mat/" + bounds_names + ".txt", 'w') as outfile:
    #         for slice_2d in cost_mat:
    #             np.savetxt(outfile, slice_2d)
    #     with open(folder_path + "/eigenvectors/" + bounds_names + ".txt", 'w') as outfile:
    #         for slice_2d in v:
    #             np.savetxt(outfile, slice_2d)
    #     np.savetxt(folder_path + "/eigenvalues/" + bounds_names + ".txt", w, delimiter=",")

if __name__ == '__main__':
    main(*sys.argv[1:])


