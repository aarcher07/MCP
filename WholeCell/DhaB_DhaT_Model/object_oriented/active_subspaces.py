"""
Parallelized the Active_Subspaces.py code.

Programme written by aarcher07
Editing History:
- 9/11/20
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

class Active_Subspaces:
    def __init__(self,jac, nfuns, nparams, max_iters, 
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
        :params max_iters      : maximum number of iterations
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
        self.max_iters = max_iters
        self.dist = dist
        self.output_level = output_level
        self.max_exceptions = max_exceptions
        self.tol= tol

    def compute_cost_matrix():
        """
        Monte Carlo integration estimate of the cost function matrix
        """

        fro_ratio = np.Inf
        iters = 0
        C = np.zeros((nfuns,M,M))
        param_samples = []
        params_samples_higher_tol = []
        if rank == 0 and self.output_level == 2:
            output_header = '%12s %30s %18s' % (' iteration ', ' max ||Delta C||_F/||C_{i}||_F ',
                                                ' Num of Value Error ')
            print(output_header)
            status = -99

        while 1:
            # updates
            x = self.dist(M) # sample transformed unit form distribution
            j = self.jac(x) #parallelize
            try: 
                outer_prod = np.array([np.outer(j[i, :],j[i, :]) for i in range(j.shape[0])])
                iters_update = 1
                n_exceptions_update = 0
                param_samples.append(x)

            except (TypeError,LinAlgError) as e:
                outer_prod = np.zeros((nfuns,M,M))
                iters_update = 0
                n_exceptions_update = 1
                params_samples_higher_tol.append(x)

            #reduce statements and ratio computations
            iters += comm.allreduce(iters_update,MPI.SUM)
            n_exceptions += comm.allreduce(n_exceptions_update,MPI.SUM)
            C += comm.allreduce(outer_prod,MPI.SUM)

            fro_outer = np.array([np.linalg.norm(outer_prod[i, :, :], ord='fro') for i in range(nfuns)])
            fro_C = np.array([np.linalg.norm(C[i, :, :], ord='fro') for i in range(nfuns)])

            fro_ratio = np.max(fro_outer / fro_C)
            max_fro_ratio = comm.allreduce(fro_ratio,MPI.MAX)

            if (rank == 0) and (iters % 5 == 0) and (self.output_level == 2):
                output = '%10i %22.4e %22i' % (iters, max_fro_ratio, n_exceptions)
                print(output)

            if (rank == 0) and (iters % 100 == 0) and (self.output_level == 2):
                print(output_header)

            # break statements
            if max_fro_ratio < tol:
                status = 0
                break

            if iters >= self.max_iters_per_processor:
                status = -1
                break

            if n_exceptions >= self.max_exceptions:
                status = -2
                break

        # print return
        if rank == 0:
            # Final output message
            if output_level >= 1:
                print('')
                print('max ||Delta C||_F/||C_{i}||_F ......................: %20.4e' % fro_ratio)
                print('total number of iterations .........................: %d' % iters)
                print('total number of solutions with ValueError .........: %d' % n_exceptions)
                print('')
                if status == 0:
                    print('Exit: Converged within tolerance.')
                elif status == -1:
                    print('Exit: Maximum number of iterations, (%d), exceeded.' %
                          self.max_iters)
                elif status == -2:
                    print('Exit: Maximum number of solutions with ValueError, (%d), exceeded. '
                          'You may need to increase atol and rtol.' %
                          max_exceptions)
                else:
                    print('ERROR: Unknown status value: %d\n' % status)

        results = {'param_samples'            : param_samples            ,
                   'params_samples_higher_tol': params_samples_higher_tol,
                   'cost_matrix'              : (1/iters)*C              }

        self.results = results
        return results

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


