"""
Parallelizes the Active_Subspaces.py code. This code generates the
average parameter directions that most affects the model in a bounded region
of parameter space.

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

from DhaB_DhaT_Model import *
from DhaB_DhaT_Model_LocalSensAnalysis import *

HRS_TO_SECS = 60*60

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

class Active_Subspaces:
    def __init__(self,jac, nfuns, nparams, names, trans_func, max_iters,
                 dist, output_level, max_exceptions, tol=1e-5):
        """
        :params jac:
        :params nfuns:
        :params names:
        :params trans_func:
        :params max_iters:
        :params dist:
        :params output_level:
        :params tol:
        :params external_volume:
        :params rc:
        :params lc:
        :params rm:
        :params ncells_per_metrecubed:
        :params cellular_geometry:
        :params ds:      
        """
        self.jac = jac
        self.nfuns = nfuns
        self.nparams = nparams
        self.names = names
        self.bounds = bounds
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
        if rank == 0:
            output_header = '%12s %30s %18s' % (' iteration ', ' max ||Delta C||_F/||C_{i}||_F ',
                                                ' Num of Value Error ')
            print(output_header)
            status = -99

        while 1:
            # updates
            x = # sample transformed unit form distribution
            dict_vals = {name:x for name,val in zip(names,x)}
            j = jac(dict_vals)
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

            if (rank == 0) and (iters % 5 == 0) and (self.output_level == 3):
                output = '%10i %22.4e %22i' % (iters, max_fro_ratio, n_exceptions)
                print(output)

            if (rank == 0) and (iters % 100 == 0) and (self.output_level == 3):
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

        return [param_samples,params_samples_higher_tol,(1/iters)*C]

class dhaB_dhaT_model_local_sens_analysis_extension(dhaB_dhaT_model_local_sens_analysis):

    def __init__(self,start_time,final_time, init_conc, init_sens,
                integration_tol, nsamples, params_values_fixed,
                params_sens_list, external_volume = 9e-6, 
                rc = 0.375e-6, lc = 2.47e-6, rm = 7.e-8, 
                ncells_per_metrecubed =8e14, cellular_geometry = "rod", 
                ds = "log2"):

        """
        :params start_time:
        :params final_time:
        :params init_conc:
        :params init_sens:
        :params integration_tol:
        :params nsamples:
        :params params_values_fixed:
        :params params_sens_list:
        :params external_volume:
        :params rc:
        :params lc:
        :params rm:
        :params ncells_per_metrecubed:
        :params cellular_geometry:
        :params ds:      
        """
        super().__init__(self,params, params_sens_list, external_volume, rc,
                        lc , rm , ncells_per_metrecubed, cellular_geometry, ds)

        # save integration parameters
        self.y0 = init_conc
        self.sens0 = init_sens
        self.xs0 = np.concatenate([init_conc,init_sens])
        self.start_time = start_time
        self.final_time = final_time
        self.integration_tol = integration_tol
        self.time_orig_hours = time_orig/HRS_TO_SECS

        # index of HPA senstivities
        self.index_3HPA_cytosol = 6 # index of H_cystosol
        self.range_3HPA_cytosol_params_sens = range(self.nvars+self.index_3HPA_cytosol*self.nparams_sens,
                                                  self.nvars+(self.index_3HPA_cytosol+1)*self.nparams_sens)

        # index of 1,3-PDO after 5 hrs
        time_check = 5.
        self.first_index_close_enough =  np.argmin(np.abs(time_orig_hours-time_check)) # index of closest time after 5hrs
        self.index_1_3PDO_ext = -3 # index of P_ext 
        range_1_3PDO_ext_param_sens = range(index_1_3PDO_ext*self.nparams_sens,
                                           (index_1_3PDO_ext+1)*self.nparams_sens)
        self.indices_1_3PDO_ext_params_sens = np.array([(self.first_index_close_enough,i) for i in range_1_3PDO_ext_param_sens])
        self.indices_sens_1_3PDO_ext_after_timecheck = np.array([(-1,i) for i in range_1_3PDO_ext_param_sens])

        # index of Glycerol after 5 hrs
        self.index_Glycerol_ext = -1 # index of Glycerol_ext 
        range_Glycerol_ext_param_sens = range(index_Glycerol_ext*self.nparams_sens,
                                             (index_Glycerol_ext+1)*self.nparams_sens)
        self.indices_Glycerol_ext_params_sens = np.array([(self.first_index_close_enough,i) for i in range_Glycerol_ext_param_sens])
        self.indices_sens_Glycerol_ext_after_timecheck = np.array([(-1,i) for i in range_1_3PDO_ext_param_sens])

        # save local senstivities analysis, results
        self.set_jacs_fun()
        self.create_jac_sens()

    def jac(params_sens_dict):
        """
        :param dict_vals:
        """
        dsens_param = lambda t, xs: model_local_sens.dsens(t,xs,params_sens_dict)
        dsens_jac_sparse_mat_fun_param = lambda t, xs: model_local_sens.dsens_jac_sparse_mat_fun(t,xs,params_sens_dict)

        timeorig = np.logspace(self.start_time,self.final_time,self.nsamples)
        
        tolsolve = 10**-4
        def event_stop(t,y):
            params = {**params_values_fixed, **params_sens_dict}
            dSsample = np.array(model_local_sens.ds(t,y[:model_local_sens.nvars],params))
            dSsample_dot = np.abs(dSsample).sum()
            return dSsample_dot - tolsolve 
        event_stop.terminal = True

        sol = solve_ivp(dsens_param,[0, fintime+1], xs0, method="BDF",
                        jac = dsens_jac_sparse_mat_fun_param,t_eval=timeorig,
                        atol=self.integration_tol, rtol=self.integration_tol,
                        events=event_stop)

        return [sol.status,sol.t,sol.y.T]


    def jac_subset(params_sens_dict):
        
        status, time, jac_sample = jac(params_sens_dict)
        
        # get sensitivities of max 3-HPA
        index_3HPA_max = np.argmax(jac_sample[:,self.index_3HPA_cytosol]) 
        indices_3HPA_max_cytosol_params_sens =  np.array([[index_3HPA_max,i] for i in self.range_3HPA_cytosol_params_sens])
        jac_HPA_max = jac_sample[tuple(indices_3HPA_max_cytosol_params_sens.T)]

        # integration completed
        if status == 0 or (time[-1] > 5*HRS_TO_SECS):
            jac_P_ext = jac_sample[tuple(self.indices_1_3PDO_ext_params_sens.T)]
            jac_G_ext = jac_sample[tuple(self.indices_Glycerol_ext_params_sens.T)]
        elif status == 1:
            jac_P_ext = jac_sample[tuple(self.indices_sens_1_PDO_ext_after_timecheck.T)]
            jac_G_ext = jac_sample[tuple(self.indices_sens_Glycerol_ext_after_timecheck.T)]
        else:
            return
        return np.array([jac_HPA_max,jac_P_ext,jac_G_ext])

def main(maxN = 100):
    maxN = int(maxN)
    params_values_fixed = {'KmDhaTH': 0.77, # mM
          'KmDhaTN': 0.03, # mM
          'kcatfDhaT': 59.4, # /seconds
          'kcatfDhaB':400, # /seconds Input
          'KmDhaBG': 0.6, # mM Input
          'dPacking': 0.64,
          'nmcps': 10,
          'enz_ratio': 1/1.33}

    # log transform parameters in params_values_fixed
    for key in params_values_fixed.keys():
        params_values_fixed[key] = np.log2(params_values_fixed[key])
        
    params_sens_dict = ['km','kc']


    init_conditions = { 'GInit': 200, #  2 * 10^(-4) mol/cm3 = 200 mM. 
                      'NInit': 1., # mM
                      'DInit': 1. # mM
                      }


    # compute non-dimensional scaling
    params_sens_dict = create_param_symbols('km','kc')

    #create lambda functions for distributions that returns a list



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


