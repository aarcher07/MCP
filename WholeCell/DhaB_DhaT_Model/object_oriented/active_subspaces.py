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
mpl.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}'] #for \text command
import warnings
import sympy as sp
import scipy.sparse as sparse
import os
import sys
import pickle
from skopt.space import Space
from dhaB_dhaT_model_jac import *
from active_subspaces_dhaT_dhaB_model import *
from skopt.sampler import Lhs
from skopt.space import Space

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

FUNCS_TO_NAMES = {'maximum concentration of 3-HPA': r'$\max_{t}\text{3-HPA}(t;\vec{p})$',
                  'Glycerol concentration after 5 hours': r'$\text{Glycerol}(5\text{ hrs}; \vec{p})$',
                  '1,3-PDO concentration after 5 hours': r'$\text{1,3-PDO}(5\text{ hrs}; \vec{p})$'}
FUNCS_TO_FILENAMES = {'maximum concentration of 3-HPA': 'max3HPA',
                     'Glycerol concentration after 5 hours': 'G5hrs',
                     '1,3-PDO concentration after 5 hours': 'P5hrs'}
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
        self.sample_space = Space([(-1,1) for _ in range(self.nparams)])
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
            x = sobol.generate(self.sample_space.dimensions, niters_rank)

        for sample in param_samples_unorganized:  
            try:
                param_samples_rank.append(sample)
                jac_list_rank.append(self.jac(sample))
            except ValueError:
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
                for jac_est in jac_list_cleaned_reordered[i]:
                    cost_matrix[i] += np.outer(jac_est,jac_est)/nfuncs_successes[i]

            # compute cost matrix
            variance_matrix = [np.zeros((self.nparams,self.nparams)) for _ in range(self.nfuncs)]
            for i in range(self.nfuncs):
                for jac_est in jac_list_cleaned_reordered[i]:
                    variance_matrix[i] += (np.outer(jac_est,jac_est)-cost_matrix[i])**2/(nfuncs_successes[i]-1)            
            
            param_results = [param_samples_flattened,param_samples_diff_int_flattened]
            fun_results = [nfuncs_successes, jac_list_cleaned_reordered, variance_matrix, cost_matrix]

            return {'parameter results': param_results, 'function results': fun_results}

def eig_plots(eigenvalues,eigenvectors,param_sens_bounds,sampling,func_name,enz_ratio_name,niters,date_string):
    #create folder name
    params_names = param_sens_bounds.keys()
    folder = "".join([key + '_' +str(val) + '_' for key,val in param_sens_bounds.items()])[:-1]
    folder = folder.replace(']','')
    folder = folder.replace('[','')
    folder = folder.replace('.',',')
    folder = folder.replace(' ','_')
    folder = folder.replace(',_','_')
    folder_name = os.path.abspath(os.getcwd())+ '/plot/'+ enz_ratio_name+ '/'+ folder
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)


    # file name
    file_name = "sampling_"+ sampling + "_" + FUNCS_TO_FILENAMES[func_name] + '_N_' + str(niters) + '_enzratio_' + enz_ratio_name+ '_'+ date_string + '.png'

    # eigenvalue plot
    grad_name = r'$E[\nabla_{\vec{\widetilde{p}}}$'+ FUNCS_TO_NAMES[func_name] +r'$(\nabla_{\vec{\widetilde{p}}}$' + FUNCS_TO_NAMES[func_name] +r'$^{\top}]$'
    x_axis_labels = [ r"$\lambda_{" + str(i+1) + "}'$" for i in range(len(eigenvalues))]

    plt.yticks(fontsize= 20)
    plt.ylabel(r'$\log_{10}(\lambda_i)$',fontsize= 20)
    plt.bar(range(len(eigenvalues)),np.log10(eigenvalues))
    plt.xticks(list(range(len(eigenvalues))),x_axis_labels,fontsize= 20)
    plt.title(r'$\log_{10}$' + ' plot of the estimated eigenvalues of \n' + grad_name + ' given an enzyme ratio, ' + enz_ratio_name,fontsize= 20)
    plt.savefig(folder_name+'/eigenvalues_'+file_name,bbox_inches='tight')
    plt.close()
    # eigenvector plot
    var_tex_names = [VARS_TO_TEX[params] for params in params_names]
    eigenvec_names = [r"$\vec{v}'_{" + str(i+1) + "}$" for i in range(len(eigenvalues))]
    rescaled = eigenvectors/np.max(np.abs(eigenvectors))
    threshold = 0
    thresholded = np.multiply(rescaled,(np.abs(rescaled) > threshold).astype(float))
    plt.imshow(thresholded,cmap='Greys')
    plt.xticks(list(range(len(eigenvalues))),eigenvec_names,fontsize= 20)
    plt.yticks(list(range(len(eigenvalues))),var_tex_names,fontsize= 20)
    plt.title('Heat map of estimated eigenvectors of \n' + grad_name +', given an enzyme ratio, ' + enz_ratio_name,fontsize= 20)
    cbar = plt.colorbar()
    cbar.ax.tick_params(labelsize=20)

    plt.savefig(folder_name+'/eigenvectors_thres'+ str(threshold).replace('.',',') + '_' + file_name,bbox_inches='tight')
    plt.close()

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

    # initialize variables
    ds = ''
    start_time = (10**(-15))
    final_time = 100*HRS_TO_SECS
    integration_tol = 1e-3
    nsamples = 500
    tolsolve = 10**-10
    sampling = 'rsampling'
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
                        'km': np.log10([10**-3,10**2]), 
                        'kc': np.log10([10**-7,10**-2]),
                        'dPacking': [0.3,0.64],
                        'nmcps': [3.,30.]}

    dhaB_dhaT_model_jacobian_as = DhaBDhaTModelJacAS(start_time, final_time, integration_tol, nsamples, tolsolve,
                                                params_values_fixed,param_sens_bounds, ds = ds)
    def dhaB_dhaT_jac(runif):
        param_sens_dict = {param_name: val for param_name,val in zip(param_sens_bounds.keys(),runif)}
        return dhaB_dhaT_model_jacobian_as.jac_subset(param_sens_dict) 

    as_dhaB_dhaT_mod = ActiveSubspaces(dhaB_dhaT_jac, 3, len(param_sens_bounds),niters=niters, sampling = sampling)

    start_time = time.time()
    results = as_dhaB_dhaT_mod.compute_cost_matrix()
    end_time = time.time()

    if rank == 0:
        param_results = results["parameter results"]
        fun_results = results["function results"]

        date_string = time.strftime("%Y_%m_%d_%H:%M")

        # create folder
        params_names = param_sens_bounds.keys()
        folder = "".join([key + '_' +str(val) + '_' for key,val in param_sens_bounds.items()])[:-1]
        folder = folder.replace(']','')
        folder = folder.replace('[','')
        folder = folder.replace('.',',')
        folder = folder.replace(' ','_')
        folder = folder.replace(',_','_')
        folder_name = os.path.abspath(os.getcwd()) + '/data/' + enz_ratio_name+ '/'+  folder
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)

        # store results
        file_name_pickle = folder_name + '/sampling_' + sampling + '_N_' + str(niters) + '_enzratio_' + enz_ratio_name+ '_'+ date_string + '.pkl'
        with open(file_name_pickle, 'wb') as f:
            pickle.dump(results, f)
 
        file_name_txt = folder_name + '/sampling_' + sampling + '_N_' + str(niters) + '_enzratio_' + enz_ratio_name+ '_'+ date_string + '.txt'
        original_stdout = sys.stdout
        with open(file_name_txt, 'w') as f:
            sys.stdout = f
            print('solve time: ' + str(end_time-start_time))
            print('\n number of processors: ' + str(size))

            print('\n number of functions evaluations')
            print(fun_results[0])

            print('\n variance')
            print(fun_results[-2])

            print('\n max variance')
            print([np_arr.max() for np_arr in fun_results[-2]])

            print('\n cost matrix')
            print(fun_results[-1])

            print('\n number of difficult parameters')
            print(len(param_results[-1]))

            print('\n difficult parameters')
            print(param_results[-1])
            
            sys.stdout = original_stdout

        for i,func_name in enumerate(QOI_NAMES):
            eigs, eigvals = np.linalg.eigh(fun_results[-1][i])
            eigs = np.flip(eigs)
            eigsvals = np.flip(eigvals, axis=1)
            eig_plots(eigs, eigvals,param_sens_bounds,sampling,func_name,enz_ratio_name,niters,date_string)
if __name__ == '__main__':
    dhaB_dhaT_model(sys.argv, len(sys.argv))

