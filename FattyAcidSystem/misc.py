"""
Misc functions 

Programme written by aarcher07
Editing History:
- 1/3/21
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
from constants import *

def generate_folder_name(param_sens_bounds):
    params_names = param_sens_bounds.keys()
    folder = "".join([key + '_' +str(val) + '_' for key,val in param_sens_bounds.items()])[:-1]
    folder = folder.replace(']','')
    folder = folder.replace('[','')
    folder = folder.replace('.',',')
    folder = folder.replace(' ','_')
    folder = folder.replace(',_','_')
    return folder

def eig_plots(eigenvalues,eigenvectors,param_sens_bounds,
              sampling,func_name,enz_ratio_name,niters,date_string,
              threshold = 0.1, save=True):
    """
    Plots eigenvalues and eigenvectors
    :eigenvalues      : eigenvalues to be plot in ascending order
    :eigenvectors     : associated eigenvectors, ordered by eigenvalues, to be ploted
    :param_sens_bounds: dictionary of parameter ranges
    :sampling         : sampling used
    :func_name        : QoI name
    :enz_ratio_name   : enzyme ratio used
    :niters           : number of iterations used in the Monte Carlo
    :date_string      : the date string attached
    :threshold        : threshold to ignore entries of eigenvector
    """

    #create folder name
    params_names = param_sens_bounds.keys()
    folder = generate_folder_name(param_sens_bounds)
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
    if save:
        plt.savefig(folder_name+'/eigenvalues_'+file_name,bbox_inches='tight')
        plt.close()
    else:
        plt.show()
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
    if save:
        plt.savefig(folder_name+'/eigenvectors_thres'+ str(threshold).replace('.',',') + '_' + file_name,bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def generate_eig_plots_QoI(cost_matrices,param_sens_bounds,sampling,
                           enz_ratio_name,niters,date_string,threshold,
                           save=True):
    """
    plots the eigenspace of the cost matrices of each QoI

    :cost_matrices    : list of numpy matrices for each QoI
    :eigenvectors     : associated eigenvectors, ordered by eigenvalues, to be ploted
    :param_sens_bounds: dictionary of parameter ranges
    :sampling         : sampling used
    :func_name        : QoI name
    :enz_ratio_name   : enzyme ratio used
    :niters           : number of iterations used in the Monte Carlo
    :date_string      : the date string attached
    :threshold        : threshold to ignore entries of eigenvector
    """
    for i,func_name in enumerate(QOI_NAMES):
       eigs, eigvals = np.linalg.eigh(cost_matrices[i])
       eigs = np.flip(eigs)
       eigsvals = np.flip(eigvals, axis=1)
       eig_plots(eigs, eigvals,param_sens_bounds,sampling,
                 func_name,enz_ratio_name,niters,date_string,
                 threshold, save)


def generate_txt_output(cost_matrices, nfunction_evals, variance, difficult_params,
                        folder_name, param_sens_bounds, size, sampling,enz_ratio_name,
                        niters,date_string,start_time,end_time):
    """
    creates and saves a text file of output

    :cost_matrices    : list of numpy cost matrices for each QoI
    :nfunction_evals  : the number of function evaluations used to generate each cost matrix
    :variance         : variance of the random variable used to generate each entry of each cost matrix
    :difficult_params : parameter samples that were difficult to integrate
    :folder_name      : QoI name
    :param_sens_bounds: dictionary of parameter ranges
    :size             : number of processors
    :sampling         : sampling used
    :enz_ratio_name   : enzyme ratio used
    :niters           : number of iterations used in the Monte Carlo
    :date_string      : the date string attached
    :start_time       : start time of code to complete Monte Carlo integration
    :end_time         : end time of the code to complete Monte Carlo integration
    """

    file_name_txt = folder_name + '/sampling_' + sampling + '_N_' + str(niters) + '_enzratio_' + enz_ratio_name+ '_'+ date_string + '.txt'
    original_stdout = sys.stdout
    with open(file_name_txt, 'w') as f:
        sys.stdout = f
        print('solve time: ' + str(end_time-start_time))
        print('\n number of processors: ' + str(size))

        print('\n number of functions evaluations')
        print(nfunction_evals)

        print('\n variance')
        print(variance)

        print('\n max variance')
        print([np.abs(np_arr).max() for np_arr in variance[-2]])

        print('\n cost matrix')
        print(cost_matrices)

        print('\n number of difficult parameters')
        print(len(difficult_params))

        print('\n difficult parameters')
        print(difficult_params)
        
        sys.stdout = original_stdout