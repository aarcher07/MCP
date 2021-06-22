"""
Misc functions 

Programme written by aarcher07
Editing History:
- 1/3/21
"""

from numpy.linalg import LinAlgError
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}' #for \text command
import os
import sys
from constants import *
import pickle

def generate_folder_name(param_sens_bounds):
    params_names = param_sens_bounds.keys()
    folder = "".join([key + '_' +str(val) + '_' for key,val in param_sens_bounds.items()])[:-1]
    folder = folder.replace(']','')
    folder = folder.replace('[','')
    folder = folder.replace('.',',')
    folder = folder.replace(' ','_')
    folder = folder.replace(',_','_')
    return folder

def eig_plots(eigenvalues,eigenvectors,params_names,folder,
              sampling,func_name,enz_ratio_name,niters,
              threshold = 0.1, save=True):
    """
    Plots eigenvalues and eigenvectors
    :eigenvalues      : eigenvalues to be plot in ascending order
    :eigenvectors     : associated eigenvectors, ordered by eigenvalues, to be ploted
    :params_names: list of parameter names
    :sampling         : sampling used
    :func_name        : QoI name
    :enz_ratio_name   : enzyme ratio used
    :niters           : number of iterations used in the Monte Carlo
    :threshold        : threshold to ignore entries of eigenvector
    """

    #create folder name
    folder_name = os.path.abspath(os.getcwd())+ '/plot/'+ enz_ratio_name+ '/'+ folder
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    # file name
    file_name = "sampling_"+ sampling + "_" + FUNCS_TO_FILENAMES[func_name] + '_N_' + str(niters) + '.png'

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

def generate_eig_plots_QoI(cost_matrices,param_names,folder,sampling,
                           enz_ratio_name,niters,threshold, save=True):
    """
    plots the eigenspace of the cost matrices of each QoI

    :cost_matrices    : list of numpy matrices for each QoI
    :eigenvectors     : associated eigenvectors, ordered by eigenvalues, to be ploted
    :param_names: list of parameter names
    :sampling         : sampling used
    :func_name        : QoI name
    :enz_ratio_name   : enzyme ratio used
    :niters           : number of iterations used in the Monte Carlo
    :threshold        : threshold to ignore entries of eigenvector
    """
    for func_name in QOI_NAMES:
       eigs, eigvals = np.linalg.eigh(cost_matrices[func_name])
       eigs = np.flip(eigs)
       eigsvals = np.flip(eigvals, axis=1)
       eig_plots(eigs, eigvals,param_names,folder,sampling,
                 func_name,enz_ratio_name,niters,threshold, save)


def generate_txt_output(cost_matrices, nfunction_evals, variance, difficult_params,
                        folder_name, transform, param_sens_bounds, size, sampling,
                        enz_ratio_name, niters,start_time,end_time):
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
    :start_time       : start time of code to complete Monte Carlo integration
    :end_time         : end time of the code to complete Monte Carlo integration
    """

    file_name_txt = folder_name + '/sampling_' + sampling + '_N_' + str(niters) + '.txt'
    original_stdout = sys.stdout
    with open(file_name_txt, 'w') as f:
        sys.stdout = f
        print('transform : ' + transform )
        print(param_sens_bounds)

        print('\n solve time: ' + str(end_time-start_time))
        print('\n number of processors: ' + str(size))

        print('\n number of functions evaluations')
        print(nfunction_evals)

        print('\n variance')
        print(variance)

        print('\n max variance')
        print([np.abs(np_arr).max() for np_arr in variance.values()])

        print('\n cost matrix')
        print(cost_matrices)

        print('\n number of difficult parameters')
        print(len(difficult_params))

        print('\n difficult parameters')
        print(difficult_params)
        
        sys.stdout = original_stdout

def unif_param_to_transform_params(params_sens,transform):
    params = {}
    if transform == "mixed":
        param_bounds = PARAM_SENS_MIXED_BOUNDS
    elif transform == "identity":
        param_bounds = PARAM_SENS_BOUNDS
    elif transform == "log2":
        param_bounds = PARAM_SENS_LOG2_BOUNDS
    elif transform == "log10":
        param_bounds = PARAM_SENS_LOG10_BOUNDS
        
    for param_name, param_val in params_sens.items():
        bound_a, bound_b = param_bounds[param_name]
        param_trans = ((bound_b - bound_a)*param_val/2 + (bound_a + bound_b)/2) 
        params[param_name] = param_trans
    return params

def load_obj(name):
    """
    Load a pickle file. Taken from
    https://stackoverflow.com/questions/19201290/how-to-save-a-dictionary-to-a-file
    :param name: Name of file
    :return: the file inside the pickle
    """
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)


def save_obj(obj, name):
    """
    Save a pickle file. Taken from
    https://stackoverflow.com/questions/19201290/how-to-save-a-dictionary-to-a-file

    :param  obj: object to save
            name: Name of file
    """
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
