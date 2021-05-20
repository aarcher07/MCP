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
import scipy.optimize as opt
import os
import sys
import pickle
from skopt.space import Space
from dhaB_dhaT_model_jac import *
from active_subspaces_dhaT_dhaB_model import *
from misc import generate_folder_name
from active_subspaces import FUNCS_TO_FILENAMES,FUNCS_TO_NAMES
import seaborn as sns



directory = '/home/aarcher/Dropbox/PycharmProjects/MCP/WholeCell/DhaB_DhaT_Model/object_oriented/data/1:3'
filename = 'log10/2021_05_04_19:41'
#name_pkl = 'sampling_rsampling_N_100000_enzratio_1:18_2021_01_29_20:01'
name_pkl = 'sampling_rsampling_N_10000'

with open(directory + '/'+ filename+'/' +name_pkl + '.pkl', 'rb') as f:
    pk_as = pickle.load(f)
cost_matrices = pk_as["FUNCTION_RESULTS"]["FINAL_COST_MATRIX"]

cost_fun_3HPA = pk_as["FUNCTION_RESULTS"]["FINAL_COST_MATRIX"][QOI_NAMES[0]]


eigvals, eigvecs = np.linalg.eigh(cost_fun_3HPA)
eigvals = np.flip(eigvals)
eigvecs = np.flip(eigvecs, axis=1)


print(eigvals)
print(100*np.cumsum(eigvals)/eigvals.sum())

eig_ind =np.argmax(np.cumsum(eigvals)/eigvals.sum() > 0.9)+1
W1 = eigvecs[:,:eig_ind]
W2 = eigvecs[:,eig_ind:]

x = np.array([0.3084162660909806,-0.49136586580345176,0.5129415947320597,0.3979400086720375,0.7474986422705001,
             0.4961407271748155,0.,0.3979400086720375,0.,0.,0.3979400086720375,0.34838396084692813,0.39794000867203794])
y = np.dot(W1.T,x.T)
z = np.dot(W2.T,x.T)

x_dim = W1.shape[0]
z_dim = W2.shape[1]
x_in_y = np.dot(W1,y)
# 1st linear constraint
A1 = np.concatenate((-np.ones((z_dim,1)),np.eye(z_dim)),axis=1)
ub_constraint1 = np.zeros((z_dim))

# 2nd linear constraint
A2 = np.concatenate((np.zeros((x_dim,1)),W2),axis=1)
ub_constraint2 =  np.ones(x_dim)- x_in_y
lb_constraint2 = np.ones(x_dim) + x_in_y

# cost function
max_c = np.zeros(z_dim+1)
max_c[0] = -1

# find the max edges of convex space
max_edge = -np.inf
for i in range(z_dim):
    #set equality constraint
    A1_eq = np.array([A1[i,:]])
    b1_eq = ub_constraint1[i]
    #set inequality constraint for upper bound
    A1_ineq = np.delete(A1,i,axis=0)
    ub1_ineq = np.delete(ub_constraint1,i,axis=0)

    # final constraints
    A_ineq = np.concatenate((A1_ineq, A2, -A2))
    b_ineq = np.concatenate((ub1_ineq,ub_constraint2,lb_constraint2))
    bounds_var = [(None,None) for _ in range(z_dim+1)]

    res_max = opt.linprog( max_c, A_ub=A_ineq, b_ub=b_ineq, A_eq=A1_eq, b_eq=b1_eq, bounds=bounds_var)
    print(res_max.fun)

    if max_edge < -res_max.fun:
        max_edge = -res_max.fun
print('\t')
# cost function
min_c = np.zeros(z_dim+1)
min_c[0] = 1

# find the min edges of convex space
min_edge = np.inf
for i in range(z_dim):
    #set equality constraint
    A1_eq = np.array([-A1[i,:]])
    b1_eq = ub_constraint1[i]

    #set inequality constraint for upper bound
    A1_ineq = np.delete(-A1,i,axis=0)
    ub1_ineq = np.delete(ub_constraint1,i,axis=0)

    # final constraints
    A_ineq = np.concatenate((A1_ineq, A2, -A2))
    b_ineq = np.concatenate((ub1_ineq,ub_constraint2,lb_constraint2))
    bounds_var = [(None,None) for _ in range(z_dim+1)]

    res_min = opt.linprog( min_c, A_ub=A_ineq, b_ub=b_ineq, A_eq=A1_eq, b_eq=b1_eq, bounds=bounds_var)
    print(res_min.fun)
    if  res_min.fun < min_edge:
        min_edge = res_min.fun

# sample z
print(max_edge)
print(min_edge)

z_ineqs_mat = np.concatenate((A2[:,1:], -A2[:,1:]))
z_ineqs_b = np.concatenate((ub_constraint2,lb_constraint2))
i=0

# create model 
transform = 'log10'
start_time = (10**(-15))
final_time = 72*HRS_TO_SECS
integration_tol = 1e-3
tolsolve = 1e-5
nsamples = 500
enz_ratio_name = "1:3"
enz_ratio_name_split =  enz_ratio_name.split(":")
enz_ratio = float(enz_ratio_name_split[0])/float(enz_ratio_name_split[1])
params_values_fixed = {'NAD_MCP_INIT': 0.1,
                      'enz_ratio': enz_ratio,
                      'G_MCP_INIT': 0,
                      'H_MCP_INIT': 0,
                      'P_MCP_INIT': 0,
                      'G_CYTO_INIT': 0,
                      'H_CYTO_INIT': 0,
                      'P_CYTO_INIT': 0 ,
                      'G_EXT_INIT': 200,
                      'H_EXT_INIT': 0,
                      'P_EXT_INIT': 0}

dhaB_dhaT_model_jacobian = DhaBDhaTModelJacAS(start_time, final_time, integration_tol, nsamples, tolsolve,
                                              params_values_fixed,list(PARAM_SENS_LOG10_BOUNDS.keys()), transform = transform)

j = 0
def max_3hpa(z):

    # get uniform parameters
    params_unif = np.dot(W1,y) + np.dot(W2,z)
    params_unif_dict = {key:val for key,val in zip(PARAM_SENS_LOG10_BOUNDS.keys(), params_unif)}

    # set differential equation and jacobian
    sdev = lambda t,x: dhaB_dhaT_model_jacobian._sderiv(t,x,params_unif_dict)
    sdev_jac  = lambda t,x: dhaB_dhaT_model_jacobian.sderiv_jac_conc_fun(t,x,params_unif_dict.values())
    x0 = np.array(dhaB_dhaT_model_jacobian.x0(**params_unif_dict))
    # event function

    tolsolve = dhaB_dhaT_model_jacobian.tolsolve
    def event_stop(t,y):
        dSsample = np.array(sdev(t,y))
        dSsample_dot = np.abs(dSsample).sum()
        return dSsample_dot - tolsolve 
    event_stop.terminal = True

    # solve ODE
    try:
        sol = solve_ivp(sdev,[0, final_time+1],x0, method="BDF",jac=sdev_jac, 
                t_eval=dhaB_dhaT_model_jacobian.time_orig, atol=dhaB_dhaT_model_jacobian.integration_tol,
                rtol=dhaB_dhaT_model_jacobian.integration_tol, events=event_stop)
    except ValueError:
        return

    status, time, sol_sample = [sol.status,sol.t,sol.y.T]

    # get max 3-HPA
    index_3HPA_max = np.argmax(sol_sample[:,dhaB_dhaT_model_jacobian.index_3HPA_cytosol]) 

    # check if derivative is 0 of 3-HPA otherwise integration stopped prematurely
    statevars_maxabs = sol_sample[index_3HPA_max,:]
    dev_3HPA = sdev(time[index_3HPA_max],statevars_maxabs)[dhaB_dhaT_model_jacobian.index_3HPA_cytosol]
    if abs(dev_3HPA) < 1e-2 and status == 0:
        volcell = dhaB_dhaT_model_jacobian.cell_volume
        volmcp = 4 * np.pi * (dhaB_dhaT_model_jacobian.rm ** 3) / 3
        external_volume = dhaB_dhaT_model_jacobian.external_volume 

        ext_masses_org = x0[-3:]* external_volume
        cell_masses_org = x0[5:8] * volcell 
        mcp_masses_org = x0[:5] * volmcp

        ext_masses_fin = np.abs(sol_sample[-1, -3:]) * external_volume
        cell_masses_fin = np.abs(sol_sample[-1,5:8]) * volcell
        mcp_masses_fin = np.abs(sol_sample[-1,:5]) * volmcp

        org_mass = ext_masses_org.sum() + cell_masses_org.sum() + mcp_masses_org.sum() 
        fin_mass = ext_masses_fin.sum() + cell_masses_fin.sum() + mcp_masses_fin.sum()
        relative_diff = fin_mass/org_mass

        if (relative_diff > 0.5 and relative_diff < 1.5):
            return sol_sample[index_3HPA_max,dhaB_dhaT_model_jacobian.index_3HPA_cytosol]
        # else:
        #     params_unif_trans = {}
        #     for param_name, param_val in zip(param_sens_bounds.keys(),params_unif):
        #         bound_a, bound_b = param_sens_bounds[param_name]
        #         param_trans = ((bound_b - bound_a)*param_val/2 + (bound_a + bound_b)/2) 
        #         if param_name in ['PermMCPPolar','NonPolarBias','PermCell']:
        #             params_unif_trans[param_name] = 10**param_trans
        #         else:
        #             params_unif_trans[param_name] = param_trans


        #     for i in range(0,3):
        #         ycell = sol_sample.T[3+i, :]
        #         plt.plot(time/HRS_TO_SECS,ycell)
        #         plt.title('Plot of cellular concentration')
        #         plt.legend(['Glycerol', '3-HPA', '1,3-PDO'], loc='upper right')
        #         plt.show()



            
# z = np.array([ 0.56125122, -0.41567089,  0.38132884, -0.10070303, -0.24315471, -0.31188756, -0.78052557, -0.55775174, -0.35813561, -0.43232742])
# max_3hpa(z)
M=int(1e2)
cumsums = []
for _ in range(10):
    max_3hpa_samples = []
    z_samples = []

    while(len(z_samples) != M):
        z_pot_sample = (max_edge-min_edge)*np.random.uniform(size= z_dim) + min_edge
        if (np.dot(z_ineqs_mat,z_pot_sample) - z_ineqs_b <= 0 ).all():
            max_3hpa_pot_sample = max_3hpa(z_pot_sample)
            if not max_3hpa_pot_sample is None:
                z_samples.append(z_pot_sample)
                max_3hpa_samples.append(max_3hpa_pot_sample)
            else:
                j += 1
            print(len(max_3hpa_samples))
        i+=1

    cumsums.append(np.divide(np.cumsum(max_3hpa_samples),list(range(1,M+1))))


    sns.histplot(data=max_3hpa_samples,stat='probability', bins='auto', color='#0504aa',alpha=0.7)
    plt.grid(axis='y', alpha=0.75)
    plt.xlabel('3-HPA Toxicity')
    plt.title('Histogram of 3-HPA' + r'$(\vec{p}) = $' + '3-HPA' + r'$(\vec{y},\vec{z})$'
        + ' for a fixed y and '+ r'$M='+str(M) + '$'+ ' samples of z')
    plt.axvline(x=cumsums[-1][-1], color='red',linewidth=4)
    plt.show()

for cumsum in cumsums:
    plt.plot(cumsum)
plt.show()
 
