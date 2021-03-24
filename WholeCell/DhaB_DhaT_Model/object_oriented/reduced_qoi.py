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
from constants import QOI_NAMES
from misc import generate_folder_name
from active_subspaces import FUNCS_TO_FILENAMES,FUNCS_TO_NAMES
import seaborn as sns
from sklearn.utils import resample
class ReducedQoI(QoI):
    def __init__(self, directory, pickle_name, eig_ind, nzsamples, start_time,final_time,
                integration_tol, nintegration_samples, tolsolve, params_values_fixed,
                param_sens_bounds, filename = None, external_volume = 9e-6, 
                rc = 0.375e-6, lc = 2.47e-6, rm = 7.e-8, 
                ncells_per_metrecubed =8e14, cellular_geometry = "rod", 
                ds = ""):


            self.directory = directory

            if filename:
                self.folder_name = generate_folder_name(param_sens_bounds)
            else:
                self.folder_name = folder_name

            with open(directory + '/'+ filename+'/' +pickle_name + '.pkl', 'rb') as f:
                pk_as = pickle.load(f)
            cost_matrices = pk_as['function results'][-1]
            self.eig_ind = eig_ind
            self.x_dim = len(param_sens_bounds.keys())
            self.y_dims = [ind for ind in eig_ind]
            self.z_dims = [self.x_dim - ind for ind in eig_ind]
            self.nzsamples = nzsamples
            super().__init__(cost_matrices, start_time,final_time, integration_tol, 
                            nintegration_samples, tolsolve, params_values_fixed,
                            param_sens_bounds, external_volume, rc, lc, rm, 
                            ncells_per_metrecubed, cellular_geometry, ds)

    def _partition_eigenvectors(self):
        W1 = {}
        W2 = {}
        for i,func_name in QOI_NAMES:
            eigvecs = self.eigenvectors_QoI[func_name]
            W1[func_name] = eigvecs[:,:eig_ind[i]]
            W2[func_name] = eigvecs[:,eig_ind[i]:]
        self.W1 = W1
        self.W2 = W2


    def _generate_optimization_parameters(self):
        max_c= {}
        min_c =[]

        #first constraint
        A1 = np.concatenate((-np.ones((z_dim,1)),np.eye(z_dim)),axis=1)
        ub_constraint1 = np.zeros((z_dim))

        #2nd linear constraint
        A2 = {}
        for i,func_name in QOI_NAMES:
            W1 = self.W1[func_name]
            W2 = self.W2[func_name]
            x_dim = self.x_dim
            z_dim = self.z_dims[i]
            A2[func_name] = np.concatenate((np.zeros((x_dim,1)),W2),axis=1)
            
            # cost functions
            max_c[func_name] = np.zeros(z_dim+1)
            max_c[func_name][0] = -1
            min_c[func_name] = np.zeros(z_dim+1)
            min_c[func_name][0] = 1

        self.max_c = max_c
        self.min_c = min_c
        self.A1 = A1
        self.ub_constraint1 = ub_constraint1
        self.A2 = A2

    def _generate_single_z_bound(self,j,func_name,lb_constraint2,
                                 ub_constraint2):


        #set equality constraint
        A1_eq = np.array([self.A1[func_name][j,:]])
        b1_eq = self.ub_constraint1[func_name][j]
        #set inequality constraint for max decision variable
        A1_ineq = np.delete(self.A1[func_name],j,axis=0)
        ub1_ineq = np.delete(self.ub_constraint1[func_name],j,axis=0)

        # final constraints for max problem
        A_ineq = np.concatenate((A1_ineq, self.A2[func_name], -self.A2[func_name]))
        b_ineq = np.concatenate((ub1_ineq,ub_constraint2,lb_constraint2))
        bounds_var = [(None,None) for _ in range(z_dim+1)]

        # find max
        res_max = opt.linprog( self.max_c[func_name], A_ub=A_ineq, b_ub=b_ineq,
                               A_eq=A1_eq, b_eq=b1_eq, bounds=bounds_var)

        # final constraints for min problem
        A_ineq = np.concatenate((-A1_ineq, self.A2[func_name], -self.A2[func_name]))
        b_ineq = np.concatenate((ub1_ineq,ub_constraint2,lb_constraint2))

        # find min
        res_min = opt.linprog( self.min_c[func_name], A_ub=A_ineq, b_ub=b_ineq,
                                A_eq=A1_eq, b_eq=b1_eq, bounds=bounds_var)

        return [res_min.fun, res_max.fun]

    def _generate_z_bounds(self,y):
        x_dim = self.x_dim
        z_given_y_bounds = {} 
        for i,func_name in QOI_NAMES:
            # grab eigenvectors
            W1 = self.W1[func_name]
            W2 = self.W2[func_name]

            z_dim = self.z_dims[i]
            y_in_x = np.dot(W1,y)
            # 2nd linear constraint
            A2 = self.A2[func_name]
            ub_constraint2 =  np.ones(x_dim)- y_in_x
            lb_constraint2 = np.ones(x_dim) + y_in_x

            # cost functions
            max_c{i}
            max_c{i}
            # find the max edges of convex space
            max_edge = -np.inf
            for j in range(z_dim):
                pot_max, pot_min = self._generate_single_z_bound(self,j,func_name, lb_constraint2, ub_constraint2)
                if max_edge < -pot_max:
                    max_edge = -pot_max
                if  pot_min < min_edge:
                    min_edge = pot_min

            z_given_y_bounds[func_name] = [max_edge,min_edge]
        return z_given_y_bounds

    def _z_sampler(self,y):
        #generate bounds of hypercube that contains z|y
        z_given_y_bounds = self._generate_z_bounds(self,y)
        z_samples = {}
        for i,func_name in QOI_NAMES:
            z_dim = self.z_dims[func_name]
            max_edge,min_edge = z_given_y_bounds[func_name]
            z_samples = []

            # create bounds for z space
            A2 = self.A2[func_name]
            y_in_x = np.dot(W1,y)
            ub_constraint2 = np.ones(x_dim)- y_in_x
            lb_constraint2 = np.ones(x_dim) + y_in_x

            z_ineqs_mat = np.concatenate((A2[:,1:], -A2[:,1:]))
            z_ineqs_b = np.concatenate((ub_constraint2,lb_constraint2))
                        
            while(len(z_samples) != self.nzsamples):
                z_pot_sample = (max_edge-min_edge)*np.random.uniform(size= z_dim) + min_edge
                if (np.dot(z_ineqs_mat,z_pot_sample) - z_ineqs_b <= 0 ).all():
                        z_samples.append(z_pot_sample)

        z_samples[func_name] = z_samples

    def generate_reduced_QoI_vals(self,y,gen_histogram=False, save = True):
        z_samples = self._z_sampler(y)

        red_qoi_sample_vals = {QOI_NAMES[0]: [[]],
                               QOI_NAMES[1]: [[]],
                               QOI_NAMES[2]: [[]]}
        for i,func_name in QOI_NAMES:
            n_sucessful_solves = 0
            for z in z_samples[func_name]:
                x = np.dot(W1,y) + np.dot(W2,z)
                param_dict = {param_name:param_val for param_name, param_val  in zip(self.params_sens_list,x)}
                sol_vals = self.generate_QoI_vals(param_dict)
                if sol_vals[func_name]:
                    n_sucessful_solves += 1
                    red_qoi_sample_vals[func_name][0].append(sol_vals[func_name])

        if gen_histogram:
            self._generate_histogram(red_qoi_sample_vals)
        red_qoi_sample_vals = [key:np.mean(qoi_vals[0]) for func_name,qoi_vals in red_qoi_sample_vals.items()]   
        return red_qoi_sample_vals

    def generate_resampled_reduced_QoI_vals(self,y,nbootstrap,random_state= None,
                                            gen_histogram = False,
                                            save = True):
        n_iters_bootstrap = 0
        red_qoi_resampled_vals = {QOI_NAMES[0]: [],
                                  QOI_NAMES[1]: [],
                                  QOI_NAMES[2]: []}
        red_qoi_sample_vals = generate_reduced_QoI_vals(y)
        while(n_iters_bootstrap < nbootstrap):
            for i,func_name in QOI_NAMES:
                red_qoi_resampled_vals[func_name].append(resample(red_qoi_sample_vals[func_name],
                                                         random_state= None))
        if gen_histogram:
            self._generate_histogram(red_qoi_resampled_vals)
        return red_qoi_resampled_vals


    def _generate_histogram(self,red_qoi_vals,save = True):
        for func_name,qoi_vals in red_qoi_resampled_vals.items():
            if len(qoi_vals) <= 5:
                indices = range(len(qoi_vals))
            else:
                indices = np.random.choice(range(len(qoi_vals)),replace=False,size=5)
            for ind in indices: 

            #CONTINUE FROM HERE. 

cost_fun_3HPA = pk_as['function results'][-1][0]


eigvals, eigvecs = np.linalg.eigh(cost_fun_3HPA)
eigvals = np.flip(eigvals)
eigvecs = np.flip(eigvecs, axis=1)


print(eigvals)
print(100*np.cumsum(eigvals)/eigvals.sum())

eig_ind =3
W1 = eigvecs[:,:eig_ind]
W2 = eigvecs[:,eig_ind:]

x = np.array([0.5,0.7,0.2,-0.5,-0.75,
              0.1,-0.3,0.25,0.45,-0.6,
              0.2])
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
    if max_edge < -res_max.fun:
        max_edge = -res_max.fun

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
    if  res_min.fun < min_edge:
        min_edge = res_min.fun

# sample z
print(max_edge)
print(min_edge)

z_ineqs_mat = np.concatenate((A2[:,1:], -A2[:,1:]))
z_ineqs_b = np.concatenate((ub_constraint2,lb_constraint2))
i=0

# create model 
ds = ''
start_time = (10**(-15))
final_time = 72*HRS_TO_SECS
integration_tol = 1e-4
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



dhaB_dhaT_model_jacobian = DhaBDhaTModelJacAS(start_time, final_time, integration_tol, nsamples, tolsolve,
                                              params_values_fixed,param_sens_bounds, ds = ds)


j = 0
def max_3hpa(z):

    # get uniform parameters
    params_unif = np.dot(W1,y) + np.dot(W2,z)
    params_unif_dict = {key:val for key,val in zip(param_sens_bounds.keys(), params_unif)}

    # set differential equation and jacobian
    sdev = lambda t,x: dhaB_dhaT_model_jacobian._sderiv(t,x,params_unif_dict)
    sdev_jac  = lambda t,x: dhaB_dhaT_model_jacobian.sderiv_jac_state_vars_sp_fun(t,x,params_unif_dict)
    y0 = np.array(dhaB_dhaT_model_jacobian.y0(**params_unif_dict))

    # event function

    tolsolve = dhaB_dhaT_model_jacobian.tolsolve
    def event_stop(t,y):
        dSsample = np.array(sdev(t,y[:dhaB_dhaT_model_jacobian.nvars]))
        dSsample_dot = np.abs(dSsample).sum()
        return dSsample_dot - tolsolve 
    event_stop.terminal = True

    # solve ODE
    try:
        sol = solve_ivp(sdev,[0, final_time+1], y0, method="BDF",jac=sdev_jac, 
            t_eval=dhaB_dhaT_model_jacobian.time_orig, atol=dhaB_dhaT_model_jacobian.integration_tol,
            rtol=dhaB_dhaT_model_jacobian.integration_tol, events=event_stop)
    except:
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

        ext_masses_org = y0[-3:]* external_volume
        cell_masses_org = y0[5:8] * volcell 
        mcp_masses_org = y0[:5] * volmcp

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
 
