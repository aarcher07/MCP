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
from misc import eig_plots
from constants import QOI_NAMES
from active_subspaces import FUNCS_TO_FILENAMES,FUNCS_TO_NAMES
import seaborn as sns

class QoI(DhaBDhaTModelJacAS):
    def __init__(self, cost_matrices, start_time,final_time,
                integration_tol, nintegration_samples, tolsolve, params_values_fixed,
                param_sens_bounds, external_volume = 9e-6, rc = 0.375e-6, lc = 2.47e-6,
                rm = 7.e-8, ncells_per_metrecubed =8e14, cellular_geometry = "rod", 
                ds = ""):
        """
        :params cost_matrices: cost matrices associated with Active Subspaces
        :params start_time: initial time of the system -- cannot be 0
        :params final_time: final time of the system 
        :params init_conc: inital concentration of the system
        :params integration_tol: integration tolerance
        :params nsamples: number of samples of time samples
        :params params_values_fixed: dictionary parameters whose senstivities are not being studied and 
                                     their values
        :params param_sens_bounds: bounds of parameters whose sensitivities are being studied
        :params external_volume: external volume of the system
        :params rc: radius of system
        :params lc: length of the cylindrical component of cellular_geometry = 'rod'
        :params rm: radius of MCP
        :params ncells_per_metrecubed: number of cells per m^3
        :params cellular_geometry: geometry of the cell, rod (cylinder with hemispherical ends)/sphere
        :params ds: transformation of the parameters, log2, log10 or [-1,1].   
        """

        self.cost_matrices = cost_matrices
        super().__init__(start_time,final_time, integration_tol, nintegration_samples, tolsolve, params_values_fixed,
                param_sens_bounds, external_volume, rc, lc, rm, ncells_per_metrecubed, cellular_geometry, 
                ds)
        self._generate_eigenspace()


    def _generate_eigenspace(self):
        """
        Generate the eigenspace associated with the cost matrix of each QoI
        """

        eigenvalues_QoI = {}
        eigenvectors_QoI = {}
        for i,func_name in enumerate(QOI_NAMES):
            eigs, eigvals = np.linalg.eigh(self.cost_matrices[i])
            eigenvalues_QoI[func_name] = np.flip(eigs)
            eigenvectors_QoI[func_name] = np.flip(eigvals, axis=1)
        self.eigenvalues_QoI = eigenvalues_QoI
        self.eigenvectors_QoI = eigenvectors_QoI



    def generate_QoI_vals(self,params_unif_dict):
        """
        Generate the QoI value at parameter value, params_unif_dict

        :params_unif_dict: dictionary of transformed parameters, log2, log10 or [-1,1]. 
        """

        sdev = lambda t,x: self._sderiv(t,x,params_unif_dict)
        sdev_jac  = lambda t,x: self.sderiv_jac_state_vars_sp_fun(t,x,params_unif_dict)
        y0 = np.array(self.y0(**params_unif_dict))

        # event function
        event_stop = lambda t,y: self._event_stop(t,y,params_unif_dict)
        event_stop.terminal = True

        # initialize
        sol_values = {QOI_NAMES[0]: None,QOI_NAMES[1]: None, QOI_NAMES[2]: None}
        # solve ODE
        try:
            sol = solve_ivp(sdev,[0, self.final_time+1], y0, method="BDF",jac=sdev_jac, 
                            t_eval=self.time_orig, atol=self.integration_tol,
                             rtol=self.integration_tol, events=event_stop)
        except ValueError:
            return sol_values

        status, time, sol_sample = [sol.status,sol.t,sol.y.T]

        # get max 3-HPA
        index_3HPA_max = np.argmax(sol_sample[:,self.index_3HPA_cytosol]) 
        # check if derivative is 0 of 3-HPA 
        statevars_maxabs = sol_sample[index_3HPA_max,:self.nvars]
        dev_3HPA = sdev(time[index_3HPA_max],statevars_maxabs)[self.index_3HPA_cytosol]


        if 'nmcps' in self.params_values_fixed.keys():
            nmcps = self.params_values_fixed['nmcps']
        else:
            bound_mcp_a,bound_mcp_b = self.param_sens_bounds['nmcps']
            nmcps = (params_unif_dict['nmcps'] +1)*(bound_mcp_b - bound_mcp_a)/2. * + bound_mcp_a
        # original mass
        ext_masses_org = y0[(self.nvars-3):self.nvars]* self.external_volume
        cell_masses_org = y0[5:8] * self.cell_volume 
        mcp_masses_org = y0[:5] * self.mcp_volume
        mass_org = ext_masses_org.sum() +  self.ncells*cell_masses_org.sum() +  self.ncells*nmcps*mcp_masses_org.sum()

        # final mass
        ext_masses_fin = sol_sample[-1,(self.nvars-3):self.nvars] * self.external_volume
        cell_masses_fin = sol_sample[-1,5:8] * self.cell_volume
        mcp_masses_fin = sol_sample[-1,:5] * self.mcp_volume
        mass_fin = ext_masses_fin.sum() + self.ncells*cell_masses_fin.sum() + self.ncells*nmcps*mcp_masses_fin.sum()
        relative_diff = mass_fin/mass_org

        # check if integrated correctly
        if (relative_diff > 0.5 and relative_diff < 1.5):

            if abs(dev_3HPA) < 1e-2:
                HPA_max = sol_sample[index_3HPA_max,self.index_3HPA_cytosol]
            else:
                HPA_max = None

            # get sensitivities of Glycerol and 1,3-PDO after 5 hrs
            if status == 0 or (time[-1] > 5*HRS_TO_SECS):
                P_ext = sol_sample[self.first_index_close_enough,self.index_1_3PDO_ext]
                G_ext = sol_sample[self.first_index_close_enough,self.index_Glycerol_ext]
            elif status == 1:
                P_ext = sol_sample[-1,self.index_1_3PDO_ext]
                G_ext = sol_sample[-1,self.index_Glycerol_ext]
            else:
                P_ext = None
                G_ext = None

            sol_values[QOI_NAMES[0]] = HPA_max,
            sol_values[QOI_NAMES[1]] =  G_ext,
            sol_values[QOI_NAMES[2]] =  P_ext
        return sol_values
            

def main(argv, arc):
    # get inputs
    enz_ratio_name = argv[1]

    # initialize variables
    ds = ''
    start_time = (10**(-15))
    final_time = 72*HRS_TO_SECS
    integration_tol = 1e-4
    tolsolve = 1e-5
    nintegration_samples = 500
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

    params_sens_dict  = {'kcatfDhaB': 400, # /seconds Input
                        'KmDhaBG': 0.6, # mM Input
                        'kcatfDhaT': 59.4, # /seconds
                        'KmDhaTH': 0.77, # mM
                        'KmDhaTN': 0.03, # mM
                        'NADH_MCP_INIT': 0.36,
                        'PermMCPPolar': np.log10(10**-3), 
                        'NonPolarBias': np.log10(10**-2), 
                        'PermCell': np.log10(10.**-7),
                        'dPacking': 0.64,
                        'nmcps': 15}

    params_unif = {}
    for param_name, param_val in params_sens_dict.items():
        bound_a,bound_b = param_sens_bounds[param_name]
        params_unif[param_name] = 2*(param_val - bound_a)/(bound_b - bound_a) - 1

    directory = '/home/aarcher/Dropbox/PycharmProjects/MCP/WholeCell/DhaB_DhaT_Model/object_oriented/data/1:3'
    filename = 'kcatfDhaB_400_860_KmDhaBG_0,6_1,1_kcatfDhaT_40,0_100,0_KmDhaTH_0,1_1,0_KmDhaTN_0,0116_0,48_NADH_MCP_INIT_0,12_0,6_PermMCPPolar_-4_-2_NonPolarBias_-2_-1_PermCell_-9_-4_dPacking_0,3_0,64_nmcps_3,0_30,0'
    name_pkl = 'sampling_rsampling_N_10000_enzratio_1:3_2021_02_09_18:41'
    with open(directory + '/'+ filename+'/' +name_pkl + '.pkl', 'rb') as f:
        pk_as = pickle.load(f)

    cost_matrices = pk_as['function results'][-1]
    qoi_ob = QoI(cost_matrices, start_time,final_time, integration_tol, nintegration_samples,
                 tolsolve, params_values_fixed, param_sens_bounds, ds=ds)
    print(qoi_ob.generate_QoI_vals(params_unif))

    for i,func_name in enumerate(QOI_NAMES):
       eig_plots(qoi_ob.eigenvalues_QoI[func_name], qoi_ob.eigenvectors_QoI[func_name],param_sens_bounds,'rsampling',
                 func_name,enz_ratio_name,10000,"2021-02-09-18:41", threshold = 0, save=False)

if __name__ == '__main__':
    main(sys.argv, len(sys.argv))