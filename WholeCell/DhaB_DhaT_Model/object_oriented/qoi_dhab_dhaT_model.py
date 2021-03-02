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
from active_subspaces import FUNCS_TO_FILENAMES,FUNCS_TO_NAMES
import seaborn as sns

class QoI(DhaBDhaTModelJacAS):
	def __init__(self, cost_matrices, start_time,final_time,
                integration_tol, nintegration_samples, tolsolve, params_values_fixed,
                param_sens_bounds, filename = None, external_volume = 9e-6, 
                rc = 0.375e-6, lc = 2.47e-6, rm = 7.e-8, 
                ncells_per_metrecubed =8e14, cellular_geometry = "rod", 
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
		super.__init__(start_time,final_time, integration_tol, nsamples, tolsolve, params_values_fixed,
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
			eigs, eigvals = np.linalg.eigh(cost_matrices[i])
			eigenvalues_QoI[func_name] = np.flip(eigs)
			eigenvectors_QoI[func_name] = np.flip(eigvals, axis=1)
		self.eigenvalues_QoI = eigenvalues_QoI
		self.eigenvectors_QoI = eigenvectors_QoI



	def QoI(self,params_unif_dict):
		"""
		Generate the QoI value at parameter value, params_unif_dict

		:params_unif_dict: dictionary of parameter values
		"""

		sdev = lambda t,x: self._sderiv(t,x,params_unif_dict)
		sdev_jac  = lambda t,x: self.sderiv_jac_state_vars_sp_fun(t,x,params_unif_dict)
		y0 = np.array(self.y0(**params_unif_dict))

		# event function
        event_stop = lambda t,y: self._event_stop(t,y,params_sens_dict)
		event_stop.terminal = True

		# solve ODE
		try:
			sol = solve_ivp(sdev,[0, final_time+1], y0, method="BDF",jac=sdev_jac, 
				t_eval=dhaB_dhaT_model_jacobian.time_orig, atol=dhaB_dhaT_model_jacobian.integration_tol,
				rtol=dhaB_dhaT_model_jacobian.integration_tol, events=event_stop)
		except:
			return [[],[],[]]

		status, time, sol_sample = [sol.status,sol.t,sol.y.T]

		# get max 3-HPA
		index_3HPA_max = np.argmax(sol_sample[:,self.index_3HPA_cytosol]) 
        # check if derivative is 0 of 3-HPA 
        statevars_maxabs = sol_sample[index_3HPA_max,:self.nvars]
        dev_3HPA = sdev(time[index_3HPA_max],statevars_maxabs)[self.index_3HPA_cytosol]


 		if 'nmcps' in self.params_values_fixed.keys():
            nmcps = self.params_values_fixed['nmcps']
        else:
            nmcps = params_sens_dict['nmcps']
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
                HPA_max = []


            # get sensitivities of Glycerol and 1,3-PDO after 5 hrs
            if status == 0 or (time[-1] > 5*HRS_TO_SECS):
                P_ext = sol_sample[self.first_index_close_enough,self.index_1_3PDO_ext]
                G_ext = sol_sample[self.first_index_close_enough,self.index_Glycerol_ext]
            elif status == 1:
                P_ext = sol_sample[-1,self.index_1_3PDO_ext]
                G_ext = sol_sample[-1,self.index_Glycerol_ext]
            else:
                P_ext = []
                G_ext = []

            sol_values = [HPA_max,
                          G_ext,
                          P_ext]
            return sol_values
        else:

   
            return [[],[],[]]