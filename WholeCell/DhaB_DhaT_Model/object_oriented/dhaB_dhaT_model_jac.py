"""
This code extends local senstivity analysis code to 
compute the Jacobian of the system. 

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


class dhaB_dhaT_model_jac(dhaB_dhaT_model_local_sens_analysis):

    def __init__(self,start_time,final_time,
                integration_tol, nsamples, params_values_fixed,
                params_sens_list, external_volume = 9e-6, 
                rc = 0.375e-6, lc = 2.47e-6, rm = 7.e-8, 
                ncells_per_metrecubed =8e14, cellular_geometry = "rod", 
                ds = "log10"):

        """
        :params start_time: initial time of the system -- cannot be 0
        :params final_time: final time of the system 
        :params init_conc: inital concentration of the system
        :params integration_tol: integration tolerance
        :params nsamples: number of samples of time samples
        :params params_values_fixed: dictionary parameters whose senstivities are not being studied and 
                                     their values
        :params params_sens_list: list of parameters whose sensitivities are being studied
        :params external_volume: external volume of the system
        :params rc: radius of system
        :params lc: length of the cylindrical component of cellular_geometry = 'rod'
        :params rm: radius of MCP
        :params ncells_per_metrecubed: number of cells per m^3
        :params cellular_geometry: geometry of the cell, rod (cylinder with hemispherical ends)/sphere
        :params ds: transformation of the parameters, log2, log10 or identity.      
        """
        
        super().__init__(self,params_values_fixed, params_sens_list, external_volume, rc,
                        lc , rm , ncells_per_metrecubed, cellular_geometry, ds)

        # save integration parameters
        self._set_initial_conditions()
        self._set_initial_senstivity_conditions()
        self.xs0 = lambda params_sens_dict: np.concatenate([self.y0(params_sens_dict.values()),self.sens0])
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

        # save local senstivities analysis
        self.set_jacs_fun()
        self.create_jac_sens()

    def _set_initial_conditions(self):
        """
        Set initial condition of the reaction system
        :params init_conditions:
        """
        y0 = np.zeros(len(VARIABLE_NAMES))
        for i,variable in enumerate(VARIABLE_NAMES):
            try:
                y0[i] = self.params_values_fixed[variable] 
            except KeyError:
                y0[i] = self.params_sens_sp_dict[variable]
        self.y0 = sp.lambdify(self.param_sens_sp,y0)

    def _set_initial_senstivity_conditions(self):
        """
        Sets initial conditions for the senstivity equations.
        """
        sens0 = np.zeros(len(self.n_sensitivity_eqs))
        for i,param in enumerate(self.params_sens_list):
            if param in VARIABLE_INIT_NAMES:
                sens0[i::model_local_sens.nparams_sens] = 1
        self.sens0

    def jac(params_sens_dict):
        """
        Computes the sensitivities of the system at self.nsamples log-spaced time points 
        between self.start_time and self.final_time.


        :param params_sens_dict: parameter values to evaluate the senstivities
        :return sol.status     : status of the integrator, -1: Integration step failed.
                                                            0: The solver successfully reached the end of tspan.
                                                            1: A termination event occurred.
        :return sol.t          : time points at which the solution is evaluated
        :return sol.y.T        : solution and senstivities of system
        """
        # intergration functions
        dsens_param = lambda t, xs: model_local_sens.dsens(t,xs,params_sens_dict)
        dsens_jac_sparse_mat_fun_param = lambda t, xs: model_local_sens.dsens_jac_sparse_mat_fun(t,xs,params_sens_dict)
        xs0 = self.xs0(params_sens_dict)

        timeorig = np.logspace(self.start_time,self.final_time,self.nsamples)
        
        #stop event
        tolsolve = 10**-4
        def event_stop(t,y):
            params = {**params_values_fixed, **params_sens_dict}
            dSsample = np.array(model_local_sens.ds(t,y[:model_local_sens.nvars],params))
            dSsample_dot = np.abs(dSsample).sum()
            return dSsample_dot - tolsolve 
        event_stop.terminal = True

        #integrate
        sol = solve_ivp(dsens_param,[0, fintime+1], xs0, method="BDF",
                        jac = dsens_jac_sparse_mat_fun_param,t_eval=timeorig,
                        atol=self.integration_tol, rtol=self.integration_tol,
                        events=event_stop)

        return [sol.status,sol.t,sol.y.T]


    def jac_subset(params_sens_dict):
        """
        Subsets sensitivities of the system, jac(params_sens_dict), to compute the
        sensitivities of 3-HPA, Glycerol and 1,3-PDO after 5 hrs

        :params params_sens_dict: dictionary of the parameter values to evaluate the
                                  sensitivities

        :return jac_values      : sensitivities of 3-HPA, Glycerol and
                                  1,3-PDO after 5 hrs wrt parameters in params_sens_dict.keys()
                                  and evaluated at params_sens_dict.values()

        """
        status, time, jac_sample = jac(params_sens_dict)
        
        # get sensitivities of max 3-HPA
        index_3HPA_max = np.argmax(jac_sample[:,self.index_3HPA_cytosol]) 
        
        # check if derivative is 0 of 3-HPA 
        statevars_maxabs = jac_sample[index_3HPA_max,:self.nvars]
        dev_3HPA = self.ds(time[index_3HPA_max],statevars_maxabs)[self.index_3HPA_cytosol]
        
        if abs(dev_3HPA) < 1e-2:
            indices_3HPA_max_cytosol_params_sens =  np.array([[index_3HPA_max,i] for i in self.range_3HPA_cytosol_params_sens])
            jac_HPA_max = jac_sample[tuple(indices_3HPA_max_cytosol_params_sens.T)]
        else:
            jac_HPA_max = -1


        # get sensitivities of Glycerol and 1,3-PDO after 5 hrs
        if status == 0 or (time[-1] > 5*HRS_TO_SECS):
            jac_P_ext = jac_sample[tuple(self.indices_1_3PDO_ext_params_sens.T)]
            jac_G_ext = jac_sample[tuple(self.indices_Glycerol_ext_params_sens.T)]
        elif status == 1:
            jac_P_ext = jac_sample[tuple(self.indices_sens_1_PDO_ext_after_timecheck.T)]
            jac_G_ext = jac_sample[tuple(self.indices_sens_Glycerol_ext_after_timecheck.T)]
        else:
            jac_P_ext = -1
            jac_G_ext = -1

        jac_values = [jac_HPA_max,
                      jac_P_ext,
                      jac_G_ext]
        return jac_values

def main():
    start_time = (10**(-15))*HRS_TO_SECS
    final_time = 72*HRS_TO_SECS
    integration_tol = 1e-3
    nsamples = 500

    params_values_fixed = {'KmDhaTH': 0.77, # mM
                          'KmDhaTN': 0.03, # mM
                          'kcatfDhaT': 59.4, # /seconds
                          'enz_ratio': 1/1.33,
                          'NADH_MCP_INIT': 0.1,
                          'NAD_MCP_INIT': 0.1,
                          'G_MCP_INIT': 0,
                          'H_MCP_INIT': 0,
                          'P_MCP_INIT': 0,
                          'G_CYTO_INIT': 0,
                          'H_CYTO_INIT': 0,
                          'P_CYTO,INIT': 0 ,
                          'G_EXT_INIT': 200,
                          'H_EXT_INIT': 0,
                          'P_EXT_INIT': 0}

    params_sens_list = ['kcatfDhaB','KmDhaBG','km',
                        'kc','dPacking', 'nmcps']

    params_sens_dict  = {'kcatfDhaB':400, # /seconds Input
                        'KmDhaBG': 0.6, # mM Input
                        'km': 10**-7, 
                        'kc': 10.**-5,
                        'dPacking': 0.64,
                        'nmcps': 10}

    dhaB_dhaT_model_jacobian = dhaB_dhaT_model_jac(start_time, final_time, integration_tol, nsamples,
                                                   params_values_fixed,params_sens_list)

    print(dhaB_dhaT_model_jacobian.jac_subset(params_sens_dict))

