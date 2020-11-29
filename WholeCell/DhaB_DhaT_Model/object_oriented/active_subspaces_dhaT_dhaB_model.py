"""
This code generates the average senstivity matrix that 
most affects the model in a bounded region of parameter space.

Programme written by aarcher07
Editing History:
- 25/11/20
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


class DhaBDhaTModelJacAs():
    def __init__(self,start_time,final_time,
                integration_tol, nsamples, params_values_fixed,
                param_bounds, external_volume = 9e-6, 
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
        :params param_bounds: bounds of parameters whose sensitivities are being studied
        :params external_volume: external volume of the system
        :params rc: radius of system
        :params lc: length of the cylindrical component of cellular_geometry = 'rod'
        :params rm: radius of MCP
        :params ncells_per_metrecubed: number of cells per m^3
        :params cellular_geometry: geometry of the cell, rod (cylinder with hemispherical ends)/sphere
        :params ds: transformation of the parameters, log2, log10 or identity.      
        """

        self.param_bounds = param_bounds

        super().__init__(self,start_time,final_time,
                        integration_tol, nsamples, params_values_fixed,
                        params_sens_list, external_volume, rc, lc, rm, 
                        ncells_per_metrecubed, cellular_geometry, ds)

    def sderiv(self,t,x,params):
        """
        Computes the spatial derivative of the system at time point, t, with the parameters
        [-1,1] transformed by transforming parameters into their original values
        :param t: time
        :param x: state variables
        :param params: [-1,1] transformed parameter list
        """
        if params is None:
            print("Please set the parameter values")
        params_unif = {}
        for param_name, param_val in params.items():
            bound_a, bound_b = self.param_bounds[param_name]
            as_bounds_2_param_bounds = (bound_b - bound_a)*param_val/2 + (bound_b + bound_a)/2
            if param in ['km','kc']:
                # permeabilities
                params_unif[param_name] = 10**as_bounds_2_param_bounds
            elif param in PARAMETER_LIST:
                # parameter except for initial conditions and permeability
                params_unif[param_name] = as_bounds_2_param_bounds
            else:
                # initial conditions
                params_unif[param] = as_bounds_2_param_bounds

        return super().sderiv(t,x,params = params_unif)

    def sderiv_log10_param(self,t,x,params):
        """
        Computes the spatial derivative of the system at time point, t, with the parameters
        [-1,1] transformed by transforming parameters into their original log10 values
        :param t: time
        :param x: state variables
        :param params: [-1,1] transformed parameter list
        """
        if params is None:
            print("Please set the parameter values")
        params_unif = {}
        for param_name, param_val in params.items():

            if param_name in self.param_bounds.keys():
                # parameter are being varied
                bound_a, bound_b = self.param_bounds[param_name]
                as_bounds_2_param_bounds = (bound_b - bound_a)*param_val/2 + (bound_b + bound_a)/2
                if param in ['km','kc']:
                    # permeabilities
                    params_unif[param_name] = as_bounds_2_param_bounds
                elif param in PARAMETER_LIST:
                    # parameter except for initial conditions and permeability
                    params_unif[param_name] = np.log10(as_bounds_2_param_bounds)
                else:
                    # initial conditions
                    params_unif[param] = as_bounds_2_param_bounds
            else:
                # parameter are fixed
                params_unif[param_name] = param_val


        return super().sderiv_log10_param(t,x,params = params_unif)

    def sderiv_log2_param(self,t,x,params):
        """
        Computes the spatial derivative of the system at time point, t, with the parameters
        [-1,1] transformed by transforming parameters into their original log2 values
        :param t: time
        :param x: state variables
        :param params: [-1,1] transformed parameter list
        """
        if params is None:
            print("Please set the parameter values")
        params_unif = {}
        for param_name, param_val in params.items():
            bound_a, bound_b = self.param_bounds[param_name]
            as_bounds_2_param_bounds = (bound_b - bound_a)*param_val/2 + (bound_b + bound_a)/2
            if param in ['km','kc']:
                # permeabilities
                params_unif[param_name] = np.log2(10**as_bounds_2_param_bounds)
            elif param in PARAMETER_LIST:
                # parameter except for initial conditions and permeability
                params_unif[param_name] = np.log2(as_bounds_2_param_bounds)
            else:
                # initial conditions
                params_unif[param] = as_bounds_2_param_bounds
        return super().sderiv_log10_param(t,x,params = params_unif)
        
def main(argv, arc):
    # get inputs
    enz_ratio_name = argv[1]
    
    # initialize variables
    ds = 'log10'
    start_time = (10**(-15))
    final_time = 72*HRS_TO_SECS
    integration_tol = 1e-3
    nsamples = 500
    enz_ratio_name_split =  enz_ratio_name.split("/")
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

    for key in params_values_fixed.keys():
        if ds == "log2":
            if key in PARAMETER_LIST:
                params_values_fixed[key] = np.log2(params_values_fixed[key])
        if ds == "log10":
            if key in PARAMETER_LIST:
                params_values_fixed[key] = np.log10(params_values_fixed[key])

    params_sens_list = ['kcatfDhaB','KmDhaBG',
                        'kcatfDhaT','KmDhaTH','KmDhaTN',
                        'NADH_MCP_INIT',
                        'km','kc',
                        'dPacking', 'nmcps']

    params_sens_dict  = {'kcatfDhaB':400, # /seconds Input
                        'KmDhaBG': 0.6, # mM Input
                        'kcatfDhaT': 59.4, # /seconds
                        'KmDhaTH': 0.77, # mM
                        'KmDhaTN': 0.03, # mM
                        'NADH_MCP_INIT': 0.1,
                        'km': 10**-7, 
                        'kc': 10.**-5,
                        'dPacking': 0.64,
                        'nmcps': 10}

    for key in params_sens_dict.keys():
        if ds == "log2":
            params_sens_dict[key] = np.log2(params_sens_dict[key])
        if ds == "log10":
            params_sens_dict[key] = np.log10(params_sens_dict[key])

    dhaB_dhaT_model_jacobian = DhaBDhaTModelJac(start_time, final_time, integration_tol, nsamples,
                                                   params_values_fixed,params_sens_list, ds = ds)
    jacobian_est = np.array(dhaB_dhaT_model_jacobian.jac_subset(params_sens_dict))
    
    # format output
    for i in range(len(jacobian_est)):
        param_names_tex = [ VARS_TO_TEX[params] for params in params_sens_dict.keys()]
        param_values_tex = [ "$" + "{:.3f}".format(10**params_sens_dict[params]) + '$ ' + VARS_TO_UNITS[params] 
                              for params in params_sens_dict.keys()]
        param_senstivities = [ "$" + ("{:.3f}".format(ja) if abs(ja) > 0.001  else "<0.001") + "$" for ja in jacobian_est[i,:]]

        DeltaQoI = [("($" + "{:.3f}".format(ja*np.log10(0.5)) + "$, $" + "{:.3f}".format(ja*np.log10(1.5)) + "$)"  
                     if abs(ja) > 0.001 else "--") for ja in jacobian_est[i,:] ]

        param_results = pd.DataFrame({'Parameter': param_names_tex,
                                      'Value': param_values_tex,
                                      'Sensitivity': param_senstivities,
                                      'Change in QoI given $\Delta x$': DeltaQoI})

        caption = "Sensitivity results for " + QOI_NAMES[i] + " given " + enz_ratio_name_split[0] + " dhaB$_1$: " + enz_ratio_name_split[1] + " dhaT."

        param_results_tex = param_results.to_latex(index=False,column_format='||c|c|c|c||',
                                                    escape=False, caption = caption)
        param_results_tex_edited = param_results_tex.splitlines().copy()
        for i in range(len(param_results_tex_edited)):
            if param_results_tex_edited[i] in ['\\midrule','\\toprule']:
                param_results_tex_edited[i] = '\\hline'
            elif param_results_tex_edited[i] == '\\bottomrule':
                param_results_tex_edited[i] = ''                
            elif param_results_tex_edited[i][-2:] == '\\\\':
                param_results_tex_edited[i] += '\\hline'
        print('\n'.join(param_results_tex_edited))

if __name__ == '__main__':
    main(sys.argv, len(sys.argv))