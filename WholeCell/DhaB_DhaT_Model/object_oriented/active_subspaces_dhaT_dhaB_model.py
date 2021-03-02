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
from skopt.space import Space

import matplotlib.pyplot as plt
import warnings
import sympy as sp
import scipy.sparse as sparse
import os
import sys

from dhaB_dhaT_model_jac import *


class DhaBDhaTModelJacAS(DhaBDhaTModelJac):
    def __init__(self,start_time,final_time,
                integration_tol, nsamples, tolsolve, params_values_fixed,
                param_sens_bounds, external_volume = 9e-6, 
                rc = 0.375e-6, lc = 2.47e-6, rm = 7.e-8, 
                ncells_per_metrecubed =8e14, cellular_geometry = "rod", 
                ds = ""):
        """
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

        self.param_sens_bounds = param_sens_bounds

        super().__init__(start_time,final_time,integration_tol, nsamples, tolsolve, params_values_fixed,
                        list(param_sens_bounds.keys()), external_volume, rc, lc, rm, 
                        ncells_per_metrecubed, cellular_geometry, ds)

    def _sderiv_id(self,t,x,params_sens):
        """
        Computes the spatial derivative of the system at time point, t, with the parameters
        [-1,1] transformed by transforming parameters into their original values
        :param t: time
        :param x: state variables
        :param params_sens: [-1,1] transformed parameter list
        """
        if params_sens is None:
            print("Please set the parameter values")
        params_unif = {}
        for param_name, param_val in params_sens.items():
            bound_a, bound_b = self.param_sens_bounds[param_name]
            param_trans = ((bound_b - bound_a)*param_val/2 + (bound_a + bound_b)/2) 
            if param_name in ['PermMCPPolar','NonPolarBias','PermCell']:
                params_unif[param_name] = 10**param_trans
            else:
                params_unif[param_name] = param_trans
        return super()._sderiv_id(t,x,params_unif)

    def _sderiv_log10(self,t,x,params_sens = None):
        """
        Computes the spatial derivative of the system at time point, t, with the
        log10 centered parameters, [-1,1], by transforming parameters into their 
        original log10 values
        :param t: time
        :param x: state variables
        :param params_sens: [-1,1] transformed parameter list
        """
        if params_sens is None:
            print("Please set the parameter values")
        params_unif = {}
        for param_name, param_val in params_sens.items():
            bound_a, bound_b = self.param_sens_bounds[param_name]
            params_unif[param_name] = ((bound_b - bound_a)*param_val/2 + (bound_a + bound_b)/2) 
        return super()._sderiv_log10(t,x,params_sens = params_unif)



def main(argv, arc):
    # get inputs
    enz_ratio_name = argv[1]
    
    # initialize variables
    ds = ''
    start_time = (10**(-15))
    final_time = 72*HRS_TO_SECS
    integration_tol = 1e-3
    tolsolve = 1e-5
    nsamples = 500
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

    # params_sens_dict  = {'kcatfDhaB': 400, # /seconds Input
    #                     'KmDhaBG': 0.6, # mM Input
    #                     'kcatfDhaT': 59.4, # /seconds
    #                     'KmDhaTH': 0.77, # mM
    #                     'KmDhaTN': 0.03, # mM
    #                     'NADH_MCP_INIT': 0.36,
    #                     'PermMCPPolar': np.log10(10**-3), 
    #                     'PermMCPNonPolar': np.log10(10**-5), 
    #                     'PermCell': np.log10(10.**-7),
    #                     'dPacking': 0.64,
    #                     'nmcps': 10}

    # params_unif = {}
    # for param_name, param_val in params_sens_dict.items():
    #     bound_a,bound_b = param_sens_bounds[param_name]
    #     params_unif[param_name] = 2*(param_val - bound_a)/(bound_b - bound_a) - 1

    sample_space = Space([(-1.,1.) for _ in range(len(param_sens_bounds))])
    params_unif = {key:val for key,val in zip(param_sens_bounds.keys(), sample_space.rvs(1)[0])}
    dhaB_dhaT_model_jacobian = DhaBDhaTModelJacAS(start_time, final_time, integration_tol, nsamples, tolsolve,
                                                params_values_fixed,param_sens_bounds, ds = ds)
    jacobian_est = np.array(dhaB_dhaT_model_jacobian.jac_subset(params_unif))
    
    print(jacobian_est)


if __name__ == '__main__':
    main(sys.argv, len(sys.argv))
