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
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}' #for \text command
import warnings
import sympy as sp
import scipy.sparse as sparse
import os
import sys
import pandas as pd
from constants import VARS_TO_TEX, VARS_TO_UNITS,QOI_NAMES
from dhaB_dhaT_model_local_sens_analysis import *
from dhaB_dhaT_model import HRS_TO_SECS




class DhaBDhaTModelJac(DhaBDhaTModelLocalSensAnalysis):

    def __init__(self,start_time,final_time, 
                integration_tol, nsamples, tolsolve, params_values_fixed,
                params_sens_list, external_volume = 9e-6, 
                rc = 0.375e-6, lc = 2.47e-6, rm = 7.e-8, 
                ncells_per_metrecubed =8e14, cellular_geometry = "rod", 
                transform = "log10"):

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
        :params transform: transformation of the parameters, log2, log10 or identity.      
        """
        
        super().__init__(params_values_fixed, params_sens_list, external_volume, rc,
                        lc , rm , ncells_per_metrecubed, cellular_geometry, transform)

        # save integration parameters
        self._set_initial_conditions()
        self._set_initial_senstivity_conditions()
        self.xs0 = lambda params_sens_dict: np.concatenate([self.x0(*params_sens_dict.values()),self.sens0])
        self.start_time = start_time
        self.final_time = final_time
        self.integration_tol = integration_tol
        self.nsamples = nsamples
        self.tolsolve = tolsolve
        self.time_orig = np.logspace(np.log10(self.start_time),np.log10(self.final_time),self.nsamples)
        self.time_orig_hours = self.time_orig/HRS_TO_SECS

        # index of HPA senstivities
        self.index_3HPA_cytosol = 4 # index of H_cystosol
        self.range_3HPA_cytosol_params_sens = range(self.nvars+self.index_3HPA_cytosol*self.nparams_sens,
                                                  self.nvars+(self.index_3HPA_cytosol+1)*self.nparams_sens)

        # index of 1,3-PDO after 5 hrs
        time_check = 5.
        self.first_index_close_enough =  np.argmin(np.abs(self.time_orig_hours-time_check)) # index of closest time after 5hrs
        self.index_1_3PDO_ext = -1 # index of P_ext 
        range_1_3PDO_ext_param_sens = range(self.index_1_3PDO_ext*self.nparams_sens,
                                            (self.index_1_3PDO_ext+1)*self.nparams_sens)
        self.indices_1_3PDO_ext_params_sens = np.array([(self.first_index_close_enough,i) for i in range_1_3PDO_ext_param_sens])
        self.indices_sens_1_3PDO_ext_before_timecheck = np.array([(-1,i) for i in range_1_3PDO_ext_param_sens])

        # index of Glycerol after 5 hrs
        self.index_Glycerol_ext = -3 # index of Glycerol_ext 
        range_Glycerol_ext_param_sens = range(self.index_Glycerol_ext*self.nparams_sens,
                                             (self.index_Glycerol_ext+1)*self.nparams_sens)
        self.indices_Glycerol_ext_params_sens = np.array([(self.first_index_close_enough,i) for i in range_Glycerol_ext_param_sens])
        self.indices_sens_Glycerol_ext_before_timecheck = np.array([(-1,i) for i in range_Glycerol_ext_param_sens])


    def _set_initial_conditions(self):
        """
        Set initial condition of the reaction system
        :params init_conditions:
        """
        x0 = np.zeros(len(VARIABLE_INIT_NAMES))
        for i,variable in enumerate(VARIABLE_INIT_NAMES):
            if variable in self.params_values_fixed.keys():
                x0[i] = self.params_values_fixed[variable] 
            else:
                x0[i] = self.params_sens_sp_dict[variable]
        self.x0 = sp.lambdify(self.params_sens_sp,x0)

    def _set_initial_senstivity_conditions(self):
        """
        Sets initial conditions for the senstivity equations.
        """
        sens0 = np.zeros(self.n_sensitivity_eqs)
        for i,param in enumerate(self.params_sens_list):
            if param in VARIABLE_INIT_NAMES:
                sens0[i::model_local_sens.nparams_sens] = 1
        self.sens0 = sens0

    def _event_stop(self,t,y,params_sens_dict):
        """
        Event stop function for integration

        :t: time
        :y: state variables
        :params_sens_dict: dictionary of parameter values
        """

        dSsample = np.array(self._sderiv(t,y[:self.nvars],params_sens_dict))
        dSsample_dot = np.abs(dSsample).sum()
        return dSsample_dot -  self.tolsolve 

    def jac(self,params_sens_dict):
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
        dsens_param = lambda t, xs: self.dsens(t,xs,params_sens_dict)
        dsens_param_jac = lambda t, xs: self.dsens_jac(t,xs,params_sens_dict)
        
        xs0 = self.xs0(params_sens_dict)

        
        #stop event
        event_stop = lambda t,y: self._event_stop(t,y,params_sens_dict)
        event_stop.terminal = True

        #integrate
        sol = solve_ivp(dsens_param,[0, self.final_time+1], xs0, method="BDF",
                        jac = dsens_param_jac,t_eval=self.time_orig,
                        atol=self.integration_tol, rtol=self.integration_tol)
                        #events=event_stop)

        return [sol.status,sol.t,sol.y.T]

    def jac_subset(self,params_sens_dict):
        """
        Subsets sensitivities of the system, jac(params_sens_dict), to compute the
        sensitivities of 3-HPA, Glycerol and 1,3-PDO after 5 hrs

        :params params_sens_dict: dictionary of the parameter values to evaluate the
                                  sensitivities

        :return jac_values      : sensitivities of 3-HPA, Glycerol and
                                  1,3-PDO after 5 hrs wrt parameters in params_sens_dict.keys()
                                  and evaluated at params_sens_dict.values()

        """
        try:
            status, time, jac_sample = self.jac(params_sens_dict)
        except ValueError:
            return 
        time_orig_hours = time/HRS_TO_SECS
       #plot parameters
        namesvars = ['Glycerol', '3-HPA', '1,3-PDO']
        colour = ['b','r','y','c','m']

        # cellular solutions
        # for i in range(0,ncompounds):
        #     ycell = jac_sample[5+i, :]
        #     plt.plot(sol.t/secstohrs, ycell, colour[i])
        index_max_3HPA = np.argmax(jac_sample[:,4])
        print(np.max(jac_sample[:,4]))
        i = 1
        ycell = jac_sample[:,3+i]
        plt.axvline(x=time_orig_hours[index_max_3HPA],ls='--',ymin=0.05,color='k')
        print(ycell.shape)
        print(time_orig_hours.shape)
        plt.plot(time_orig_hours, ycell, colour[i])
        plt.title('Plot of cellular 3-HPA concentration')
        plt.legend(['3-HPA'], loc='upper right')
        plt.xlabel('time (hr)')
        plt.ylabel('concentration (mM)')
        plt.grid() 
        plt.show()

        namesvars = ['Glycerol', '3-HPA', '1,3-PDO']

        # plot sensitivity variable solutions for MCP variables
        for i in range(0,len(namesvars)):
            figure, axes = plt.subplots(nrows=int(math.ceil(len(self.params_sens_list)/2)), ncols=2, figsize=(10,10), sharex=True, sharey=True)
            soly = jac_sample[:,(self.nvars + i*self.nparams_sens):(self.nvars + (i+1)*self.nparams_sens)]
            maxy = np.max(soly)
            miny =np.min(soly)
            yub = 1.15*maxy if maxy > 0 else 0.85*maxy
            lub = 0.85*miny if miny > 0 else 1.15*miny
            for j,param in enumerate(self.params_sens_list):
                axes[j // 2, j % 2].plot(time_orig_hours, soly[:,j].T)
                axes[j // 2, j % 2].set_ylabel(r'$\partial (\text{' + namesvars[i] + '})/\partial \log_{10}' + VARS_TO_TEX[param][1:])

                axes[j // 2, j % 2].set_title(VARS_TO_TEX[param])
                axes[j // 2, j % 2].set_ylim([lub, yub])
                axes[j // 2, j % 2].grid()
                if j >= (self.nparams_sens-2):
                    axes[(self.nparams_sens-1) // 2, j % 2].set_xlabel('time/hrs')

            figure.suptitle(r'Sensitivity, $\partial (\text{' + namesvars[i]+'})/\partial\log_{10} p_i$, of the MCP concentration of '
                           + namesvars[i] + ' wrt $p_i$', y = 0.92)
            # figure.suptitle(r'Sensitivity, $\partial (' + namesvars[i]+')/\partial p_i$, of the MCP concentration of '
            #                 + namesvars[i] + ' wrt $p_i$', y = 0.92)
            # plt.savefig('/Users/aarcher/PycharmProjects/MCP/WholeCell/plots/Perm_SensitivityInternal_'+ namesvars[i-2] +'.png',
            #             bbox_inches='tight')
            plt.show()


        # plot sensitivity variable solutions for cellular variables
        for i in range(len(namesvars),2*len(namesvars)):
            figure, axes = plt.subplots(nrows=int(math.ceil(len(self.params_sens_list)/2)), ncols=2, figsize=(10,10), sharex=True, sharey=True)
            soly = jac_sample[:,(self.nvars + i*self.nparams_sens):(self.nvars + (i+1)*self.nparams_sens)]
            maxy = np.max(soly)
            miny = np.min(soly)
            yub = 1.15*maxy if maxy > 0 else 0.85*maxy
            lub = 0.85*miny if miny > 0 else 1.15*miny
            for j,param in enumerate(self.params_sens_list):
                print(param)
                axes[j // 2, j % 2].plot(time_orig_hours, soly[:,j].T)
                axes[j // 2, j % 2].set_ylabel(r'$\partial (\text{' + namesvars[i-len(namesvars)] + '})/\partial \log_{10}' + VARS_TO_TEX[param][1:])
                #axes[j // 2, j % 2].set_ylabel(r'$\partial (' + namesvars[i-len(namesvars)] + ')/\partial ' + sens_vars_names[j][1:])
                axes[j // 2, j % 2].set_title(VARS_TO_TEX[param])
                axes[j // 2, j % 2].grid()
                axes[j // 2, j % 2].set_ylim([lub, yub])
                axes[j // 2, j % 2].axvline(x=time_orig_hours[index_max_3HPA],ls='--',ymin=0.05,color='k')

                if j >= (self.nparams_sens-2):
                    axes[(self.nparams_sens-1) // 2, j % 2].set_xlabel('time/hrs')


            figure.suptitle(r'Sensitivity, $\partial (\text{' + namesvars[i-len(namesvars)]+'})/\partial \log_{10} p_i$, of the cellular concentration of '
                            + namesvars[i-len(namesvars)] + ' wrt $p_i$', y = 0.92)
            # figure.suptitle(r'Sensitivity, $\partial (' + namesvars[i-len(namesvars)]+')/\partial p_i$, of the cellular concentration of '
            #                 + namesvars[i-len(namesvars)] + ' wrt $p_i$', y = 0.92)
            # plt.savefig('/Users/aarcher/PycharmProjects/MCP/WholeCell/plots/Perm_SensitivityInternal_'+ namesvars[i-2] +'.png',
            #             bbox_inches='tight')
            plt.show()


        # sensitivity variables
        for i in range(-len(namesvars),0):
            figure, axes = plt.subplots(nrows=int(math.ceil(len(self.params_sens_list)/2)), ncols=2, figsize=(10,10), sharex=True,sharey=True)
            if i == -3:
                soly = jac_sample[:,-(self.nparams_sens):]
            else:
                soly = jac_sample[:,-(i+4)*self.nparams_sens:-(i+3)*self.nparams_sens]
            maxy = np.max(soly)
            miny = np.min(soly)
            yub = 1.15*maxy if maxy > 0 else 0.85*maxy
            lub = 0.85*miny if miny > 0 else 1.15*miny
            for j,param in enumerate(self.params_sens_list):
                axes[j // 2, j % 2].plot(time_orig_hours, soly[:,j].T)
                axes[j // 2, j % 2].set_ylabel(r'$\partial (\text{' + namesvars[i+3] + '})/\partial \log_{10}' + VARS_TO_TEX[param][1:])
                # axes[j // 2, j % 2].set_ylabel(r'$\partial (' + namesvars[i+3] + ')/\partial ' + sens_vars_names[j][1:])
                axes[j // 2, j % 2].set_ylim([lub, yub])
                axes[j // 2, j % 2].set_title(VARS_TO_TEX[param])
                axes[j // 2, j % 2].grid()
                if j >= (self.nparams_sens-2):
                    axes[(self.nparams_sens-1) // 2, j % 2].set_xlabel('time/hrs')

            figure.suptitle(r'Sensitivity, $\partial (\text{' + namesvars[i + 3]+'})/\partial \log_{10} p_i$, of the external concentration of '
                            + namesvars[i + 3] + ' wrt $p_i$', y = 0.92)
            # figure.suptitle(r'Sensitivity, $\partial (' + namesvars[i + 3]+')/\partial \log_{10} p_i$, of the external concentration of '
            #                 + namesvars[i + 3] + ' wrt $p_i$', y = 0.92)
            # plt.savefig('/Users/aarcher/PycharmProjects/MCP/WholeCell/plots/Perm_SensitivityExternal_'+ namesvars[i+3]  +'.png',
            #             bbox_inches='tight')
            plt.show()


        # conservation of mass
        x0 = np.array(self.x0(**params_sens_dict))
        if 'nmcps' in self.params_values_fixed.keys():
            nmcps = self.params_values_fixed['nmcps']
        else:
            nmcps = params_sens_dict['nmcps']
        # original mass
        ext_masses_org = x0[(self.nvars-3):self.nvars]* self.external_volume
        cell_masses_org = x0[5:8] * self.cell_volume 
        mcp_masses_org = x0[:5] * self.mcp_volume
        mass_org = ext_masses_org.sum() +  self.ncells*cell_masses_org.sum() +  self.ncells*nmcps*mcp_masses_org.sum()

        # final mass
        ext_masses_fin = jac_sample[-1,(self.nvars-3):self.nvars] * self.external_volume
        cell_masses_fin = jac_sample[-1,5:8] * self.cell_volume
        mcp_masses_fin = jac_sample[-1,:5] * self.mcp_volume
        mass_fin = ext_masses_fin.sum() + self.ncells*cell_masses_fin.sum() + self.ncells*nmcps*mcp_masses_fin.sum()
        relative_diff = mass_fin/mass_org
        
        # get sensitivities of max 3-HPA
        index_3HPA_max = np.argmax(jac_sample[:,self.index_3HPA_cytosol]) 
        # check if derivative is 0 of 3-HPA 
        statevars_maxabs = jac_sample[index_3HPA_max,:self.nvars]
        dev_3HPA = self._sderiv(time[index_3HPA_max],statevars_maxabs,params_sens_dict)[self.index_3HPA_cytosol]
        # check if integrated correctly
        if (relative_diff > 0.5 and relative_diff < 1.5):
            if abs(dev_3HPA) < 1e-2:
                indices_3HPA_max_cytosol_params_sens =  np.array([[index_3HPA_max,i] for i in self.range_3HPA_cytosol_params_sens])
                jac_HPA_max = jac_sample[tuple(indices_3HPA_max_cytosol_params_sens.T)]
            else:
                jac_HPA_max = []


            # get sensitivities of Glycerol and 1,3-PDO after 5 hrs
            if status == 0 or (time[-1] > 5*HRS_TO_SECS):
                jac_P_ext = jac_sample[tuple(self.indices_1_3PDO_ext_params_sens.T)]
                jac_G_ext = jac_sample[tuple(self.indices_Glycerol_ext_params_sens.T)]
            elif status == 1:
                jac_P_ext = jac_sample[tuple(self.indices_sens_1_3PDO_ext_before_timecheck.T)]
                jac_G_ext = jac_sample[tuple(self.indices_sens_Glycerol_ext_before_timecheck.T)]
            else:
                jac_P_ext = []
                jac_G_ext = []

            jac_values = [jac_HPA_max,
                          jac_G_ext,
                          jac_P_ext]
            return jac_values
        else:
            return 

def main(argv, arc):
    # get inputs
    enz_ratio_name = argv[1]
    
    # initialize variables
    transform = 'log10'

    start_time = (10**(-15))
    final_time = 100*HRS_TO_SECS

    integration_tol = 1e-3
    nsamples = 500
    tolsolve = 10**-10
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


    params_sens_list = ['kcatfDhaB','KmDhaBG',
                        'kcatfDhaT','KmDhaTH','KmDhaTN',
                        'NADH_MCP_INIT',
                        'PermMCPPolar','PermMCPNonPolar',
                        'PermCellGlycerol','PermCellPDO','PermCell3HPA',
                        'dPacking', 'nmcps']

    params_sens_dict  = {'kcatfDhaB':400, # /seconds Input
                        'KmDhaBG': 0.6, # mM Input
                        'kcatfDhaT': 59.4, # /seconds
                        'KmDhaTH': 0.77, # mM
                        'KmDhaTN': 0.03, # mM
                        'NADH_MCP_INIT': 0.36,
                        'PermMCPPolar': 10**-3, 
                        'PermMCPNonPolar': 10**-2, 
                        'PermCellGlycerol': 10**-7,
                        'PermCellPDO':  10**-5,
                        'PermCell3HPA': 10**-2,
                        'dPacking': 0.64,
                        'nmcps': 15}

    for key in params_sens_dict.keys():
        if transform == "log2":
            params_sens_dict[key] = np.log2(params_sens_dict[key])
        if transform == "log10":
            params_sens_dict[key] = np.log10(params_sens_dict[key])

    dhaB_dhaT_model_jacobian = DhaBDhaTModelJac(start_time, final_time, integration_tol, nsamples,tolsolve,
                                                   params_values_fixed,params_sens_list, transform = transform)
    dhaB_dhaT_model_jacobian._set_jacs_fun()
    dhaB_dhaT_model_jacobian._create_jac_sens()    
    jacobian_est = np.array(dhaB_dhaT_model_jacobian.jac_subset(params_sens_dict))

    # format output
    _, _, jac_sample = dhaB_dhaT_model_jacobian.jac(params_sens_dict)
    max_3_HPA = np.max(jac_sample[:,dhaB_dhaT_model_jacobian.index_3HPA_cytosol])

    min_max_3_HPA = max_3_HPA
    max_max_3_HPA = max_3_HPA

    for i in range(len(jacobian_est)):
        param_names_tex = [ VARS_TO_TEX[params] for params in params_sens_dict.keys()]
        param_values_tex = [ "$" + "{:.3f}".format(10**params_sens_dict[params]) + '$ ' + VARS_TO_UNITS[params] 
                              for params in params_sens_dict.keys()]
        param_senstivities = [ "$" + ("{:.3f}".format(ja) if abs(ja) > 0.001  else "<0.001") + "$" for ja in jacobian_est[i,:]]

        DeltaQoI = [("($" + "{:.3f}".format(ja*np.log10(0.8)) + "$, $" + "{:.3f}".format(ja*np.log10(1.2)) + "$)"  
                     if abs(ja) > 0.001 else "--") for ja in jacobian_est[i,:] ]
        percentage_change = 20/100
        if i == 0:
            for ja in jacobian_est[i,:]:
                if ja > 0:
                    min_max_3_HPA += ja*np.log10(1 - percentage_change) 
                    max_max_3_HPA += ja*np.log10(1 + percentage_change)
                else:   
                    min_max_3_HPA += ja*np.log10(1 + percentage_change) 
                    max_max_3_HPA += ja*np.log10(1- percentage_change)
            print(min_max_3_HPA)
            print(max_max_3_HPA)
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

