'''
This scripts generates the local sensitivity analysis plots associated
with DhaB_DhaT_Model_LocalSensAnalysis.py.

Programme written by aarcher07

Editing History:
- 26/10/20
'''

import numpy as np
from scipy.integrate import solve_ivp
from scipy.constants import *
import sys
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib
import math
import sympy as sp
import scipy.sparse as sparse
import time
import math
from numpy.linalg import LinAlgError
from DhaB_DhaT_Model_LocalSensAnalysis import *

nsamples = 500
external_volume =  9e-6
NcellsPerMCubed = 8e14 # 7e13-8e14 cells per m^3
Ncells = external_volume*NcellsPerMCubed   
mintime = 10**(-15)
secstohrs = 60*60
fintime = 72*60*60
integration_params = initialize_integration_params(external_volume = external_volume, 
                                                   Ncells =Ncells,cellular_geometry="rod",
                                                   Rc = 0.375e-6, Lc = 2.47e-6)
params = {'KmDhaTH': 0.77, # mM
      'KmDhaTN': 0.03, # mM
      'kcatfDhaT': 59.4, # /seconds
      'kcatfDhaB':400, # /seconds Input
      'KmDhaBG': 0.6, # mM Input
      'km': 10**-7, 
      'kc': 10.**-5,
      'dPacking': 0.64,
      'Nmcps': 10,
      'enz_ratio': 1/1.33}

init_conditions = { 'GInit': 200, #  2 * 10^(-4) mol/cm3 = 200 mM. 
                  'NInit': 1., # mM
                  'DInit': 1. # mM
                  }
Nmcps = params['Nmcps']
tolG = 0.5*init_conditions['GInit']

def event_Gmin(t,y):
    return y[-3] - tolG
def event_Pmax(t,y):
    return y[-1] - tolG

params_sens_dict = create_param_symbols('kcatfDhaB',
                                        'KmDhaBG',
                                        'km',
                                        'kc',
                                        'dPacking',
                                        'Nmcps')

# log transform parameters in params_sens_dict
for key in params.keys():
    params[key] = np.log2(params[key])
dS = SDerivLog2Param
# store info about parameters
nParams = len(params_sens_dict)
integration_params['nParams'] = nParams
integration_params['Sensitivity Params'] = params_sens_dict
nSensitivityEqs = integration_params['nParams']*integration_params['nVars']
integration_params['nSensitivityEqs'] = nSensitivityEqs

#################################################
# Integrate with BDF
#################################################

# initial conditions
n_compounds_cell = 3
nVars = integration_params['nVars']
y0 = np.zeros(nVars) 
y0[-3] = init_conditions['GInit']  # y0[-5] gives the initial state of the external substrate.
y0[0] = init_conditions['NInit']  # y0[5] gives the initial state of the external substrate.
y0[1] = init_conditions['DInit']  # y0[6] gives the initial state of the external substrate.
# time samples
# initial conditions -- sensitivity equation
sens0 = np.zeros(nSensitivityEqs)
for i,param in enumerate(params_sens_dict):
    if param in ['GInit', 'IInit', 'NInit', 'DInit']:
        sens0[i:nSensitivityEqs:nParams] = 1
xs0 = np.concatenate([y0,sens0])
# setup differential eq
x_sp, sensitivity_sp = create_state_symbols(integration_params['nVars'], integration_params['nParams'])
SDerivSymbolicJacParamsLambFun, SDerivSymbolicJacConcLambFun = compute_jacs(x_sp, params_sens_dict,integration_params, diffeq_params=params, dS=dS)
dSensParams = lambda t,xs: dSens(t, xs, params, integration_params, SDerivSymbolicJacParamsLambFun,
                                 SDerivSymbolicJacConcLambFun, dS=dS)
#create jacobian of dSensParams
dSensSymJacSparseMatLamFun = create_jac_sens(x_sp, sensitivity_sp, params, integration_params,
                                             SDerivSymbolicJacParamsLambFun, SDerivSymbolicJacConcLambFun, dS = dS)

# solution params
tol = 1e-3
timeorig = np.logspace(np.log10(mintime),np.log10(fintime),nsamples)
print(dSensParams(0,xs0))

# # terminal event
# starttime = time.time()
# sol = solve_ivp(dSensParams,[0, fintime+10], xs0, method="BDF", jac = dSensSymJacSparseMatLamFun, t_eval=timeorig,
#                  atol=tol,rtol=tol, events=[event_Gmin,event_Pmax])
# endtime = time.time()


# #################################################
# # Plot solution
# #################################################
# volcell = integration_params['cell volume']
# volmcp = 4 * np.pi * (integration_params['Rm'] ** 3) / 3
# external_volume = integration_params['external_volume']
# colour = ['b','r','y','c','m']


# print('code time: ' + str(endtime-starttime))
# # plot state variables solution
# print(sol.message)


# # rescale the solutions
# ncompounds = 3
# timeorighours = sol.t/secstohrs


# #plot parameters
# namesvars = ['Glycerol', '3-HPA', '1,3-PDO']
# sens_vars_names = [r'$kcat_f^{DhaB}$', r'$K_M^{DhaB}$', r'$k_{m}$', r'$k_{c}$', r'$dPacking$', r'$MCP$']#, r'$G_0$', r'$I_0$']
# colour = ['b','r','y','c','m']

# # get index of max 3-HPA 
# index_max_3HPA = np.argmax(sol.y[6,:])

# # cellular solutions
# for i in range(0,ncompounds):
#     ycell = sol.y[5+i, :]
#     plt.plot(sol.t/secstohrs,ycell, colour[i])
# plt.axvline(x=timeorighours[index_max_3HPA],ls='--',ymin=0.05,color='k')
# plt.title('Plot of cellular concentration')
# plt.legend(['Glycerol', '3-HPA', '1,3-PDO','time of max 3-HPA'], loc='upper right')
# plt.xlabel('time (hr)')
# plt.ylabel('concentration (mM)')
# plt.grid() 
# plt.show()

# index_3HPA_cell = 2+len(namesvars) + 1
# figure, axes = plt.subplots(nrows=int(math.ceil(len(params_sens_dict)/2)), ncols=2, figsize=(10,10), sharex=True, sharey=True)
# soly = sol.y[(nVars + index_3HPA_cell*nParams):(nVars + (index_3HPA_cell+1)*nParams), :]
# maxy = np.max(soly)
# miny = np.min(soly)
# yub = 1.15*maxy if maxy > 0 else 0.85*maxy
# lub = 0.85*miny if miny > 0 else 1.15*miny
# for j in range(nParams):
#     axes[j // 2, j % 2].plot(timeorighours, soly[j,:].T)
#     axes[j // 2, j % 2].axvline(x=timeorighours[index_max_3HPA],ls='--',color='k')
#     axes[j // 2, j % 2].legend([r'$\partial (' + namesvars[i-2-len(namesvars)] + ')/\partial \log_2(' + sens_vars_names[j][1:]+ ')','time of max 3-HPA'], loc='upper right')
#     axes[j // 2, j % 2].set_ylabel(r'$\partial (' + namesvars[i-2-len(namesvars)] + ')/\partial  \log_2(' + sens_vars_names[j][1:] + ')')
#     axes[j // 2, j % 2].set_title(sens_vars_names[j])
#     axes[j // 2, j % 2].grid()
#     axes[j // 2, j % 2].set_ylim([lub, yub])
#     print('Senstivity of maximum 3-HPA concentration to log_2('+sens_vars_names[j][1:-1] + '): ' + str(soly[j,index_max_3HPA]) + ')')
#     if j >= (nParams-2):
#         axes[(nParams-1) // 2, j % 2].set_xlabel('time/hrs')


# figure.suptitle(r'Sensitivity, $\partial (' + namesvars[index_3HPA_cell-2-len(namesvars)]+')/\partial \log_2(p_i)$, of the cellular concentration of '
#                 + namesvars[index_3HPA_cell-2-len(namesvars)] + ' wrt $\log_2(p_i)$', y = 0.92)
# # plt.savefig('/Users/aarcher/PycharmProjects/MCP/WholeCell/plots/Perm_SensitivityInternal_'+ namesvars[i-2] +'.png',
# #             bbox_inches='tight')
# plt.show()


# # sensitivity variables external
# timecheck = 5.
# ind_first_close_enough =  np.argmin(np.abs(timeorighours-timecheck))

# for i in range(-len(namesvars),0):
#     figure, axes = plt.subplots(nrows=int(math.ceil(len(params_sens_dict)/2)), ncols=2, figsize=(10,10), sharex=True,sharey=True)
#     if i == -3:
#         soly = sol.y[-(nParams):,:]
#     else:
#         soly = sol.y[-(i+1)*nParams:-i*nParams, :]
#     maxy = np.max(soly)
#     miny = np.min(soly)
#     yub = 1.15*maxy if maxy > 0 else 0.85*maxy
#     lub = 0.85*miny if miny > 0 else 1.15*miny
#     for j in range(nParams):
#         axes[j // 2, j % 2].plot(timeorighours, soly[j,:].T)
#         axes[j // 2, j % 2].set_ylabel(r'$\partial (' + namesvars[i+3] + ')/\partial \log_2(' + sens_vars_names[j][1:] + ')')
#         axes[j // 2, j % 2].set_ylim([lub, yub])
#         axes[j // 2, j % 2].set_title(sens_vars_names[j])
#         axes[j // 2, j % 2].grid()


#         if i == -3:
#           axes[j // 2, j % 2].axvline(x=timeorighours[index_max_3HPA],ls='--',ymin=-0.05,color='k')
#           print('Senstivity of Glycerol at 5 hrs to log_2('+sens_vars_names[j][1:-1] + '): ' + str(soly[j,ind_first_close_enough]))
#         if i == -1:
#           axes[j // 2, j % 2].axvline(x=timeorighours[index_max_3HPA],ls='--',ymin=-0.05,color='k')
#           print('Senstivity of 1,3-PDO at 5 hrs to log_2('+sens_vars_names[j][1:-1] + '): ' + str(soly[j,ind_first_close_enough]))
#         if j >= (nParams-2):
#             axes[(nParams-1) // 2, j % 2].set_xlabel('time/hrs')

#     figure.suptitle(r'Sensitivity, $\partial (' + namesvars[i + 3]+')/\partial \log_2(p_i)$, of the external concentration of '
#                     + namesvars[i + 3] + ' wrt $\log_2(p_i)$', y = 0.92)
#     # plt.savefig('/Users/aarcher/PycharmProjects/MCP/WholeCell/plots/Perm_SensitivityExternal_'+ namesvars[i+3]  +'.png',
#     #             bbox_inches='tight')
#     plt.show()



# #check mass balance
# ext_masses_org = y0[(nVars-3):nVars]* external_volume
# cell_masses_org = y0[5:8] * volcell 
# mcp_masses_org = y0[:5] * volmcp
# ext_masses_fin = sol.y[(nVars-3):nVars, -1] * external_volume
# cell_masses_fin = sol.y[5:8,-1] * volcell
# mcp_masses_fin = sol.y[:5, -1] * volmcp
# print(ext_masses_org.sum() + Ncells*cell_masses_org.sum() + Ncells*Nmcps*mcp_masses_org.sum())
# print(ext_masses_fin.sum() + Ncells*cell_masses_fin.sum() + Ncells*Nmcps*mcp_masses_fin.sum())
