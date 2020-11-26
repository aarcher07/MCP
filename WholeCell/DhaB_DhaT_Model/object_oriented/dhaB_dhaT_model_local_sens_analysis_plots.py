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
from dhaB_dhaT_model_local_sens_analysis import *

nsamples = 500
external_volume =  9e-6
ncells_per_metrecubed = 8e14 # 7e13-8e14 cells per m^3
ncells = external_volume*ncells_per_metrecubed   
mintime = 10**(-15)
secstohrs = 60*60
fintime = 72*60*60
ds = "log10"
tol = 10**-4

#################################################
# Initialize Parameters
################################################# 

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


for key in params_values_fixed.keys():
    if ds == "log2":
        if key in PARAMETER_LIST:
            params_values_fixed[key] = np.log2(params_values_fixed[key])
    if ds == "log10":
        if key in PARAMETER_LIST:
            params_values_fixed[key] = np.log10(params_values_fixed[key])

params_sens_list = ['kcatfDhaB','KmDhaBG','km',
                   'kc','dPacking', 'nmcps']


params_sens_dict  = {'kcatfDhaB':400, # /seconds Input
                    'KmDhaBG': 0.6, # mM Input
                    'km': 10**-7, 
                    'kc': 10.**-5,
                    'dPacking': 0.64,
                    'nmcps': 10}

for key in params_sens_dict.keys():
    if ds == "log2":
        params_sens_dict[key] = np.log2(params_sens_dict[key])
    if ds == "log10":
        params_sens_dict[key] = np.log10(params_sens_dict[key])

try:
    nmcps = params_values_fixed['nmcps']
except KeyError:
    nmcps = params_sens_dict['nmcps']


#################################################
# Initialize Sensitivity Equations
#################################################
model_local_sens = DhaBDhaTModelLocalSensAnalysis(params_values_fixed, params_sens_list, external_volume = external_volume, rc = 0.375e-6,
                                            lc =  2.47e-6, rm = 7.e-8, ncells_per_metrecubed = ncells_per_metrecubed, cellular_geometry = "rod", ds = ds)
model_local_sens.set_jacs_fun()
model_local_sens.create_jac_sens()
dsens_param = lambda t, xs: model_local_sens.dsens(t,xs,params_sens_dict)
dsens_jac_sparse_mat_fun_param = lambda t, xs: model_local_sens.dsens_jac_sparse_mat_fun(t,xs,params_sens_dict)
nparams_sens = model_local_sens.nparams_sens
n_sensitivity_eqs = model_local_sens.n_sensitivity_eqs

#################################################
# Initial Variables for Sensitivity Equation
#################################################

# initial conditions
nvars = model_local_sens.nvars
y0 = np.zeros(nvars) 
y0[-3] = params_values_fixed['G_EXT_INIT']  # y0[-5] gives the initial state of the external substrate.

sens0 = np.zeros(n_sensitivity_eqs)
for i,param in enumerate(params_sens_dict):
    if param in VARIABLE_INIT_NAMES:
        sens0[i:n_sensitivity_eqs:nparams_sens] = 1
xs0 = np.concatenate([y0,sens0])


#################################################
# Integrate with BDF
#################################################

    # terminal event
tolsolve = 10**-8
def event_stop(t,y):
    params = {**params_values_fixed, **params_sens_dict}
    dSsample = np.array(model_local_sens.ds(t,y[:model_local_sens.nvars],params))
    dSsample_dot = np.abs(dSsample).sum()
    return dSsample_dot - tolsolve
event_stop.terminal = True

timeorig = np.logspace(np.log10(mintime),np.log10(fintime),nsamples)

starttime = time.time()
sol = solve_ivp(dsens_param,[0, fintime+10], xs0, method="BDF", 
                jac = dsens_jac_sparse_mat_fun_param, events=event_stop,
                t_eval=timeorig, atol=tol,rtol=tol)
endtime = time.time()

#################################################
# Plot solution
#################################################
volcell = model_local_sens.cell_volume
volmcp = 4 * np.pi * (model_local_sens.rm ** 3) / 3
external_volume = model_local_sens.external_volume
colour = ['b','r','y','c','m']


print('code time: ' + str(endtime-starttime))
# plot state variables solution
print(sol.message)


# rescale the solutions
ncompounds = 3
timeorighours = sol.t/secstohrs


#plot parameters
namesvars = ['Glycerol', '3-HPA', '1,3-PDO']
sens_vars_names = [r'$kcat_f^{DhaB}$', r'$K_M^{DhaB}$', r'$k_{m}$', r'$k_{c}$', r'$dPacking$', r'$MCP$']#, r'$G_0$', r'$I_0$']
colour = ['b','r','y','c','m']

# get index of max 3-HPA 
index_max_3HPA = np.argmax(sol.y[4,:])

# cellular solutions
for i in range(0,ncompounds):
    ycell = sol.y[3+i, :]
    plt.plot(sol.t/secstohrs,ycell, colour[i])
plt.axvline(x=timeorighours[index_max_3HPA],ls='--',ymin=0.05,color='k')
plt.title('Plot of cellular concentration')
plt.legend(['Glycerol', '3-HPA', '1,3-PDO','time of max 3-HPA'], loc='upper right')
plt.xlabel('time (hr)')
plt.ylabel('concentration (mM)')
plt.grid() 
plt.show()

index_3HPA_cell = len(namesvars) + 1
figure, axes = plt.subplots(nrows=int(math.ceil(len(params_sens_dict)/2)), ncols=2, figsize=(10,10), sharex=True, sharey=True)
soly = sol.y[(nvars + index_3HPA_cell*nparams_sens):(nvars + (index_3HPA_cell+1)*nparams_sens), :]
maxy = np.max(soly)
miny = np.min(soly)
yub = 1.15*maxy if maxy > 0 else 0.85*maxy
lub = 0.85*miny if miny > 0 else 1.15*miny
for j in range(nparams_sens):
    axes[j // 2, j % 2].plot(timeorighours, soly[j,:].T)
    axes[j // 2, j % 2].axvline(x=timeorighours[index_max_3HPA],ls='--',color='k')
    axes[j // 2, j % 2].legend([r'$\partial (' + namesvars[i-len(namesvars)] + ')/\partial \log_10(' + sens_vars_names[j][1:]+ ')','time of max 3-HPA'], loc='upper right')
    axes[j // 2, j % 2].set_ylabel(r'$\partial (' + namesvars[i-len(namesvars)] + ')/\partial  \log_10(' + sens_vars_names[j][1:])
    axes[j // 2, j % 2].set_title(sens_vars_names[j])
    axes[j // 2, j % 2].grid()
    axes[j // 2, j % 2].set_ylim([lub, yub])
    print('Senstivity of maximum 3-HPA concentration to log_10('+sens_vars_names[j][1:-1] + '): ' + str(soly[j,index_max_3HPA]))
    if j >= (nparams_sens-2):
        axes[(nparams_sens-1) // 2, j % 2].set_xlabel('time/hrs')


figure.suptitle(r'Sensitivity, $\partial (' + namesvars[index_3HPA_cell-len(namesvars)]+')/\partial \log_10(p_i)$, of the cellular concentration of '
                + namesvars[index_3HPA_cell-len(namesvars)] + ' wrt $\log_10(p_i)$', y = 0.92)
# plt.savefig('/Users/aarcher/PycharmProjects/MCP/WholeCell/plots/Perm_SensitivityInternal_'+ namesvars[i-2] +'.png',
#             bbox_inches='tight')
plt.show()


# sensitivity variables external
timecheck = 5.
ind_first_close_enough =  np.argmin(np.abs(timeorighours-timecheck))

for i in range(-len(namesvars),0):
    figure, axes = plt.subplots(nrows=int(math.ceil(len(params_sens_dict)/2)), ncols=2, figsize=(10,10), sharex=True,sharey=True)
    if i == -3:
        soly = sol.y[-(nparams_sens):,:]
    else:
        soly = sol.y[-(i+1)*nparams_sens:-i*nparams_sens, :]
    maxy = np.max(soly)
    miny = np.min(soly)
    yub = 1.15*maxy if maxy > 0 else 0.85*maxy
    lub = 0.85*miny if miny > 0 else 1.15*miny
    for j in range(nparams_sens):
        axes[j // 2, j % 2].plot(timeorighours, soly[j,:].T)
        axes[j // 2, j % 2].set_ylabel(r'$\partial (' + namesvars[i+3] + ')/\partial \log_10(' + sens_vars_names[j][1:] + ')')
        axes[j // 2, j % 2].set_ylim([lub, yub])
        axes[j // 2, j % 2].set_title(sens_vars_names[j])
        axes[j // 2, j % 2].grid()


        if i == -3:
          axes[j // 2, j % 2].axvline(x=timeorighours[index_max_3HPA],ls='--',ymin=-0.05,color='k')
          print('Senstivity of Glycerol at 5 hrs to log_10('+sens_vars_names[j][1:-1] + '): ' + str(soly[j,ind_first_close_enough]))
        if i == -1:
          axes[j // 2, j % 2].axvline(x=timeorighours[index_max_3HPA],ls='--',ymin=-0.05,color='k')
          print('Senstivity of 1,3-PDO at 5 hrs to log_10('+sens_vars_names[j][1:-1] + '): ' + str(soly[j,ind_first_close_enough]))
        if j >= (nparams_sens-2):
            axes[(nparams_sens-1) // 2, j % 2].set_xlabel('time/hrs')

    figure.suptitle(r'Sensitivity, $\partial (' + namesvars[i + 3]+')/\partial \log_10(p_i)$, of the external concentration of '
                    + namesvars[i + 3] + ' wrt $\log_10(p_i)$', y = 0.92)
    # plt.savefig('/Users/aarcher/PycharmProjects/MCP/WholeCell/plots/Perm_SensitivityExternal_'+ namesvars[i+3]  +'.png',
    #             bbox_inches='tight')
    plt.show()



#check mass balance
ext_masses_org = y0[(nvars-3):nvars]* external_volume
cell_masses_org = y0[3:6] * volcell 
mcp_masses_org = y0[:3] * volmcp
ext_masses_fin = sol.y[(nvars-3):nvars, -1] * external_volume
cell_masses_fin = sol.y[3:6,-1] * volcell
mcp_masses_fin = sol.y[:3, -1] * volmcp
print(ext_masses_org.sum() + ncells*cell_masses_org.sum() + ncells*nmcps*mcp_masses_org.sum())
print(ext_masses_fin.sum() + ncells*cell_masses_fin.sum() + ncells*nmcps*mcp_masses_fin.sum())
