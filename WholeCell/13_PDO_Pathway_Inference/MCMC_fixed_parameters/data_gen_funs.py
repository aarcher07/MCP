import numpy as np
import math
import matplotlib.pyplot as plt 
from constants import *
from misc_functions import *
from dhaB_dhaT_model import DhaBDhaTModel
from scipy.integrate import solve_ivp

def f(params,init_conds,dhaB_dhaT_model,tsamples=TIME_EVALS,tol = 10**-5):
	tsamplessecs = np.array([t*HRS_TO_SECS for t in tsamples])
	ds = lambda t,x: dhaB_dhaT_model.ds(t,x,params)
	ds_jac = lambda t,x: dhaB_dhaT_model.sderiv_jac_state_vars_fun(t,x,params)
	y0 = np.zeros(len(VARIABLE_INIT_NAMES))
	for i,init_names in enumerate(VARIABLE_INIT_NAMES):
		y0[i] = init_conds[init_names]

	sol = solve_ivp(ds,[0, tsamplessecs[-1]+10], y0, method = 'BDF', jac = ds_jac, t_eval=tsamplessecs,
					atol=tol,rtol=tol)#, events=event_stop)
	return sol.y[DATA_COLUMNS,:].T

def generate_data(params,dhaB_dhaT_model,tsamples=TIME_EVALS,tol = 10**-5):

	#CALIBRATION CONSTANT
	if dhaB_dhaT_model.ds_name == 'log_unif':
		bound_a,bound_b = PARAMETER_LOG_UNIF_BOUNDS['scalar']
		scalar = 10**((bound_b - bound_a)*params[0] + bound_a) 
	elif dhaB_dhaT_model.ds_name == 'log_norm':
		scalar = 10**(params[0])
	else:
		scalar= params[0]

	# PARAMETERS FOR MODEL
	params_to_dict = {}
	for param,key in zip(params[1:],MODEL_PARAMETER_LIST):
		params_to_dict[key] = param

	f_data = []

	for conds in INIT_CONDS_GLY_PDO_DCW.values():
		init_conds = {'G_CYTO_INIT': 0, 
					  'H_CYTO_INIT': 0,
					  'P_CYTO_INIT': 0,
					  'G_EXT_INIT': conds[0], 
					  'H_EXT_INIT': 0,
					  'P_EXT_INIT': conds[1],
					  'CELL_CONC_INIT': DCW_TO_COUNT_CONC*scalar*conds[2]
					}

		fvals = f(params_to_dict,init_conds,dhaB_dhaT_model,tsamples,tol)
		# compute difference for loglikelihood
		fvals[:,2] = fvals[:,2]/(scalar*DCW_TO_COUNT_CONC)
		f_data.append(fvals.flatten('F'))

	return f_data