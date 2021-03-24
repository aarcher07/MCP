import numpy as np
import math
import matplotlib.pyplot as plt 
from constants import *
import pickle
from dhaB_dhaT_model import DhaBDhaTModel
import pandas as pd
from scipy.integrate import solve_ivp


def generate_computer_simulation_data(filename,
									  tol = 10**-5):

	with open('emulator_data/' + filename + '.pkl', 'rb') as f:
	    param_syn = pickle.load(f)

	ds='log_' + filename[:4]

	dhaB_dhaT_model = DhaBDhaTModel(ds =ds)

	
	data_columns = [3,5,6]

	def f(params,init_conds,tsamples):
	    tsamplessecs = np.array([t*HRS_TO_SECS for t in tsamples])
	    ds = lambda t,x: dhaB_dhaT_model.ds(t,x,params)
	    ds_jac = lambda t,x: dhaB_dhaT_model.sderiv_jac_state_vars_fun(t,x,params)
	    y0 = np.zeros(len(VARIABLE_INIT_NAMES))
	    for i,init_names in enumerate(VARIABLE_INIT_NAMES):
	        y0[i] = init_conds[init_names]

	    sol = solve_ivp(ds,[0, tsamplessecs[-1]+10], y0, method = 'BDF', jac = ds_jac, t_eval=tsamplessecs,
	                    atol=tol,rtol=tol)#, events=event_stop)
	    return  sol.y[data_columns,:].T

	data_syn = []
	param_data = []
			
	for i in range(param_syn.shape[0]):
		params = {key: val for key,val in zip(PARAMETER_LIST,param_syn[i,:])}
		params_array =param_syn[i,:]

		data_syn_conds = []
		try:
			for init_cond in INIT_CONDS_GLYPDODCW.values():

				init_conds = {'G_CYTO_INIT': 0, 
							  'H_CYTO_INIT': 0,
							  'P_CYTO_INIT': 0,
							  'G_EXT_INIT': init_cond[0], 
							  'H_EXT_INIT': 0,
							  'P_EXT_INIT': init_cond[1],
							  'CELL_CONC_INIT': DCW_TO_COUNT*init_cond[2]/dhaB_dhaT_model.external_volume
							}

				data_cols = f(params,init_conds,TIME_EVALS)
				param_data.append(np.concatenate((params_array,init_cond)))
				
				data_cols[:,2] = data_cols[:,2]*dhaB_dhaT_model.external_volume/DCW_TO_COUNT
				data_row = data_cols.flatten('F')
				data_syn.append(data_row)

		except ValueError:
			continue



	param_data = np.array(param_data)
	print(param_data.shape)
	file_name_full_param = 'emulator_data/params_w_init_' + filename
	np.savetxt(file_name_full_param + '.csv', param_data, delimiter=",")

	data_syn = np.array(data_syn)
	print(data_syn.shape)
	file_name_data_csv = 'emulator_data/computer_data_'+ filename
	np.savetxt(file_name_data_csv + '.csv', data_syn, delimiter=",")

filename_list = [
				 #'unif_sample_paramspace_len_50_date_2021_03_18_14:54',
				 #'unif_sample_paramspace_len_100_date_2021_03_18_14:59',
				 'norm_sample_paramspace_len_50_date_2021_03_19_00:59',
				 'norm_sample_paramspace_len_100_date_2021_03_19_00:59'
				 ]


for filename in filename_list:
	generate_computer_simulation_data(filename)
