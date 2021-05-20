import os
import pickle
import numpy as np

for ds in ["norm"]:
	param_sample = []
	for file in os.listdir("../emulator_MCMC_results_data/"):
		#if file.startswith(ds):

		if file.startswith("emulator_" + ds +"_sample_paramspace_len_50"):
				with open(os.path.join("../emulator_MCMC_results_data", file), 'rb') as f:
					param_load = pickle.load(f)
					param_sample.append(param_load[range(1000,len(param_load),25),:])

	mean_val = np.mean(np.concatenate(param_sample),axis=0)
	lb = np.quantile(np.concatenate(param_sample),0.025,axis=0)
	ub = np.quantile(np.concatenate(param_sample),0.975,axis=0)
	mean_dict = {}
	lb_dict = {}
	ub_dict = {}
	param_dict= {}
	if ds == 'unif':
	    for i, param_name in enumerate(PARAMETER_LIST):
	        bound_a,bound_b = param_sens_log_unif_bounds[param_name]
	        mean_dict[param_name] = 10**((bound_b - bound_a)*mean_val[i] + bound_a)  
	        lb_dict[param_name] = 10**((bound_b - bound_a)*lb[i] + bound_a) 
	        ub_dict[param_name] = 10**((bound_b - bound_a)*ub[i] + bound_a) 
	        param_dict[param_name] = mean_val[i]

	elif ds == 'norm':
	    for i, param_name in enumerate(PARAMETER_LIST):
	        mean_dict[param_name] = 10**(mean_val[i])
	        lb_dict[param_name] = 10**(lb[i])
	        ub_dict[param_name] = 10**(ub[i] )
	        param_dict[param_name] = mean_val[i]

	print(mean_dict)

