import numpy as np
import pandas as pd
from constants import *
import matplotlib.pyplot as plt
from dhaB_dhaT_model import DhaBDhaTModel
from MCMC import postdraws

from scipy.integrate import solve_ivp
import scipy.stats as stats

time_series_df = pd.read_csv("data_time_series.csv")
init_gly_conds = time_series_df.loc[:,"Glycerol Init"].unique()
times_samples = {} 
data_samples = {}

for gly_cond in init_gly_conds:
	rows_bool = time_series_df.loc[:,"Glycerol Init"] == gly_cond
	times_samples[gly_cond] = time_series_df.loc[rows_bool,"Time"]
	data_samples[gly_cond] = time_series_df[["Glycerol","PDO","DCW"]][rows_bool].to_numpy()

ds='log_norm'

dhaB_dhaT_model = DhaBDhaTModel(ds =ds)
tol = 10**-10
data_columns = [3,5,6]

def f(params,tsamples):
    tsamplessecs = np.array([t*HRS_TO_SECS for t in tsamples])
    ds = lambda t,x: dhaB_dhaT_model.ds(t,x,params)
    ds_jac = lambda t,x: dhaB_dhaT_model.sderiv_jac_state_vars_fun(t,x,params)
    y0 = np.zeros(len(VARIABLE_INIT_NAMES))
    for i,init_names in enumerate(VARIABLE_INIT_NAMES):
        y0[i] = params[init_names]

    sol = solve_ivp(ds,[0, tsamplessecs[-1]+10], y0, method = 'BDF', jac = ds_jac, t_eval=tsamplessecs,
                    atol=tol,rtol=tol)#, events=event_stop)

    return sol.y.T

def grad_f(params,tsamples):
    tsamplessecs = np.array([t*HRS_TO_SECS for t in tsamples])
    dsens = lambda t,x: dhaB_dhaT_model.dsens(t,x,params)
    dsens_jac = lambda t,x: dhaB_dhaT_model.dsens_jac(t,x,params)
    y0 = np.zeros(len(VARIABLE_INIT_NAMES) + len(VARIABLE_INIT_NAMES)*len(PARAMETER_LIST))
    for i,init_names in enumerate(VARIABLE_INIT_NAMES):
        y0[i] = params[init_names]
    sol = solve_ivp(dsens,[0, tsamplessecs[-1]+10], y0, method = 'BDF', jac = dsens_jac, t_eval=tsamplessecs,
                    atol=tol,rtol=tol)#, events=event_stop)
    return sol.y.T

def rprior(n):
	samples = []
	if ds == "log_unif":
		for key,vals in param_sens_log_unif_bounds.items():
			if not key in ['NADH','ATP']:
				samples.append(stats.uniform.rvs(loc=-1,scale=2,size=n))
	elif ds == "log_norm":
		for key,vals in param_sens_log_norm_bounds.items():
			if not key in ['NADH','ATP']:
				samples.append(stats.norm.rvs(loc=vals[0],scale=vals[1],size=n))			
	else:
		for key,vals in param_sens_bounds.items():
			if not key in ['NADH','ATP']:
				samples.append(stats.loguniform.rvs(a=vals[0],b=vals[1],size=n))		
	return np.array(samples).T


def logprior(params):
	logpdf = 0
	if ds == "log_unif":
		for i,(key,vals) in enumerate(param_sens_log_unif_bounds.items()):
			if not key in ['NADH','ATP']:
				logpdf += stats.uniform.logpdf(params[i],loc=-1,scale=2)
	elif ds == "log_norm":
		for i,(key,vals) in enumerate(param_sens_log_norm_bounds.items()):
			if not key in ['NADH','ATP']:
				logpdf += stats.norm.logpdf(params[i],loc=vals[0],scale=vals[1])				
	else:
		for i,(key,vals) in enumerate(param_sens_bounds.items()):
			if not key in ['NADH','ATP']:
				logpdf += stats.loguniform.logpdf(params[i],a=vals[0],b=vals[1])
	return logpdf

def loglik(params,sigma=0.1):
	params_to_dict = {}
	for param,key in zip(params,param_sens_bounds.keys()):
		if not key in ['NADH','ATP']: 
			params_to_dict[key] = param
	diff_f_data = []
	for gly_cond in init_gly_conds:
		data_sample_df = data_samples[gly_cond]
		init_conds =   {'NADH': 0.60,
						'ATP': 5.,
						'G_CYTO_INIT': 0, 
						'H_CYTO_INIT': 0,
						'P_CYTO_INIT': 0,
						'G_EXT_INIT': data_sample_df[0,0], 
						'H_EXT_INIT': 0,
						'P_EXT_INIT': data_sample_df[0,1],
						'CELL_CONC_INIT': DCW_TO_COUNT*data_sample_df[0,2]/dhaB_dhaT_model.external_volume
						}
		params_dict = {**params_to_dict, **init_conds}

		tsamp = times_samples[gly_cond]
		ysol = f(params_dict,tsamp)

		# compute difference for loglikelihood
		fvals = ysol[:,data_columns]
		fvals[:,2] = fvals[:,2]*dhaB_dhaT_model.external_volume/DCW_TO_COUNT
		data_diff_matrix = fvals-data_sample_df
		data_diff_matrix[:,2] = gly_cond*data_diff_matrix[:,2]
		diff_f_data.extend(data_diff_matrix.ravel())

	return -0.5*np.dot(diff_f_data,diff_f_data) / sigma**2

def grad_loglik(params,sigma=0.1):
	params_to_dict = {}
	for param,key in zip(params,param_sens_bounds.keys()):
		if not key in ['NADH','ATP']: 
			params_to_dict[key] = param
	grad_likelihood = np.zeros(len(params))
	for gly_cond in init_gly_conds:
		data_sample_df = data_samples[gly_cond]
		init_conds =   {'NADH': 0.60,
						'ATP': 5.,
						'G_CYTO_INIT': 0, 
						'H_CYTO_INIT': 0,
						'P_CYTO_INIT': 0,
						'G_EXT_INIT': data_sample_df[0,0], 
						'H_EXT_INIT': 0,
						'P_EXT_INIT': data_sample_df[0,1],
						'CELL_CONC_INIT': data_sample_df[0,2]*DCW_TO_COUNT/dhaB_dhaT_model.external_volume
						}


		params_dict = {**params_to_dict, **init_conds}
		tsamp = times_samples[gly_cond]
		ysol = grad_f(params_dict,tsamp)
		# compute difference for loglikelihood
		for i,col in enumerate(data_columns):
			if not i == 3:
				grad_qoi_times = ysol[:,(6 + col*len(PARAMETER_LIST)):(6 + (col+1)*len(PARAMETER_LIST))]
				fvals = ysol[:,col]
				data_diff_vec = fvals-data_sample_df[:,i]
				grad_likelihood += np.matmul(grad_qoi_times[:,:len(params)].T,data_diff_vec)
			else:
				grad_qoi_times = ysol[:,(6 + col*len(PARAMETER_LIST)):(6 + (col+1)*len(PARAMETER_LIST))]/DCW_TO_COUNT
				fvals = ysol[:,col]/DCW_TO_COUNT
				data_diff_vec = fvals-data_sample_df[:,i]
				grad_likelihood += np.matmul(grad_qoi_times[:len(params)].T,data_diff_matrix[:,i])
	return -grad_likelihood / sigma**2

nsamples = 10**2

params_trans = {'maxGrowthRate': 10**-6,
                'saturation_const': 100, 
                'PermCellGlycerol': 10**-2, 
                'PermCellPDO': 10**-1, 
                'PermCell3HPA':10**-2,
                'DhaB1Conc': 1,
                'DhaTConc': 1,
                'GlpKConc': 1, 
                'kcatfDhaB': 800, 
                'KmDhaBG': 0.1 ,
                'kcatfDhaT': 1000,
                'KmDhaTH': 0.1,
                'kcatfGlpK': 500 ,
                'KmGlpKG': 0.1}

if ds == 'log_unif':
    params_dict = {}
    for param_name, param_val in params_trans.items():
        if not param_name in VARIABLE_INIT_NAMES:
            bound_a,bound_b = param_sens_log_unif_bounds[param_name]
            params_dict[param_name] = 2*(np.log10(param_val) - bound_a)/(bound_b - bound_a) - 1
        else:
            params_dict[param_name] = param_val
elif ds == 'log_norm':
    params_dict = {}
    for param_name, param_val in params_trans.items():
        if not param_name in VARIABLE_INIT_NAMES:
            params_dict[param_name] = np.log10(param_val)
        else:
            params_dict[param_name] = param_val
else:
    params_dict = params_trans
params = np.array(list(params_dict.values()))
sigma = np.sqrt(1)
loglik_sigma = lambda params: loglik(params,sigma=sigma)
logpost =lambda params: loglik(params,sigma=sigma) + logprior(params)

print(loglik_sigma(params))
print(logprior(params))
print(logpost(params))

tdraws = postdraws(rprior,logpost, nsamp = nsamples, jac = None)
# for i in range(tdraws.shape[1]):
#  	plt.plot(range(int(nsamples)),tdraws[:,i])
#  	plt.show()
