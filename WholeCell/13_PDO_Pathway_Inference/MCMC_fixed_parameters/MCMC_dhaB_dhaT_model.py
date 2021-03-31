import numpy as np
import pandas as pd
from constants import *
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'
from dhaB_dhaT_model import DhaBDhaTModel
from MCMC import postdraws
import time
import pickle
from scipy.integrate import solve_ivp
import scipy.stats as stats
from mpi4py import MPI
import sys

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()


time_series_df = pd.read_csv("data_time_series.csv")
init_gly_conds = time_series_df.loc[:,"Glycerol Init"].unique()
times_samples = {} 
data_samples = {}

for gly_cond in init_gly_conds:
	rows_bool = time_series_df.loc[:,"Glycerol Init"] == gly_cond
	times_samples[gly_cond] = time_series_df.loc[rows_bool,"Time"]
	data_samples[gly_cond] = time_series_df[["Glycerol","PDO","DCW"]][rows_bool].to_numpy()

data_columns = [3,5,6]

def f(params,init_conds,tsamples,dhaB_dhaT_model,tol = 10**-5):

	tsamplessecs = np.array([t*HRS_TO_SECS for t in tsamples])
	ds = lambda t,x: dhaB_dhaT_model.ds(t,x,params)
	ds_jac = lambda t,x: dhaB_dhaT_model.sderiv_jac_state_vars_fun(t,x,params)
	y0 = np.zeros(len(VARIABLE_INIT_NAMES))
	for i,init_names in enumerate(VARIABLE_INIT_NAMES):
		y0[i] = init_conds[init_names]

	sol = solve_ivp(ds,[0, tsamplessecs[-1]+10], y0, method = 'BDF', jac = ds_jac, t_eval=tsamplessecs,
					atol=tol,rtol=tol)#, events=event_stop)
	return sol.y.T

def rprior(n,ds):
	samples = []
	if ds == "log_unif":
		for key,vals in param_sens_log_unif_bounds.items():
			samples.append(stats.uniform.rvs(size=n))
	elif ds == "log_norm":
		for key,vals in param_sens_log_norm_bounds.items():
			samples.append(stats.norm.rvs(loc=vals[0],scale=vals[1],size=n))			
	else:
		for key,vals in param_sens_bounds.items():
			samples.append(stats.loguniform.rvs(a=vals[0],b=vals[1],size=n))		
	return np.array(samples).T


def logprior(params,ds):
	logpdf = 0
	if ds == "log_unif":
		for i,(key,vals) in enumerate(param_sens_log_unif_bounds.items()):
			logpdf += stats.uniform.logpdf(params[i])
	elif ds == "log_norm":
		for i,(key,vals) in enumerate(param_sens_log_norm_bounds.items()):
			logpdf += stats.norm.logpdf(params[i],loc=vals[0],scale=vals[1])				
	else:
		for i,(key,vals) in enumerate(param_sens_bounds.items()):
			logpdf += stats.loguniform.logpdf(params[i],a=vals[0],b=vals[1])
	return logpdf

def loglik(params,dhaB_dhaT_model,sigma=0.1):

	#CALIBRATION CONSTANT
	if dhaB_dhaT_model.ds_name == 'log_unif':
		bound_a,bound_b = param_sens_log_unif_bounds['scalar']
		scalar = 10**((bound_b - bound_a)*params[0] + bound_a) 
	elif dhaB_dhaT_model.ds_name == 'log_norm':
		scalar = 10**(params[0])
	else:
		scalar= params[0]

	# PARAMETERS FOR MODEL
	params_to_dict = {}
	for param,key in zip(params[1:],PARAMETER_LIST):
		params_to_dict[key] = param

	diff_f_data = []

	for gly_cond in init_gly_conds:
		data_sample_df = data_samples[gly_cond]
		init_conds = {'G_CYTO_INIT': 0, 
					  'H_CYTO_INIT': 0,
					  'P_CYTO_INIT': 0,
					  'G_EXT_INIT': data_sample_df[0,0], 
					  'H_EXT_INIT': 0,
					  'P_EXT_INIT': data_sample_df[0,1],
					  'CELL_CONC_INIT': DCW_TO_COUNT_CONC*scalar*data_sample_df[0,2]
					}
		tsamp = times_samples[gly_cond]
		ysol = f(params_to_dict,init_conds,tsamp,dhaB_dhaT_model)
		# compute difference for loglikelihood
		fvals = ysol[:,data_columns]
		fvals[:,2] = fvals[:,2]/(scalar*DCW_TO_COUNT_CONC)
		data_diff_matrix = (fvals-data_sample_df)/data_sample_df
		data_diff_matrix[:,2] = data_diff_matrix[:,2]
		diff_f_data.extend(data_diff_matrix.ravel())
	return -0.5*np.dot(diff_f_data,diff_f_data) / sigma**2

def test(sigma = 0.01,ds = "log_norm"):
	dhaB_dhaT_model = DhaBDhaTModel(ds =ds)

	params_trans = {'cellperGlyMass': 2*10**5,
					'PermCellGlycerol':10**-4,
					'PermCellPDO': 10**-3,
					'PermCell3HPA': 10**-2,
					'VmaxfDhaB': 800, 
					'KmDhaBG': 0.1 ,
					'VmaxfDhaT': 500,
					'KmDhaTH': 0.1,
					'VmaxfGlpK': 500 ,
					'KmGlpKG': 10}

	init_conds={'G_CYTO_INIT': 0, 
				'H_CYTO_INIT': 0,
				'P_CYTO_INIT': 0,
				'G_EXT_INIT': 50, 
				'H_EXT_INIT': 0,
				'P_EXT_INIT': 0,
				'CELL_CONC_INIT': 0.2*DCW_TO_COUNT_CONC
				}


	if dhaB_dhaT_model.ds_name == 'log_unif':
		params_dict = {}
		for param_name, param_val in params_trans.items():
			bound_a,bound_b = param_sens_log_unif_bounds[param_name]
			params_dict[param_name] = (np.log10(param_val) - bound_a)/(bound_b - bound_a)

	elif dhaB_dhaT_model.ds_name == 'log_norm':
		params_dict = {}
		for param_name, param_val in params_trans.items():
			params_dict[param_name] = np.log10(param_val)

	params = np.zeros(1 + len(params_dict.values()))
	params[0] = np.log10(1/2.)
	params[1:] = list(params_dict.values())
	
	loglik_sigma = lambda param: loglik(param,dhaB_dhaT_model,sigma=sigma)
	logpost = lambda param: loglik_sigma(param) + logprior(param, dhaB_dhaT_model.ds_name)

	print(loglik_sigma(params))
	print(logprior(params,dhaB_dhaT_model.ds_name))
	print(logpost(params))


def main(argv, arc):

	nsamps = float(argv[1])
	sigma = float(argv[2])
	dhaB_dhaT_model = DhaBDhaTModel(ds =argv[3])

	loglik_sigma = lambda params: loglik(params,dhaB_dhaT_model,sigma=sigma)
	logpost = lambda params: loglik_sigma(params) + logprior(params, dhaB_dhaT_model.ds_name)
	rprior_ds = lambda n: rprior(n,dhaB_dhaT_model.ds_name)

	nsamples = int(nsamps)

	tdraws = postdraws(rprior_ds,logpost, nsamp = nsamples, jac = None)
	date_string = time.strftime("%Y_%m_%d_%H:%M")

	for i,param_name in enumerate(VARS_TO_TEX.keys()):
		plt.plot(range(int(nsamples)),tdraws[:,i])
		plt.title('Plot of MCMC distribution of ' + r'$\log(' + VARS_TO_TEX[param_name][1:-1] + ')$')
		plt.xlabel('iterations index')
		plt.ylabel(r'$\log(' + VARS_TO_TEX[param_name][1:-1] + ')$')
		file_name = ds[4:] + "_nsamples_"+ str(nsamples)  + "_sigma_" + str(sigma)[:4].replace('.',',') +'_date_'+date_string +"_param_" + param_name + "_rank_" + str(rank)+ '.png'
		plt.savefig('MCMC_results_plots/'+file_name,bbox_inches='tight')
		plt.close()

	file_name_pickle = 'MCMC_results_data/' + ds[4:] +  '_nsamples_'+ str(nsamples) + "_sigma_" + str(sigma)[:4].replace('.',',') +'_date_'+date_string + "_rank_" + str(rank) + '.pkl'
	with open(file_name_pickle, 'wb') as f:
		 pickle.dump(tdraws, f)


if __name__ == '__main__':
	test(sigma = float(sys.argv[2]))
	for ds in ['log_norm']:
		argv = sys.argv
		argv.append(ds)
		main(sys.argv, len(sys.argv))