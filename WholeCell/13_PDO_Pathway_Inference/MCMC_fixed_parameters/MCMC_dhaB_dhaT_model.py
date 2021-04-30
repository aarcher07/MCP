import numpy as np
import pandas as pd
from constants import *
import matplotlib.pyplot as plt
from numpy.random import standard_normal
import matplotlib as mpl
mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'
from dhaB_dhaT_model import DhaBDhaTModel
from MCMC import postdraws,adaptive_postdraws, maxpostdensity
import time
import pickle
from scipy.integrate import solve_ivp
import scipy.stats as stats
from mpi4py import MPI
import sys
from misc_functions import *
from data_gen_funs import f
import time
from pathlib import Path

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


def rprior(n,ds):
	samples = []
	if ds == "log_unif":
		for key,vals in PARAMETER_LOG_UNIF_BOUNDS.items():
			samples.append(stats.uniform.rvs(size=n))
	elif ds == "log_norm":
		for key,vals in PARAMETER_LOG_NORM_BOUNDS.items():
			samples.append(stats.norm.rvs(loc=vals[0],scale=vals[1],size=n))			
	else:
		for key,vals in PARAMETER_BOUNDS.items():
			samples.append(stats.loguniform.rvs(a=vals[0],b=vals[1],size=n))		
	return np.array(samples).T


def logprior(params,ds):
	logpdf = 0
	if ds == "log_unif":
		for i,(key,vals) in enumerate(PARAMETER_LOG_UNIF_BOUNDS.items()):
			logpdf += stats.uniform.logpdf(params[i])
	elif ds == "log_norm":
		for i,(key,vals) in enumerate(PARAMETER_LOG_NORM_BOUNDS.items()):
			logpdf += stats.norm.logpdf(params[i],loc=vals[0],scale=vals[1])				
	else:
		for i,(key,vals) in enumerate(PARAMETER_BOUNDS.items()):
			logpdf += stats.loguniform.logpdf(params[i],a=vals[0],b=vals[1])
	return logpdf

def loglik(params,dhaB_dhaT_model,sigma=[2,2,0.1]):

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
		fvals = f(params_to_dict,init_conds,dhaB_dhaT_model,tsamp)

		# compute difference for loglikelihood
		fvals[:,2] = fvals[:,2]/(scalar*DCW_TO_COUNT_CONC)
		sigma = np.array(sigma)
		data_diff_matrix = (fvals-data_sample_df)/sigma[np.newaxis,:]
		diff_f_data.extend(data_diff_matrix.ravel())
	return -0.5*np.dot(diff_f_data,diff_f_data) 

def test(sigma = [2,2,0.1],ds = "log_unif"):
	dhaB_dhaT_model = DhaBDhaTModel(ds =ds)

	file_name = 'MCMC_results_data/old_files/adaptive_lambda_0,01_beta_0,05_norm_nsamples_1000_sigma_[2,2,0,2]_date_2021_04_15_17_00_rank_2'
	params= load_obj(file_name)[-1]
	loglik_sigma = lambda param: loglik(param,dhaB_dhaT_model,sigma=sigma)
	logpost = lambda param: loglik_sigma(param) + logprior(param, dhaB_dhaT_model.ds_name)

	print(loglik_sigma(params))
	print(logprior(params,dhaB_dhaT_model.ds_name))
	print(logpost(params))

def argmaxdensity(argv, arc):
	# get arguments 
	sigma = [float(arg) for arg in argv[2:5]]
	dhaB_dhaT_model = DhaBDhaTModel(ds =argv[5])


	# set distributions
	loglik_sigma = lambda params: loglik(params,dhaB_dhaT_model,sigma=sigma)
	logpost = lambda params: loglik_sigma(params) + logprior(params, dhaB_dhaT_model.ds_name)
	rprior_ds = lambda n: rprior(n,dhaB_dhaT_model.ds_name)


	# set inital starting point
	def initial_param():
		file_name = 'MCMC_results_data/old_files/adaptive_lambda_0,01_beta_0,05_norm_nsamples_10_sigma_[2,2,0,2]_date_2021_04_12_21_28_rank_0'
		param_start = load_obj(file_name)[-1]
		param_start = param_start + 0.1 * standard_normal(len(param_start)) 
		return param_start
	print(logpost(initial_param()))
	tmax = maxpostdensity(rprior_ds,logpost,max_opt_iters = 1, initial_param = initial_param, 
						  maxiter = 10, jac=None, disp = True)
	print(tmax)


def main(argv, arc):
	# get arguments 
	nsamps = int(float(argv[1]))
	sigma = [float(arg) for arg in argv[2:5]]
	ds =argv[5]
	dhaB_dhaT_model = DhaBDhaTModel(ds =ds)
	lbda = float(argv[6])
	adaptive = int(argv[7])
	if adaptive:
		beta = float(argv[8])


	# set distributions
	loglik_sigma = lambda params: loglik(params,dhaB_dhaT_model,sigma=sigma)
	logpost = lambda params: loglik_sigma(params) + logprior(params, ds)
	rprior_ds = lambda n: rprior(n, ds)


	# set inital starting point
	def initial_param():
		file_name = 'MCMC_results_data/old_files/adaptive_lambda_0,01_beta_0,05_norm_nsamples_1000_sigma_[2,2,0,2]_date_2021_04_15_00_21_rank_0'
		param_start = load_obj(file_name)[-1]
		return param_start

 	# if adaptive or fixed MCMC
	if adaptive:
		time_start = time.time()
		tdraws = adaptive_postdraws(logpost, initial_param, nsamp=nsamps,beta=beta, lbda = lbda)
		time_end = time.time()
		print((time_end-time_start)/float(nsamps))
	else:
		time_start = time.time()
		tdraws = postdraws(logpost, rprior_ds,initial_param,  nsamp=nsamps,lbda = lbda)
		time_end = time.time()
		print((time_end-time_start)/float(2*nsamps))
	# store results
	date_string = time.strftime("%Y_%m_%d_%H_%M")

	# store images
	for i,param_name in enumerate(VARS_TO_TEX.keys()):
		plt.plot(range(int(nsamps)),tdraws[:,i])
		plt.title('Plot of MCMC distribution of ' + r'$\log(' + VARS_TO_TEX[param_name][1:-1] + ')$')
		plt.xlabel('iterations index')
		plt.ylabel(r'$\log(' + VARS_TO_TEX[param_name][1:-1] + ')$')
		if adaptive:
			adapt_name = "adaptive"
			folder_name = 'MCMC_results_plots/'+ adapt_name + "/sigma_"  + str(np.round(sigma,decimals=3)).replace('.',',').replace(' ','') +  "/" + "lambda_" + str(lbda).replace('.',',') + "_beta_" +  str(beta).replace('.',',') 
		else:
			adapt_name = "fixed"
			folder_name = 'MCMC_results_plots/'+ adapt_name + "/sigma_"  + str(np.round(sigma,decimals=3)).replace('.',',').replace(' ','') +  "/" + "lambda_" + str(lbda).replace('.',',') 

		folder_name += "/nsamples_"+ str(nsamps) +"/" +  ds[4:]+"/param_" + param_name 
		Path(folder_name).mkdir(parents=True, exist_ok=True)
		file_name = folder_name +'/date_'+date_string  + "_rank_" + str(rank)+ '.png'
		plt.savefig(file_name,bbox_inches='tight')
		plt.close()

	# save pickle data
	if adaptive:
		adapt_name = "adaptive"
		folder_name = 'MCMC_results_data/' + adapt_name + "/sigma_"  + str(np.round(sigma,decimals=3)).replace('.',',').replace(' ','') +  "/" + "lambda_" + str(lbda).replace('.',',') + "_beta_" +  str(beta).replace('.',',') 
	else:
		adapt_name = "fixed"
		folder_name ='MCMC_results_data/' + adapt_name + "/sigma_"  + str(np.round(sigma,decimals=3)).replace('.',',').replace(' ','') + "/"+  "lambda_" + str(lbda).replace('.',',')
	
	folder_name += "/nsamples_"+ str(nsamps) +"/" +  ds[4:]
	Path(folder_name).mkdir(parents=True, exist_ok=True)
	file_name = folder_name + "/date_" +date_string  + "_rank_" + str(rank)
	save_obj(tdraws,file_name)


if __name__ == '__main__':
	test(sigma = [float(arg) for arg in sys.argv[2:5]] , ds = sys.argv[5])
	main(sys.argv, len(sys.argv))