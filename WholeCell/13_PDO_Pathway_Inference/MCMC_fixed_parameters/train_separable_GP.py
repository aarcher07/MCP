import numpy as np
import math
import matplotlib.pyplot as plt 
import matplotlib as mpl
mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'
from constants import *
import pickle
from dhaB_dhaT_model import DhaBDhaTModel
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
import scipy.stats as stats
from build_separable_GP import *
from sklearn.model_selection import train_test_split
from MCMC import *
import time
from mpi4py import MPI
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()



comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()
# load data 

# input synthetic parameters
filename_list = [
				 #'unif_sample_paramspace_len_50_date_2021_03_18_14:54',
				 #'unif_sample_paramspace_len_100_date_2021_03_18_14:59',
				 # 'norm_sample_paramspace_len_50_date_2021_03_19_00:59',
				 'norm_sample_paramspace_len_100_date_2021_03_19_00:59'
				 ]


time_series_df = pd.read_csv("data_time_series_cleaned.csv").iloc[:,2:].to_numpy()
time_series_df_means = time_series_df.mean(axis=0)
time_series_df = time_series_df - time_series_df_means[np.newaxis,:]

for filename in filename_list:
	file_name_full_param = 'emulator_data/params_w_init_' + filename
	input_params = np.loadtxt(file_name_full_param + '.csv',delimiter=",")

	# QoI evaluations 
	filename_QoI = 'emulator_data/computer_data_' + filename
	QoI_syn = np.loadtxt(filename_QoI + '.csv',delimiter=",")
	QoI_syn_means = QoI_syn.mean(axis=0)
	QoI_syn = QoI_syn - QoI_syn_means[np.newaxis,:]

	# training data split for emulator
	inputtrain, inputtest, ftrain, ftest = train_test_split( input_params, QoI_syn, test_size=0.3,shuffle=True,random_state = 1)
	fitted_info = fitGP(inputtrain, ftrain, init_logeta=np.ones(nparams+2), lowerb=0*np.ones(nparams+2), upperb=5*np.ones(nparams+2))

	predmean = np.zeros(ftrain.shape)
	predLB =  np.zeros(ftrain.shape)
	predUB =  np.zeros(ftrain.shape)

	for k in range(inputtrain.shape[0]):
		predinfo = predictGP(fitted_info, inputtrain[k].reshape(1,-1), inputtrain, ftrain)
		predmean[k,] = predinfo['pred_mean']
		predLB[k,] = predmean[k,]- stats.norm.ppf(1-0.05/2)*np.sqrt(predinfo['pred_var'])
		predUB[k,] = predmean[k,]+ stats.norm.ppf(1-0.05/2)*np.sqrt(predinfo['pred_var'])
	print('GPMethod: Train MSE:' +str(np.round(np.mean((predmean-ftrain)**2),decimals=3))+', Coverage:' + str(100*np.round(np.mean((predLB<ftrain)*(predUB>ftrain)),decimals=3)) + '%')

	predmean = np.zeros(ftest.shape)
	predLB =  np.zeros(ftest.shape)
	predUB =  np.zeros(ftest.shape)
	for k in range(inputtest.shape[0]):
		predinfo = predictGP(fitted_info, inputtest[k].reshape(1,-1), inputtrain, ftrain)
		predmean[k,] = predinfo['pred_mean']
		predLB[k,] = predmean[k,]- stats.norm.ppf(1-0.05/2)*np.sqrt(predinfo['pred_var'])
		predUB[k,] = predmean[k,]+ stats.norm.ppf(1-0.05/2)*np.sqrt(predinfo['pred_var'])
	print('GPMethod: Test MSE:' +str(np.round(np.mean((predmean-ftest)**2),decimals=3))+', Coverage:' + str(100*np.round(np.mean((predLB<ftest)*(predUB>ftest)),decimals=3)) + '%')

	ds = 'log_' + filename[:4]


	def rprior(n):
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


	def logprior(params):
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

	def loglik(params,sigma=0.1):
		params_to_dict = {}
		for param,key in zip(params,param_sens_bounds.keys()):
			params_to_dict[key] = param
		diff_f_data = []
		params_init = np.zeros((len(INIT_CONDS_GLYPDODCW.values()),nparams))
		for i,init in enumerate(INIT_CONDS_GLYPDODCW.values()):
			init_conds = {'G_CYTO_INIT': 0, 
						  'H_CYTO_INIT': 0,
						  'P_CYTO_INIT': 0,
						  'G_EXT_INIT': init[0], 
						  'H_EXT_INIT': 0,
						  'P_EXT_INIT': init[1],
						  'CELL_CONC_INIT': DCW_TO_COUNT*init[2]/EXTERNAL_VOLUME
						}

			params_init[i,:len(PARAMETER_LIST)] = params
			params_init[i,len(PARAMETER_LIST):] = init
					
		etahat = fitted_info['etahat']
		etahat1 = etahat[:nparams]
		etahat2 = etahat[nparams]
		etahat3 = etahat[nparams+1]
		sigmahat = fitted_info['sigmahat']
		gammahat = fitted_info['gammahat']

		# obtain the final covariance structure

		corrCC = corr(inputtrain, inputtrain, etahat1)# c(t_tr, t_tr)
		corrYC = corr(params_init, inputtrain, etahat1)# c(t_tr, t_tr)
		corrCY = corr(inputtrain,params_init , etahat1)# c(t_tr, t_tr)
		corrYY = corr(params_init, params_init, etahat1)# c(t_tr, t_tr)
		timeSigma0 = timeSigma(etahat2)# timeSigma
		varSigma0 = varSigma(etahat3)# varSigma  
		fhat = np.matmul(np.kron(np.eye(varSigma0.shape[0]*timeSigma0.shape[0]),np.matmul(corrYC,np.linalg.inv(corrCC))),ftrain.flatten('F'))
		corrCCinvcorCY=np.linalg.solve(corrCC,corrCY)
		SigmaTheta = sigmahat*np.kron(np.kron(varSigma0,timeSigma0),corrYY-np.matmul(corrYC,corrCCinvcorCY)) + (sigma**2)*np.eye(varSigma0.shape[0]*timeSigma0.shape[0]*len(INIT_CONDS_GLYPDODCW))
		SigmaTheta_diff= np.linalg.solve(SigmaTheta,time_series_df.flatten('F') - fhat)
		return -np.dot(time_series_df.flatten('F') - fhat,SigmaTheta_diff) / sigma**2 - np.log(np.linalg.det(SigmaTheta))

	params_trans = {'maxGrowthRate': 10**-8,
	                'PermCellGlycerol':10**-4,
	                'PermCellPDO': 10**-3,
	                'PermCell3HPA': 10**-2,
	                'VmaxfDhaB': 800, 
	                'KmDhaBG': 0.77 ,
	                'VmaxfDhaT': 50,
	                'KmDhaTH': 1.,
	                'VmaxfGlpK': 50 ,
	                'KmGlpKG': 0.01}

	if ds == 'log_unif':
	    params_dict = {}
	    for param_name, param_val in params_trans.items():
	        bound_a,bound_b = param_sens_log_unif_bounds[param_name]
	        params_dict[param_name] = (np.log10(param_val) - bound_a)/(bound_b - bound_a) 

	elif ds == 'log_norm':
	    params_dict = {}
	    for param_name, param_val in params_trans.items():
	        params_dict[param_name] = np.log10(param_val)


	params = np.array(list(params_dict.values()))
	sigma = np.sqrt(5)
	loglik_sigma = lambda params: loglik(params,sigma=sigma)
	logpost =lambda params: loglik(params,sigma=sigma) + logprior(params)

	print(loglik_sigma(params))
	print(logprior(params))
	print(logpost(params))


	nsamples = int(1e5)
	tdraws = postdraws(rprior,logpost, nsamp = nsamples, jac = None)
	date_string = time.strftime("%Y_%m_%d_%H:%M")

	for i,param_name in enumerate(PARAMETER_LIST):
		plt.plot(range(int(nsamples)),tdraws[:,i])
		plt.title('Plot of MCMC distribution of ' + r'$\log(' + VARS_TO_TEX[param_name][1:-1] + ')$')
		plt.xlabel('iterations index')
		plt.ylabel(r'$\log(' + VARS_TO_TEX[param_name][1:-1] + ')$')
		#plt.show()
		file_name_add = "_nsamples_"+ str(nsamples)  + "_sigma_" + str(sigma)[:4].replace('.',',') +'_date_'+date_string +"_param_" + param_name + "_rank_" + str(rank)+ '.png'
		plt.savefig('emulator_MCMC_results_plots/emulator_' +filename +file_name_add,bbox_inches='tight')
		plt.close()

	file_name_pickle = 'emulator_MCMC_results_data/emulator_'  +filename + '_nsamples_'+ str(nsamples) + "_sigma_" + str(sigma)[:4].replace('.',',') +'_date_'+date_string + "_rank_" + str(rank) + '.pkl'
	with open(file_name_pickle, 'wb') as f:
	    pickle.dump(tdraws, f)

