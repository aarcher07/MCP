import numpy as np
import math
import matplotlib.pyplot as plt 
from constants import *
from misc_functions import *
import pickle
import time
import scipy.stats as stats
from skopt.space import Space
from skopt.sampler import Lhs
from dhaB_dhaT_model import DhaBDhaTModel
from scipy.integrate import solve_ivp
from build_separable_GP import *
from sklearn.model_selection import train_test_split
from mpi4py import MPI
from data_gen_funcs import *

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()



ds = 'log_unif'
tol = 1e-7
dhaB_dhaT_model = DhaBDhaTModel(ds =ds)
n_samples = 10
n_more_params = 2
max_training_length = (n_samples+ n_more_params)* 4
ninitial = 10

# generate initial parameters and evaluation data
space = Space([(0., 1.) for _ in range(len(QoI_PARAMETER_LIST))])
lhs = Lhs(lhs_type="classic", criterion='maximin')

input_params = lhs.generate(space.dimensions, n_samples)
input_train_prop, input_test_prop = train_test_split( input_params, test_size=0.3, random_state = 1)

input_train, input_test = [], []
ftrain, ftest = [], []

for explan_set_prop, (explan_set,response_data) in zip([input_train_prop,input_test_prop],zip([input_train,input_test],[ftrain, ftest])):
	for param in explan_set_prop:
		try:
			response_data.append(generate_data(param,dhaB_dhaT_model,tol = tol))
			explan_set.append(param)
			print('hi')
		except TypeError:
			print(param)

input_train = [np.concatenate((param,cond)) for param in input_train for cond in INIT_CONDS_GLY_PDO_DCW.values()]
input_test = [np.concatenate((param,cond)) for param in input_test for cond in INIT_CONDS_GLY_PDO_DCW.values()]
input_train = np.array(input_train)
input_test = np.array(input_test)

ftest = np.concatenate(ftest)
ftrain = np.concatenate(ftrain)

PARAMETER_BOUNDS_ARRAY = np.array([param_val for param_val in PARAMETER_LOG_UNIF_BOUNDS.values()])



etahat = np.ones(NPARAMS+2)



# prep training data and arrays
ftrain_means = ftrain.mean(axis=0)
ftrain_centered = ftrain - ftrain_means[np.newaxis,:]

# do initial fit
fitted_info = fitGP(input_train, ftrain_centered, init_logeta=np.log(etahat), 
					lowerb=-np.ones(NPARAMS+2), upperb=np.ones(NPARAMS+2))
input_train_prev = input_train
ftrain_prev = ftrain
fitted_info_prev = fitted_info
predmean = np.zeros(ftest.shape)
for k in range(input_test.shape[0]):
	predinfo = predictGP(fitted_info, input_test[k].reshape(1,-1), input_train, ftrain)
	predmean[k,] = predinfo['pred_mean']
rmse = np.mean((predmean+ftrain_means[np.newaxis,:]-ftest)**2) 
rmse_array = []
rmse_array.append(rmse)
while(input_train.shape[0] < max_training_length):

	etahat = fitted_info['etahat']
	etahat1 = etahat[:NPARAMS]
	etahat2 = etahat[NPARAMS]
	etahat3 = etahat[NPARAMS+1]
	sigmahat = fitted_info['sigmahat']

	#create correlation matrix
	corr_tr = corr(input_train, input_train, etahat1)

	def neg_log_det_fun(param_new): 
		p_new = np.array([np.concatenate((param_new,cond)) for cond in INIT_CONDS_GLY_PDO_DCW.values()])
		corr_tr_new = corr(input_train, p_new, etahat1)
		corr_new_new = corr(p_new, p_new, etahat1)
		corr_prods = np.matmul(corr_tr_new.T,np.linalg.solve(corr_tr,corr_tr_new))
		cov_mat = sigmahat*(corr_new_new - corr_prods)
		return -np.log(np.linalg.det(cov_mat))

	######################################################################################################
	#################################### CHOOSE DESIGN POINT #############################################
	######################################################################################################
	rmse_prev = np.inf	
	params_init = lhs.generate(space.dimensions, ninitial)
	for i in range(ninitial):
		# find proposed point
		param_prop = minimize(neg_log_det_fun, params_init[i], method="L-BFGS-B",
							  bounds= [[0.,1.] for _ in range(len(QoI_PARAMETER_LIST))],
						      options={'maxiter': 10**3}).x
		param_prop_full = np.array([np.concatenate((param_prop,cond)) for cond in INIT_CONDS_GLY_PDO_DCW.values()])
		
		# adjust data and refit GP with proposed data point
		input_train_prop = np.concatenate((input_train,param_prop_full))
		try:
			y_prop = generate_data(param_prop,dhaB_dhaT_model,tol = tol)
		except TypeError:
			print(param_prop)
			continue
		ftrain_prop = np.concatenate((ftrain, y_prop))

		ftrain_prop_means = ftrain_prop.mean(axis=0)
		ftrain_prop_centered = ftrain_prop - ftrain_prop_means[np.newaxis,:]

		fitted_info_prop = fitGP(input_train_prop, ftrain_prop_centered, init_logeta=np.log(etahat), 
								lowerb=-np.ones(NPARAMS+2), upperb=np.ones(NPARAMS+2))

		# sanity check rmse should be zero
		predmean_prop = np.zeros(ftrain_prop.shape)
		for k in range(input_train_prop.shape[0]):
			predinfo_prop = predictGP(fitted_info_prop, input_train_prop[k].reshape(1,-1), input_train_prop, ftrain_prop_centered)
			predmean_prop[k,] = predinfo_prop['pred_mean']
		rmse_prop = np.mean((predmean_prop-ftrain_prop_centered)**2)
		print(rmse_prop)
		# compute rmse test of new potential fit
		predmean_prop = np.zeros(ftest.shape)
		for k in range(input_test.shape[0]):
			predinfo_prop = predictGP(fitted_info_prop, input_test[k].reshape(1,-1), input_train_prop, ftrain_prop_centered)
			predmean_prop[k,] = predinfo_prop['pred_mean']
		rmse_prop = np.mean((predmean_prop+ftrain_means[np.newaxis,:]-ftest)**2)

		if rmse_prop < rmse_prev:
			rmse_prev = rmse_prop
			input_train_prev = input_train_prop
			ftrain_prev = ftrain_prop
			fitted_info_prev = fitted_info_prop
		print(rmse_prop)
		print(rmse_prev)


	#update data
	rmse_array.append(rmse_prev)
	input_train = input_train_prev
	ftrain = ftrain_prev
	fitted_info = fitted_info_prev
date_string = time.strftime("%Y_%m_%d_%H:%M")
file_name= 'separableGPtraining' + '_nsamples_'+ str(max_training_length/4) +'_date_'+date_string + "_rank_" + str(rank) 

plt.scatter(range(len(rmse_array)),rmse_array)
plt.savefig('emulator_MCMC_results_plots/' +file_name + '.jpg',bbox_inches='tight')

dict_data = {'input_training_set': input_train,
			 'ftrain': ftrain,
			 'fitted_info': fitted_info}

save_obj(dict_data,'emulator_MCMC_results_data/'+ file_name)


