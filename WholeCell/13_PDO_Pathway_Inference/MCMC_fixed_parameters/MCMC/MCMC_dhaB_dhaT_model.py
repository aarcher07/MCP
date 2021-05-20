import matplotlib.pyplot as plt
from numpy.random import standard_normal
import matplotlib as mpl
mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'
from MCMC import postdraws,adaptive_postdraws, maxpostdensity
from mpi4py import MPI
import sys
sys.path.insert(0, '.')
from base_dhaB_dhaT_model.misc_functions import *
from base_dhaB_dhaT_model.data_set_constants import *
from prior_constants import *
import time
from pathlib import Path
from dhaB_dhaT_model_prior import DhaBDhaTModelMCMC

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

def rprior(n,transform):
	"""
	Generate n samples for the the prior distribution
	@param n: number of samples
	@param transform: "log_unif"- log uniform distribution of parameters, "log_norm" - log normal distribution,
						" " - uniform distributioon
	@return: n samples of the prior distribution
	"""
	samples = []
	if transform == "log_unif":
		for key,vals in LOG_UNIF_PRIOR_PARAMETERS.items():
			samples.append(stats.uniform.rvs(size=n))
	elif transform == "log_norm":
		for key,vals in LOG_NORM_PRIOR_PARAMETERS.items():
			samples.append(stats.norm.rvs(loc=vals[0],scale=vals[1],size=n))			
	else:
		for key,vals in PARAMETER_BOUNDS.items():
			samples.append(stats.loguniform.rvs(a=vals[0],b=vals[1],size=n))		
	return np.array(samples).T


def logprior(params,transform):
	"""
	Computes the loglikelihood of the prior distribution
	@param params:
	@param transform: "log_unif"- log uniform distribution of parameters, "log_norm" - log normal distribution,
						" " - uniform distributioon
	@return: log likelihood
	"""
	logpdf = 0
	if transform == "log_unif":
		for i,(key,vals) in enumerate(LOG_UNIF_PRIOR_PARAMETERS.items()):
			logpdf += stats.uniform.logpdf(params[i])
	elif transform == "log_norm":
		for i,(key,vals) in enumerate(LOG_NORM_PRIOR_PARAMETERS.items()):
			logpdf += stats.norm.logpdf(params[i],loc=vals[0],scale=vals[1])				
	else transform == " ":
		for i,(key,vals) in enumerate(PARAMETER_BOUNDS.items()):
			logpdf += stats.loguniform.logpdf(params[i],a=vals[0],b=vals[1])
	return logpdf

def loglik(params,dhaB_dhaT_model,sigma=[2,2,0.1]):
	"""
	Computes the log likelihood Gaussian
	@param params: parameters for the model
	@param dhaB_dhaT_model: instance of the DhaBDhaTModel class
	@param sigma: standard deviation for the external Glycerol, external 1,3-PDO and DCW
	@return: Gaussian log-likelihood
	"""
	#CALIBRATION CONSTANT
	if dhaB_dhaT_model.transform_name == 'log_unif':
		bound_a,bound_b = LOG_UNIF_PRIOR_PARAMETERS['scalar']
		scalar = 10**((bound_b - bound_a)*params[0] + bound_a) 
	elif dhaB_dhaT_model.transform_name == 'log_norm':
		scalar = 10**(params[0])
	else:
		scalar= params[0]

	# PARAMETERS FOR MODEL
	params_to_dict = {}
	params_to_dict['scalar'] = scalar
	for param,key in zip(params[1:],MODEL_PARAMETER_LIST):
		params_to_dict[key] = param

	diff_f_data = []

	for gly_cond in TIME_SAMPLES.keys():
		data_sample_df = DATA_SAMPLES[gly_cond] # experimental data

		init_conds = {'G_CYTO_INIT': 0,
					  'H_CYTO_INIT': 0,
					  'P_CYTO_INIT': 0,
					  'G_EXT_INIT': data_sample_df[0,0],
					  'H_EXT_INIT': 0,
					  'P_EXT_INIT': data_sample_df[0,1],
					  'CELL_CONC_INIT': DCW_TO_COUNT_CONC*scalar*data_sample_df[0,2]
					  } # set initial conditions

		tsamp = TIME_SAMPLES[gly_cond]
		fvals = QoI(params_to_dict,init_conds,dhaB_dhaT_model,tsamp)
		# compute difference for loglikelihood
		fvals[:,2] = fvals[:,2]/(scalar*DCW_TO_COUNT_CONC)
		sigma = np.array(sigma)
		data_diff_matrix = (fvals-data_sample_df)/sigma[np.newaxis,:]
		diff_f_data.extend(data_diff_matrix.ravel())
	return -0.5*np.dot(diff_f_data,diff_f_data) 

def test(sigma = [2,2,0.1],transform = "log_unif"):
	dhaB_dhaT_model = DhaBDhaTModel(transform=transform)

	file_name = 'MCMC_results_data/old_files/adaptive_lambda_0,01_beta_0,05_norm_nsamples_1000_sigma_[2,2,0,2]_date_2021_04_15_17_00_rank_2'
	params= load_obj(file_name)[-1]
	loglik_sigma = lambda param: loglik(param,dhaB_dhaT_model,sigma=sigma)
	logpost = lambda param: loglik_sigma(param) + logprior(param, dhaB_dhaT_model.transform_name)

	print(loglik_sigma(params))
	print(logprior(params, dhaB_dhaT_model.transform_name))
	print(logpost(params))

def argmaxdensity(argv, arc):
	"""
	Computes the argmax of the log likelihood given the prior distribution
	@param argv: argv[2:5] - standard deviation of external Glycerol, external 1,3-PDO and DCW, argv[5] - parameter distribution
	@param arc: number of parameters
	@return: argmax of the posterior density
	"""
	# get arguments 
	sigma = [float(arg) for arg in argv[2:5]]
	dhaB_dhaT_model = DhaBDhaTModelMCMC(transform=argv[5])

	# set distributions
	loglik_sigma = lambda params: loglik(params,dhaB_dhaT_model,sigma=sigma)
	logpost = lambda params: loglik_sigma(params) + logprior(params, dhaB_dhaT_model.transform_name)
	rprior_ds = lambda n: rprior(n, dhaB_dhaT_model.transform_name)

	# set inital starting point
	def initial_param():
		file_name = 'MCMC_results_data/old_files/adaptive_lambda_0,01_beta_0,05_norm_nsamples_10_sigma_[2,2,0,2]_date_2021_04_12_21_28_rank_0'
		param_start = load_obj(file_name)[-1]
		param_start = param_start + 0.1 * standard_normal(len(param_start)) 
		return param_start

	param_max = maxpostdensity(rprior_ds,logpost,max_opt_iters = 1, initial_param = initial_param,
						  maxiter = 10, jac=None, disp = True)
	return param_max

def main(argv, arc):
	"""

	@param argv:
	@param arc:
	@return:
	"""
	# get arguments 
	nsamps = int(float(argv[1]))
	sigma = [float(arg) for arg in argv[2:5]]
	transform =argv[5]
	dhaB_dhaT_model = DhaBDhaTModel(transform=transform)
	lbda = float(argv[6])
	adaptive = int(argv[7])
	if adaptive:
		beta = float(argv[8])

	# set distributions
	loglik_sigma = lambda params: loglik(params,dhaB_dhaT_model,sigma=sigma)
	logpost = lambda params: loglik_sigma(params) + logprior(params, transform)
	rprior_ds = lambda n: rprior(n, transform)

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

		folder_name += "/nsamples_" + str(nsamps) +"/" + transform[4:] + "/param_" + param_name
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
	
	folder_name += "/nsamples_" + str(nsamps) +"/" + transform[4:]
	Path(folder_name).mkdir(parents=True, exist_ok=True)
	file_name = folder_name + "/date_" +date_string  + "_rank_" + str(rank)
	save_obj(tdraws,file_name)


if __name__ == '__main__':
	test(sigma=[float(arg) for arg in sys.argv[2:5]], transform=sys.argv[5])
	main(sys.argv, len(sys.argv))
