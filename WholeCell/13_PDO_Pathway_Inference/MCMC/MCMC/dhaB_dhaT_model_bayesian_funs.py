from base_dhaB_dhaT_model.data_set_constants import TIME_SAMPLES, DATA_SAMPLES
from base_dhaB_dhaT_model.model_constants import MODEL_PARAMETER_LIST,DCW_TO_COUNT_CONC
from .prior_constants import LOG_UNIF_PRIOR_PARAMETERS,LOG_NORM_PRIOR_PARAMETERS,UNIF_PRIOR_PARAMETERS
import numpy as np
import scipy.stats as stats

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
		for key,vals in UNIF_PRIOR_PARAMETERS.items():
			samples.append(stats.loguniform.rvs(a=vals[0],b=vals[1],size=n))		
	return np.array(samples).T


def logprior(params,transform):
	"""
	Computes the loglikelihood of the prior distribution
	@param params: parameter values
	@param transform: "log_unif"- log uniform distribution of parameters, "log_norm" - log normal distribution,
						" " - uniform distribution
	@return: log likelihood
	"""
	logpdf = 0
	if transform == "log_unif":
		for i,(key,vals) in enumerate(LOG_UNIF_PRIOR_PARAMETERS.items()):
			logpdf += stats.uniform.logpdf(params[i])
	elif transform == "log_norm":
		for i,(key,vals) in enumerate(LOG_NORM_PRIOR_PARAMETERS.items()):
			logpdf += stats.norm.logpdf(params[i],loc=vals[0],scale=vals[1])				
	elif transform == " ":
		for i,(key,vals) in enumerate(UNIF_PRIOR_PARAMETERS.items()):
			logpdf += stats.loguniform.logpdf(params[i],a=vals[0],b=vals[1])
	else:
		raise ValueError('Unknown transform')
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
	elif dhaB_dhaT_model.transform_name == " ":
		scalar= params[0]
	else:
		raise ValueError('Unknown transform')

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
		fvals = dhaB_dhaT_model.QoI(params_to_dict,init_conds,tsamp)
		# compute difference for loglikelihood
		fvals[:,2] = fvals[:,2]/(scalar*DCW_TO_COUNT_CONC)
		sigma = np.array(sigma)
		data_diff_matrix = (fvals-data_sample_df)/sigma[np.newaxis,:]
		diff_f_data.extend(data_diff_matrix.ravel())
	return -0.5*np.dot(diff_f_data,diff_f_data) 




