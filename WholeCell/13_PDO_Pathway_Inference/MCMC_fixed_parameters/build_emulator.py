import numpy as np
import math
import matplotlib.pyplot as plt 
from constants import *
import pickle
from dhaB_dhaT_model import DhaBDhaTModel
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.optimize import minimize

# load data 

# synthetic parameters
filename = 'norm_sample_paramspace_len_50_date_2021_03_17_16:11.pkl'

file_name_full_param = 'data/' + "cleaned_params_init_params_"+ filename[:-4]

pv = np.loadtxt(file_name_full_param + '.csv',delimiter=",")

# QoI synthetic
filename_QoI = 'data/' + "PCGP_synthetic_data_" + filename[:-4]

QoI_syn = np.loadtxt(filename_QoI + '.csv',delimiter=",")


# time points and init conditions
time_vals = [0,2, 4, 6, 8]
init_conds = [[48.4239274209863, 0.861642364731331, 0.060301507537688],
              [57.3451166180758, 1.22448979591837, 0.100696991929568],
              [72.2779071192256, 1.49001347874035, 0.057971014492754],
              [80.9160305343512, 1.52671755725191, 0.07949305141638]]
nparams = len(PARAMETER_LIST) + len(init_conds[0])

# edit param data



def corr(pv1,pv2, eta1):
	K = np.ones((pv1.shape[0],pv2.shape[0]))
	for dlcv in range(pv1.shape[1]):
		K = K*np.exp(-eta1[dlcv]*np.abs(np.subtract.outer(pv1[:,dlcv],pv2[:,dlcv])))
	return K

def timeSigma(eta2):
	sigma_f = np.exp(-eta2*np.abs(np.subtract.outer(time_vals, time_vals)))
	return sigma_f


def varSigma(eta3):
	indices = [1,2,3]
	sigma_f = np.exp(-eta3*np.abs(np.subtract.outer(indices, indices)))
	return(sigma_f)

def MLE(logeta, to, fo):
	logeta1 = logeta[:nparams]
	logeta2 = logeta[nparams]
	logeta3 = logeta[nparams+1]

	# obtain the covariance via kronecker product
	corr0 = corr(to, to, np.exp(logeta1))# c(., .)
	timeSigma0 = timeSigma(np.exp(logeta2))# timeSigma 
	varSigma0 = varSigma(np.exp(logeta3))# varSigma

	kappa0 = np.kron(varSigma0,np.kron(timeSigma0,corr0))# varSigma x timeSigma  x c(., .) 
	kappa0_inv = np.kron(np.linalg.inv(varSigma0), np.kron(np.linalg.inv(timeSigma0),np.linalg.inv(corr0)))
	kappa0_inv_f0 = np.linalg.solve(kappa0, fo.flatten('F'))
	kappa0_inv_1 = np.linalg.solve(kappa0, np.ones(len(fo.flatten('F'))))# estimated values


	gammahat = np.sum(kappa0_inv_f0)/np.sum(kappa0_inv_1)
	sigmahat = np.mean((kappa0_inv_f0 - gammahat*kappa0_inv_1)*(fo.flatten('F') - gammahat))
	return gammahat, sigmahat

def negloglik(logeta, t0, f0):
	n = len(f0)# obtain sigmahat for a given eta
	logeta1 = logeta[:nparams]
	logeta2 = logeta[nparams]
	logeta3 = logeta[nparams+1]
	gammahat, sigmahat = MLE(logeta, t0, f0)
	corr0 = corr(t0, t0, np.exp(logeta1))# c(t0, t0)
	timeSigma0 = timeSigma(np.exp(logeta2))# Sigma
	varSigma0 = varSigma(np.exp(logeta3))# Sigma
	# obtain the log of determinants efficiently
	logdetcorr0 = np.log(np.linalg.det(corr0))
	logdettimeSigma0 =  np.log(np.linalg.det(timeSigma0))
	logdetvarSigma0 = np.log(np.linalg.det(varSigma0))

	logdetkappa0 = corr0.shape[0]*logdetcorr0 + timeSigma0.shape[0]*logdettimeSigma0 + varSigma0.shape[0]*logdetvarSigma0

	return 1/2*logdetkappa0 + n/2*np.log(sigmahat)

def fitGP(t_tr, f_tr, init_logeta=0*np.ones(nparams+2),
		  lowerb=-5*np.ones(nparams+2),
		  upperb=5*np.ones(nparams+2)):
  negloglik_log_eta = lambda log_eta: negloglik(log_eta,t_tr, f_tr)
  log_eta_hat = minimize(negloglik_log_eta, init_logeta, 
  						 method="L-BFGS-B", 
  						 bounds=np.concatenate((lowerb.reshape(-1,1),upperb.reshape(-1,1)),axis=1),
  						 options={'maxiter': int(2),'disp':True}).x
  gammahat, sigmahat = MLE(log_eta_hat, t_tr, f_tr)
  return {'etahat': np.exp(log_eta_hat),'sigmahat': sigmahat, 'gammahat': gammahat}

def predictGP(fitted_info, t_test, t_tr, f_tr):# obtain the fitted info
	etahat = fitted_info['etahat']
	etahat1 = etahat[:nparams]
	etahat2 = etahat[nparams]
	etahat3 = etahat[nparams+1]
	sigmahat = fitted_info['sigmahat']
	gammahat = fitted_info['gammahat']

	# obtain the final covariance structure

	corr0 = corr(t_tr, t_tr, etahat1)# c(t_tr, t_tr)
	timeSigma0 = timeSigma(etahat2)# timeSigma
	varSigma0 = varSigma(etahat3)# varSigma  
	kappa0_inv = (1/sigmahat)*np.kron(np.linalg.inv(varSigma0),np.kron(np.linalg.inv(timeSigma0),np.linalg.inv(corr0)))
	kappa0_inv_f0 = np.matmul(kappa0_inv,f_tr.flatten('F') - gammahat)
	corr_Tv = corr(t_test, t_tr, etahat1)# c(t_test, t_tr)
	kappa_Tv = sigmahat*np.kron(varSigma0,(np.kron(timeSigma0,corr_Tv)))# varSigma x timeSigma   x c(t_test, t_tr)

	corr_toTv = corr(t_tr, t_test, etahat1)# c(t_tr, t_test)
	kappa_toTv = sigmahat*np.kron(varSigma0,np.kron(timeSigma0, corr_toTv)) # varSigma x timeSigma   x c(t_tr,t_test )

	gamma_vec = gammahat + np.matmul(kappa_Tv,kappa0_inv_f0)

	dkappa = sigmahat * np.ones(t_test.shape[0]*f_tr.shape[1])
	kappa_vec = np.maximum(dkappa -np.sum((np.matmul(kappa_Tv,kappa0_inv)).T*(kappa_toTv),axis=0), 10**(-13))

	return {'pred_mean': gamma_vec, 'pred_var':kappa_vec}

fitted_info = fitGP(pv,QoI_syn)
print(predictGP(fitted_info, pv[0,:].reshape(1,-1), pv, QoI_syn))
print(QoI_syn[0,:])