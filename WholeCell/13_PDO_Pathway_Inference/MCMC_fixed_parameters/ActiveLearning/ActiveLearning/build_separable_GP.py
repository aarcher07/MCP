import numpy as np
import math
import matplotlib.pyplot as plt 
import pickle
from base_dhaB_dhaT_model.data_set_constants import TIME_EVALS, NPARAMS
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.optimize import minimize


def corr(pv1,pv2, eta1):
	"""

	@param pv1:
	@param pv2:
	@param eta1:
	@return:
	"""
	K = np.ones((pv1.shape[0],pv2.shape[0]))
	for dlcv in range(pv1.shape[1]):
		K = K*np.exp(-eta1[dlcv]*np.abs(np.subtract.outer(pv1[:,dlcv],pv2[:,dlcv])))
	return K

def timeSigma(eta2):
	"""

	@param eta2:
	@return:
	"""
	sigma_f = np.exp(-eta2*np.abs(np.subtract.outer(TIME_EVALS, TIME_EVALS)))
	return sigma_f


def varSigma(eta3):
	"""

	@param eta3:
	@return:
	"""
	indices = [1,2,3]
	sigma_f = np.exp(-eta3*np.abs(np.subtract.outer(indices, indices)))
	return(sigma_f)

def MLE(logeta, to, fo):
	"""

	@param logeta:
	@param to:
	@param fo:
	@return:
	"""
	logeta1 = logeta[:NPARAMS]
	logeta2 = logeta[NPARAMS]
	logeta3 = logeta[NPARAMS+1]

	# obtain the covariance via kronecker product
	corr0 = corr(to, to, np.exp(logeta1))# c(., .)
	timeSigma0 = timeSigma(np.exp(logeta2))# timeSigma 
	varSigma0 = varSigma(np.exp(logeta3))# varSigma

	kappa0_inv = np.kron(np.linalg.inv(varSigma0), np.kron(np.linalg.inv(timeSigma0),np.linalg.inv(corr0)))
	kappa0_inv_f0 = np.matmul(kappa0_inv, fo.flatten('F'))
	kappa0_inv_1 = np.matmul(kappa0_inv, np.ones(len(fo.flatten('F'))))# estimated values

	gammahat = np.sum(kappa0_inv_f0)/np.sum(kappa0_inv_1)
	sigmahat = np.mean((kappa0_inv_f0 - gammahat*kappa0_inv_1)*(fo.flatten('F') - gammahat))

	return gammahat, sigmahat

def negloglik(logeta, t0, f0):
	"""

	@param logeta:
	@param t0:
	@param f0:
	@return:
	"""
	n = len(f0)# obtain sigmahat for a given eta
	logeta1 = logeta[:NPARAMS]
	logeta2 = logeta[NPARAMS]
	logeta3 = logeta[NPARAMS+1]
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

def fitGP(t_tr, f_tr, init_logeta=0*np.ones(NPARAMS+2),
		  lowerb=-np.ones(NPARAMS+2), upperb=np.ones(NPARAMS+2),
		  maxiter = int(10**3)):
	"""

	@param t_tr:
	@param f_tr:
	@param init_logeta:
	@param lowerb:
	@param upperb:
	@param maxiter:
	@return:
	"""
	negloglik_log_eta = lambda log_eta: negloglik(log_eta,t_tr, f_tr)
	log_eta_hat = minimize(negloglik_log_eta, init_logeta, method="L-BFGS-B",
						   bounds=np.concatenate((lowerb.reshape(-1,1),upperb.reshape(-1,1)),axis=1),
						   options={'maxiter': maxiter}).x

	gammahat, sigmahat = MLE(log_eta_hat, t_tr, f_tr)
	return {'etahat': np.exp(log_eta_hat),'sigmahat': sigmahat, 'gammahat': gammahat}

def predictGP(fitted_info, t_test, t_tr, f_tr):# obtain the fitted info
	"""

	@param fitted_info:
	@param t_test:
	@param t_tr:
	@param f_tr:
	@return:
	"""

	etahat = fitted_info['etahat']
	etahat1 = etahat[:NPARAMS]
	etahat2 = etahat[NPARAMS]
	etahat3 = etahat[NPARAMS+1]
	sigmahat = fitted_info['sigmahat']
	gammahat = fitted_info['gammahat']

	# obtain the final covariance structure

	corr0 = corr(t_tr, t_tr, etahat1)# c(t_tr, t_tr)
	timeSigma0 = timeSigma(etahat2)# timeSigma
	varSigma0 = varSigma(etahat3)# varSigma  
	kappa0_inv = (1/sigmahat)*np.kron(np.linalg.inv(varSigma0),np.kron(np.linalg.inv(timeSigma0),np.linalg.inv(corr0)))
	kappa0_inv_f0 = np.matmul(kappa0_inv,f_tr.flatten('F') - gammahat)
	corr_testtr = corr(t_test, t_tr, etahat1)# c(t_test, t_tr)
	kappa_testtr = sigmahat*np.kron(varSigma0,(np.kron(timeSigma0,corr_testtr)))# varSigma x timeSigma   x c(t_test, t_tr)

	corr_trtest = corr(t_tr, t_test, etahat1)# c(t_tr, t_test)
	kappa_trtest = sigmahat*np.kron(varSigma0,np.kron(timeSigma0, corr_trtest)) # varSigma x timeSigma   x c(t_tr,t_test )

	# prediction vector
	gamma_pred = gammahat + np.matmul(kappa_testtr,kappa0_inv_f0)

	# covariance matrix
	corr_testtest = corr(t_test, t_test, etahat1)# c(t_test, t_test)
	kappa_testtest = sigmahat*np.kron(varSigma0,np.kron(timeSigma0, corr_testtest)) # varSigma x timeSigma x c(t_test,t_test )
	dkappa = sigmahat * kappa_testtest 
	kappa_pred = dkappa - np.matmul(np.matmul(kappa_testtr,kappa0_inv),kappa_trtest)

	return {'pred_mean': gamma_pred, 'pred_var':kappa_pred}
