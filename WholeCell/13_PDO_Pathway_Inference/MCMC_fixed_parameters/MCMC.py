import numpy as np
from scipy.optimize import minimize
from numpy.random import standard_normal,uniform,lognormal,exponential,gamma
from scipy.linalg import sqrtm
from scipy.special import lambertw
import scipy.stats as stats
import matplotlib.pyplot as plt


def postdraws(rprior,logpost,lbda = None,nsamp=2000,
			  max_opt_iters = 10,maxpostdens = True,
			  initial_param = None, maxiter = 10**3,
			  jac=None, disp=True):
	"""
	nsamps MCMC draws of a exp(logpost) distribution density. This MCMC assumes
	posterior is approximately normal with some mean and covariance. It converges best under
	these conditions.  

	: rprior		: function takes n, the number of samples, and outputs the n draws from the prior
	: logpost		: function computes the log posterior density at a point
	: nsamples      : number of MCMC points
	: lamb          : perturbation magnitude
	: max_opt_inters: number of optimization searches to find most likely point
	: initial_param : function that describes how to initialize optimization
	: maxiter       : maximum number of iterations for each optimization problem
	: jac			: jacobian of the log posterior
	"""




	if maxpostdens == True:
		tcurr = maxpostdensity(rprior,logpost,nsamp, max_opt_iters = max_opt_iters,
								initial_param = initial_param,  maxiter = maxiter, jac=jac,
								disp=disp)
	elif initial_param:
		tcurr = initial_param() # initalize tcurr
	else:
		tcurr = rprior(1).reshape(-1) 

	d = len(tcurr)
	if not lbda and d > 5:
		lbda = (2.38)/np.sqrt(d)
	else:
		lbda = 0.01


	# do MCMC with prior std
	lpcurr = logpost(tcurr)
	sca = np.std(rprior(2000),axis=0)
	vec = np.zeros((nsamp,d))
	i=0

	while(i < nsamp):
		tprop = tcurr + lbda * sca* standard_normal(d)
		try:
			lpprop= logpost(tprop)
		except (ValueError, TypeError):
			continue
		u = min(1,np.exp(lpprop-lpcurr))
		if (u > uniform(size=1)[0]):
			tcurr = tprop
			lpcurr = lpprop
		vec[i,:]= tcurr
		i+=1

	# do MCMC walk again but with learned, corrected scaling
	sca = np.real(sqrtm(np.cov(vec,rowvar=False)))
	thetadraw =  np.zeros((nsamp,d))
	i=0
	accepted_count = 0
	while(i < nsamp):
		tprop = tcurr + lbda*np.dot(sca,standard_normal(d))
		try:
			lpprop = logpost(tprop)
		except (ValueError, TypeError):
			continue

		u = min(1,np.exp(lpprop-lpcurr))
		if (u > uniform(size=1)[0]):
			tcurr = tprop
			lpcurr = lpprop
			accepted_count +=1 
		thetadraw[i,:]= tcurr
		i+=1
	print('Acceptance Rate: ' +str(np.round(accepted_count/nsamp,decimals=3)))
	print('The optimal acceptance propability is 0.234.')

	return thetadraw

def adaptive_postdraws(rprior,logpost,nsamp=2000, beta=0.05,
			  max_opt_iters = 10,initial_param = None,
			  maxiter = 10**3, jac=None):
	"""
	nsamps adpative MCMC draws of a exp(logpost) distribution density.
	See Examples of Adaptive MCMC, Chelsea Lofland (https://pdfs.semanticscholar.org/b377/fa9c1a8455a1b5696b84b34f9d6bd220793a.pdf)
	See Examples of Adaptive MCMC by Roberts and Rosenthal for source material.

	: rprior		: function takes n, the number of samples, and outputs the n draws from the prior
	: logpost		: function computes the log posterior density at a point
	: nsamples      : number of MCMC points
	: lamb          : perturbation magnitude
	: max_opt_inters: number of optimization searches to find most likely point
	: initial_param : function that describes how to initialize optimization
	: maxiter       : maximum number of iterations for each optimization problem
	: jac			: jacobian of the log posterior
	"""

	optimizelp = lambda tcurr: -logpost(tcurr)
	if initial_param:
		tcurr = initial_param()
	else:
		tcurr = rprior(1).reshape(-1) # initalize tcurr

	d = len(tcurr)

	# first 2d steps
	lpcurr = logpost(tcurr)
	thetadraw = np.zeros((nsamp,d))
	lbda1 = 0.1/np.sqrt(d)
	i=0
	while(i < min(2*d,nsamp)):
		tprop = tcurr + lbda1 * standard_normal(d)
		try:
			lpprop= logpost(tprop)
		except (ValueError, TypeError):
			continue
		u = min(1,np.exp(lpprop-lpcurr))
		if (u > uniform(size=1)[0]):
			tcurr = tprop
			lpcurr = lpprop
		thetadraw[i,:]= tcurr
		i+=1

	# do MCMC walk again but with learned, corrected scaling
	sca = np.real(sqrtm(np.cov(thetadraw[:i,:],rowvar=False)))
	accepted_count = 0
	lbda2 = 2.38/np.sqrt(d)

	while(i < nsamp):
		tprop = tcurr + np.dot(lbda2*(1-beta)*sca + beta*lbda1*np.identity(d),standard_normal(d))
		try:
			lpprop = logpost(tprop)
		except (ValueError, TypeError):
			continue

		u = min(1,np.exp(lpprop-lpcurr))
		if (u > uniform(size=1)[0]):
			tcurr = tprop
			lpcurr = lpprop
			accepted_count +=1 
		thetadraw[i,:]= tcurr
		i+=1
		sca = np.real(sqrtm(np.cov(thetadraw[:i,:],rowvar=False)))

	print('Acceptance Rate: ' +str(np.round(accepted_count/nsamp,decimals=3)))

	return thetadraw

def maxpostdensity(rprior,logpost,nsamp=2000,max_opt_iters = 10,
					initial_param = None, maxiter = 10**3, jac=None,
					disp = True):

	# set up optimization
	optimizelp = lambda tcurr: -logpost(tcurr)
	lb = np.quantile(rprior(10**6),0.001,axis=0)
	ub = np.quantile(rprior(10**6),0.999,axis=0)

	tmax = rprior(1).reshape(-1)
	tmaxval = optimizelp(tmax)

	k=0
	# run optimization max_opt_iters times
	while(k < max_opt_iters):

		# initalize optimization
		if initial_param:
			tinit = initial_param()
		else:
			tinit = rprior(1).reshape(-1)
		# do optimization
		try:
			res = minimize(optimizelp, tinit, method="L-BFGS-B", jac=jac,
							 bounds=np.concatenate((lb.reshape(-1,1),ub.reshape(-1,1)),axis=1),
							 options={'maxiter': maxiter, 'disp': disp})
			
		except (ValueError, TypeError):
			continue

		tprop = res.x
		tpropval = res.fun
		k+=1

		print(tprop)
		
		if tpropval < tmaxval:
			tmax = tprop
			tmaxval = tpropval

	return tmax

def test_norm(m = 20):
	"""
	compares the MCMC walks given a target distribution covariance diag(range(1,n+1)) and a ones 
	vector mean
	"""
	cov_true = np.diag(range(1,m+1))
	cov_inv_true = np.diag(1/np.array(range(1,m+1)))
	mean_true = np.ones(m)

	loglik =lambda y: -0.5*np.dot(y-mean_true,np.dot(cov_inv_true,y-mean_true))
	rprior =lambda n: standard_normal(n*m).reshape(n,m) + mean_true[np.newaxis,:]
	initial_param = lambda: np.ones(m)
	nsamples = 10**3

	tdraws = postdraws(rprior,loglik, lbda=None, nsamp = nsamples, maxpostdens = False, 
					   initial_param = initial_param)
	mvn_obj = stats.multivariate_normal(mean=mean_true, cov=cov_true)
	print(stats.kstest(tdraws, mvn_obj.cdf))
	tdraws = adaptive_postdraws(rprior,loglik, beta=0.05, nsamp = nsamples)
	print(stats.kstest(tdraws, mvn_obj.cdf))

def test():
	def hitf(vals):
		g = vals[:,0]
		m = vals[:,1]
		r = vals[:,2]
		coeff = vals[:,3]
		h = vals[:,4]
		a = (r**2 * np.pi * coeff)/m
		term1 = h*a**2 /g+ 1
		term2 = np.exp(-term1)
		lamW = lambertw(-term2)
		fterm = (lamW + term1)/a
		return np.real(fterm)

	def rprior(n):
		return np.array([lognormal(2,0.25, size=n),
						 exponential(1,n),
						 gamma(5,20,n),
						 uniform(0.1,2,n)]).T

	logprior=lambda theta: stats.lognorm.logpdf(theta[0],s=0.25,scale=np.exp(2))+stats.expon.logpdf(theta[1],scale=1) +stats.gamma.logpdf(theta[2],a=5,scale=1/20) +stats.uniform.logpdf(theta[3],loc=0.1,scale=2-0.1)

	h = np.array([5,10,20,30,80])
	y = np.array([1.174, 1.576, 2.065, 2.715, 4.427])
	f = lambda theta: hitf(np.concatenate((np.repeat(theta.reshape(1,-1),repeats=len(h),axis=0),h.reshape(-1,1)),1))


	loglik =lambda theta: -0.5*np.dot(y-f(theta),y-f(theta)) / 0.1**2
	logpost =lambda theta: loglik(theta) + logprior(theta)
	nsamples = 10**4
	tdraws = postdraws(rprior,logpost, lbda=100, nsamp = nsamples, disp = False)
	for i in range(tdraws.shape[1]):
		plt.plot(range(int(nsamples)),tdraws[:,i])
		plt.show()
	tdraws = adaptive_postdraws(rprior,logpost, nsamp = nsamples)
	for i in range(tdraws.shape[1]):
		plt.plot(range(int(nsamples)),tdraws[:,i])
		plt.show()
if __name__ == '__main__':
	test_norm()
	test()
