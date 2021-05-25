import numpy as np
from scipy.optimize import minimize
from numpy.random import standard_normal,uniform
from scipy.linalg import sqrtm

def postdraws(logpost, rprior, initial_param, lbda = None,nsamp=2000):
	"""
	nsamps MCMC draws of a exp(logpost) distribution density. This MCMC assumes
	posterior is approximately normal with some mean and covariance. It converges best under
	these conditions.  

	: logpost		: function computes the log posterior density at a point
	: rprior		: function that samples the prior distribution
	: initial_param : function that describes how to initialize optimization
	: lbda          : perturbation magnitude
	: nsamples      : number of MCMC points
	"""

	tcurr = initial_param() # initalize tcurr
	d = len(tcurr)
	if not lbda and d > 5:
		lbda = (2.38)/np.sqrt(d)

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
	cov_mat = np.cov(vec,rowvar=False)
	if np.all(np.abs(cov_mat) < 1e-15):
		sca = np.zeros((d,d))
	else:
		sca = np.real(sqrtm(cov_mat))
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

def adaptive_postdraws(logpost, initial_param, nsamp=2000, beta=0.05, lbda = 0.1):
	"""
	nsamps adpative MCMC draws of a exp(logpost) distribution density.
	See Examples of Adaptive MCMC, Chelsea Lofland (https://pdfs.semanticscholar.org/b377/fa9c1a8455a1b5696b84b34f9d6bd220793a.pdf)
	See Examples of Adaptive MCMC by Roberts and Rosenthal for source material.

	lmda is chosen to be small "since enough samples are needed to obtain a valid covariance matrix 
	before adaptation, a small covariance matrix is often used initially to encourage a high acceptance rate."
	(https://m-clark.github.io/docs/ld_mcmc/index_onepage.html#adaptive_metropolis)
	
	beta is chosen to be small. It scales the covariance of second normal distribution so that,
	in the event that the covariance learned at a point is the zero matrix, a smaller second covariance matrix
	would encourage high acceptance.

	: logpost		: function computes the log posterior density at a point
	: initial_param : function that describes how to initialize optimization	
	: nsamp      	: number of MCMC points
	: beta          : weight of fixed metropolis walk (1-beta is the weight of the empirical adaptive walk)
	: lbda          : perturbation magnitude
	"""
	tcurr = initial_param()
	d = len(tcurr)

	# first 2d steps
	lpcurr = logpost(tcurr)
	thetadraw = np.zeros((nsamp,d))
	lbda1 = lbda/np.sqrt(d)
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

	# compute cov = Z^TZ
	cov_mat = np.cov(thetadraw[:i,:],rowvar=False)
	if np.all(np.abs(cov_mat) < 1e-15):
		sca = np.zeros((d,d))
	else:
		sca = np.real(sqrtm(cov_mat))
	accepted_count = 0
	lbda2 = 2.38/np.sqrt(d)
	# start adaptive MCMC
	while(i < nsamp):
		#proposal point
		tprop = tcurr + np.dot(lbda2*(1-beta)*sca + beta*lbda1*np.identity(d),standard_normal(d))
		try:
			lpprop = logpost(tprop)
		except (ValueError, TypeError):
			continue

		# MCMC relative prop
		u = min(1,np.exp(lpprop-lpcurr))
		if (u > uniform(size=1)[0]):
			tcurr = tprop
			lpcurr = lpprop
			accepted_count +=1 

		# updates
		thetadraw[i,:]= tcurr
		cov_mat = np.cov(thetadraw[:i,:],rowvar=False)

		i+=1
		if np.all(np.abs(cov_mat) < 1e-15):
			sca = np.zeros((d,d))
		else:
			sca = np.real(sqrtm(cov_mat))

	print('Acceptance Rate: ' +str(np.round(accepted_count/nsamp,decimals=3)))

	return thetadraw

def maxpostdensity(rprior,logpost,max_opt_iters = 10, initial_param = None, 
					maxiter = 10**3, jac=None, disp = True):
	"""
	Computes the argmax of the posterior density with log probability, logpost
	@param rprior: random draw from the prior
	@param logpost: log of the posterior distribution
	@param max_opt_iters: maximum iterations of the optimizer
	@param initial_param: function that returns a starting point of the optimizer
	@param maxiter: number of runs of the optimizers
	@param jac: jacobian of the posterior density
	@param disp: Boolean display optimizer output
	@return: the argmax of the posterior density
	"""

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
		if tpropval < tmaxval:
			tmax = tprop
			tmaxval = tpropval

	return tmax

