import numpy as np
from scipy.optimize import minimize
from numpy.random import standard_normal,uniform,lognormal,exponential,gamma
from scipy.linalg import sqrtm
from scipy.special import lambertw
import scipy.stats as stats
import matplotlib.pyplot as plt

def postdraws(rprior,logpost,jac=None,nsamp=2000):
	lb = np.quantile(rprior(2000),0.001,axis=0)
	print(lb)
	ub = np.quantile(rprior(2000),0.999,axis=0)
	print(ub)

	optimizelp = lambda tcurr: -logpost(tcurr)
	tcurr = rprior(1).reshape(-1)
	p = len(tcurr)

	k=0
	while(k < 1):
		print(k)
		tprop = rprior(1).reshape(-1)
		try:
			tprop = minimize(optimizelp, tprop, method="L-BFGS-B", jac=jac,
				             bounds=np.concatenate((lb.reshape(-1,1),ub.reshape(-1,1)),axis=1),
				             options={'maxiter': 10,'disp': True}).x
			k+=1
			if(optimizelp(tprop) < optimizelp(tcurr)):
				tcurr = tprop
		except ValueError:
			continue

	lpcurr = logpost(tcurr)
	sca = np.std(rprior(2000),axis=0)
	print(sca)
	lbda = 0.01
	n = nsamp
	vec = np.zeros((n,p))
	i=0
	while(i < n):
		tprop = tcurr + lbda * sca* standard_normal(p)
		try:
			lpprop= logpost(tprop)
			print(lpprop)
			u = min(1,np.exp(lpprop-lpcurr))
			print(u)
			if (u > uniform(size=1)[0]):
				print(4)
				tcurr = tprop
				lpcurr = lpprop
				vec[i,:]= tcurr
			i+=1
		except ValueError:
			continue

	sca = sqrtm(np.cov(vec,rowvar=False))
	lmbda = 1./np.sqrt(4.)
	n = nsamp
	thetadraw =  np.zeros((n,p))
	i=0
	while(i < n):
		tprop = tcurr + lmbda*np.dot(sca,standard_normal(p))
		try:
			lpprop = logpost(tprop)
			u = min(1,np.exp(lpprop-lpcurr))
			if (u > uniform(size=1)[0]):
				tcurr = tprop
				lpcurr = lpprop
			thetadraw[i,:]= tcurr
			i+=1
		except ValueError:
			continue
	return thetadraw


def main():
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
	tdraws = postdraws(rprior,logpost, nsamp = nsamples)
	for i in range(tdraws.shape[1]):
		plt.plot(range(int(nsamples)),tdraws[:,i])
		plt.show()

if __name__ == '__main__':
    main()
