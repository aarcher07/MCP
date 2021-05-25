import numpy as np
from numpy.random import standard_normal,uniform,normal,exponential,gamma
from scipy.special import lambertw
import matplotlib.pyplot as plt
from MCMC.MCMC import postdraws,adaptive_postdraws, maxpostdensity
import scipy.stats as stats

def test_norm(m = 20):
    """
    compares the MCMC walks given a target distribution covariance diag(range(1,m+1)) and a ones
    vector mean

    @param m: length of the diagonal of the diagonal matrix
    @return:
    """
    # generate covariances
    cov_true = np.diag(range(1,m+1))
    cov_inv_true = np.diag(1/np.array(range(1,m+1)))
    mean_true = np.ones(m)

    # likelihood and prior
    loglik =lambda y: -0.5*np.dot(y-mean_true,np.dot(cov_inv_true,y-mean_true))
    f = lambda n: standard_normal(n*m).reshape(n,m)
    rprior = lambda n: standard_normal(n*m).reshape(n,m) + mean_true[np.newaxis,:]
    initial_param = lambda: np.ones(m)
    nsamples = 10**4

    #fixed MCMC
    tdraws = postdraws(loglik,rprior,initial_param, nsamp = nsamples)
    for i in range(m):
        n, bins, patches = plt.hist(x=tdraws[:,i], bins='auto', color='#0504aa',
                                alpha=0.7,density=True, rwidth=0.85)
        mvn_obj = stats.multivariate_normal(mean=mean_true[i], cov=np.diag(cov_true)[i])
        plt.plot(bins, mvn_obj.pdf(bins), color='black')
        plt.show()

    #adaptive MCMC
    tdraws = adaptive_postdraws(loglik, initial_param, beta=0.05, lbda = 0.1, nsamp = nsamples)
    for i in range(m):
        n, bins, patches = plt.hist(x=tdraws[:,i], bins='auto', color='#0504aa',
                                alpha=0.7,density=True, rwidth=0.85)
        mvn_obj = stats.multivariate_normal(mean=mean_true[i], cov=np.diag(cov_true)[i])
        plt.plot(bins, mvn_obj.pdf(bins), color='black')
        plt.show()


def test():
    """
    MCMC for Bayesian inference of falling object
    """

    # create model
    def hitf(vals):
        g = np.exp(vals[:,0])
        m = np.exp(vals[:,1])
        r = np.exp(vals[:,2])
        coeff = vals[:,3]
        h = vals[:,4]
        a = (r**2 * np.pi * coeff)/m
        term1 = h*a**2 /g+ 1
        term2 = np.exp(-term1)
        lamW = lambertw(-term2)
        fterm = (lamW + term1)/a
        return np.real(fterm)

    # create prior, likelihood and posterior
    def rprior(n):
        return np.array([normal(2*np.log(2) - (1/2)*np.log(0.25 + 2**2), -2*np.log(2) + np.log(0.25 + 2**2), size=n),
                         np.log(exponential(1,n)),
                         np.log(gamma(5,20,n)),
                         uniform(0.1,2,n)]).T

    logprior=lambda theta: stats.norm.logpdf(theta[0],loc=2*np.log(2) - (1/2)*np.log(0.25 + 2**2),scale= -2*np.log(2) + np.log(0.25 + 2**2))\
                           +stats.expon.logpdf(np.exp(theta[1]),scale=1)\
                           +stats.gamma.logpdf(np.exp(theta[2]),a=5,scale=1/20) \
                           +stats.uniform.logpdf(theta[3],loc=0.1,scale=2-0.1)

    h = np.array([5,10,20,30,80])
    y = np.array([1.174, 1.576, 2.065, 2.715, 4.427])
    f = lambda theta: hitf(np.concatenate((np.repeat(theta.reshape(1,-1),repeats=len(h),axis=0),h.reshape(-1,1)),1))
    loglik =lambda theta: -0.5*np.dot(y-f(theta),y-f(theta)) / 0.1**2
    logpost =lambda theta: loglik(theta) + logprior(theta)

    # args for MCMC
    nsamples = 10**5
    tmax = maxpostdensity(rprior,logpost,disp=False)
    initial_param = lambda: tmax

    #fixed MCMC
    tdraws = postdraws(logpost,rprior, initial_param,lbda=0.5, nsamp = nsamples)
    for i in range(tdraws.shape[1]):
        plt.plot(range(int(nsamples)),tdraws[:,i])
        plt.show()

    #adaptive MCMC
    tdraws = adaptive_postdraws(logpost,initial_param, beta=0.9, lbda = 0.1, nsamp = nsamples)
    for i in range(tdraws.shape[1]):
        plt.plot(range(int(nsamples)),tdraws[:,i])
        plt.show()

if __name__ == '__main__':
    test_norm()
    test()
