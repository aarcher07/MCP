from numpy.random import standard_normal
import matplotlib as mpl
mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'
from MCMC import *
from base_dhaB_dhaT_model.misc_functions import *
from os.path import dirname, abspath
ROOT_PATH =dirname(dirname(abspath(__file__)))

def test(sigma = [2,2,0.1],transform = "log_unif"):
    dhaB_dhaT_model = DhaBDhaTModelMCMC(transform=transform)

    file_name = ROOT_PATH+'/output/MCMC_results_data/old_files/adaptive_lambda_0,01_beta_0,05_norm_nsamples_1000_sigma_[2,2,0,2]_date_2021_04_15_17_00_rank_2'
    params= load_obj(file_name)[-1]
    loglik_sigma = lambda param: loglik(param,dhaB_dhaT_model,sigma=sigma)
    logpost = lambda param: loglik_sigma(param) + logprior(param, dhaB_dhaT_model.transform_name)

    return loglik_sigma(params), logprior(params, dhaB_dhaT_model.transform_name), logpost(params)

def argmaxdensity(argv, arc):
    """
    Computes the argmax of the log likelihood given the prior distribution
    @param argv: argv[2:5] - standard deviation of external Glycerol, external 1,3-PDO and DCW, argv[5] - parameter distribution
    @param arc: number of parameters
    @return: argmax of the posterior density
    """
    # get arguments
    sigma = [float(arg) for arg in argv[:3]]
    dhaB_dhaT_model = DhaBDhaTModelMCMC(transform=argv[3])

    # set distributions
    loglik_sigma = lambda params: loglik(params,dhaB_dhaT_model,sigma=sigma)
    logpost = lambda params: loglik_sigma(params) + logprior(params, dhaB_dhaT_model.transform_name)
    rprior_ds = lambda n: rprior(n, dhaB_dhaT_model.transform_name)

    # set inital starting point
    def initial_param():
        file_name = ROOT_PATH+'/output/MCMC_results_data/old_files/adaptive_lambda_0,01_beta_0,05_norm_nsamples_1000_sigma_[2,2,0,2]_date_2021_04_15_00_21_rank_0'
        param_start = load_obj(file_name)[-1]
        param_start = param_start + 0.1 * standard_normal(len(param_start))
        return param_start

    param_max = maxpostdensity(rprior_ds,logpost,max_opt_iters = 1, initial_param = initial_param,
                               maxiter = 10, jac=None, disp = True)
    return param_max

if __name__ == '__main__':
    print(test(sigma=[2,2,0.2], transform='log_norm'))
    print(argmaxdensity([2,2,0.2, 'log_norm'], 4))
