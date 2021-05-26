import matplotlib as mpl
mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'
import matplotlib.pyplot as plt
from MCMC import *
from mpi4py import MPI
import time
from os.path import dirname, abspath
ROOT_PATH =dirname(abspath(__file__))

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

def main(argv, arc):
	"""
	Generates plots and data files of MCMC walks of the posterior distribution of dhaB_dhaT_model

	@param argv[1]: number of MCMC walks
	@param argv[2:5]: standard deviation for external glycerol, 1,3-PDO and DCW
	@param argv[5]: parameter distribution -- "log_norm" or "log_unif"
	@param argv[6]: boolean for fixed or adaptive. 0 for fixed, 1 for adaptive
	@param argv[7]: fixed MCMC step size
	@param argv[8]: weighed step size between adaptive MCMC step size and fixed MCMC step size
	@param arc: number of arguments
	@return:
	"""
	# get arguments
	nsamps = int(float(argv[1]))
	sigma = [float(arg) for arg in argv[2:5]]
	transform =argv[5]
	dhaB_dhaT_model = DhaBDhaTModelMCMC(transform=transform)
	adaptive = int(argv[6])
	lbda = float(argv[7])
	if adaptive:
		beta = float(argv[8])

	# set distributions
	loglik_sigma = lambda params: loglik(params,dhaB_dhaT_model,sigma=sigma)
	logpost = lambda params: loglik_sigma(params) + logprior(params, transform)
	rprior_ds = lambda n: rprior(n, transform)

	# set inital starting point
	def initial_param():
		file_name = ROOT_PATH+'/output/MCMC_results_data/old_files/adaptive_lambda_0,01_beta_0,05_norm_nsamples_1000_sigma_[2,2,0,2]_date_2021_04_15_00_21_rank_0'
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
			folder_name = ROOT_PATH+'/output/MCMC_results_plots/'+ adapt_name + "/sigma_"  + str(np.round(sigma,decimals=3)).replace('.',',').replace(' ','') +  "/" + "lambda_" + str(lbda).replace('.',',') + "_beta_" +  str(beta).replace('.',',')
		else:
			adapt_name = "fixed"
			folder_name = ROOT_PATH+'/output/MCMC_results_plots/'+ adapt_name + "/sigma_"  + str(np.round(sigma,decimals=3)).replace('.',',').replace(' ','') +  "/" + "lambda_" + str(lbda).replace('.',',')

		folder_name += "/nsamples_" + str(nsamps) +"/" + transform[4:] + "/param_" + param_name
		Path(folder_name).mkdir(parents=True, exist_ok=True)
		file_name = folder_name +'/date_'+date_string  + "_rank_" + str(rank)+ '.png'
		plt.savefig(file_name,bbox_inches='tight')
		plt.close()

	# save pickle data
	if adaptive:
		adapt_name = "adaptive"
		folder_name = ROOT_PATH+'/output/MCMC_results_data/' + adapt_name + "/sigma_"  + str(np.round(sigma,decimals=3)).replace('.',',').replace(' ','') +  "/" + "lambda_" + str(lbda).replace('.',',') + "_beta_" +  str(beta).replace('.',',')
	else:
		adapt_name = "fixed"
		folder_name =ROOT_PATH+'/output/MCMC_results_data/' + adapt_name + "/sigma_"  + str(np.round(sigma,decimals=3)).replace('.',',').replace(' ','') + "/"+  "lambda_" + str(lbda).replace('.',',')

	folder_name += "/nsamples_" + str(nsamps) +"/" + transform[4:]
	Path(folder_name).mkdir(parents=True, exist_ok=True)
	file_name = folder_name + "/date_" +date_string  + "_rank_" + str(rank)
	save_obj(tdraws,file_name)


if __name__ == '__main__':
	main(sys.argv, len(sys.argv))
