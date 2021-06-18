"""
Parallelized the Active_Subspaces.py code.

Programme written by aarcher07
Editing History:
- 9/11/20
"""

from numpy.linalg import LinAlgError
import matplotlib as mpl
mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}' #for \text command
from active_subspaces_dhaT_dhaB_model import *
from misc import *
from skopt.sampler import Lhs, Sobol
from skopt.space import Space

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

class ActiveSubspaces:
    def __init__(self,jac, nfuncs,funcnames, 
                 nparams, niters=10**3, sampling = 'rsampling'):
        """
        Initializes a class that computes and ranks the average sensitivity matrix  
        of each function used to compute jac, the sensitivity matrix of the functions. 

        The average sensitivity matrix is computed using Monte Carlo Integration. 

        :params jac            : qoi values and sensitivities of the problem at hand. 
        :params nparams        : number of parameters whose senstivities being studied                         
        :params nfuncs          : number of functions whose jacobians are being evaluated
        :params niters         : maximum number of iterations
        :params dist           : distribution of the parameters
        """

        self.jac = jac
        self.nfuncs = nfuncs
        self.funcnames = funcnames
        self.nparams = nparams
        self.niters = niters
        self.sampling = sampling
        self.sample_space = Space([(-1.,1.) for _ in range(self.nparams)])
        self.param_samples = []

    def compute_cost_matrix(self):
        """
        Monte Carlo integration estimate of the cost function matrix
        """

        if rank == 0:
            #do random sampling of a parameters
            if self.sampling == "LHS":
                lhs = Lhs(lhs_type="classic", criterion=None)
                param_samples = lhs.generate(self.sample_space, self.niters)
            elif self.sampling == "rsampling":
                param_samples = self.sample_space.rvs(self.niters)
            elif self.sampling == "Sobol":
                sobol = Sobol()
                param_samples = sobol.generate(self.sample_space.dimensions, self.niters)
        
            # generate param samples split
            niters_rank0 = self.niters//size + self.niters % size
            niters_rank = self.niters//size
            count_scatter = [niters_rank0]
            count_scatter.extend((size-2)*[niters_rank])
            count_scatter = np.cumsum(count_scatter)

            param_samples_split = np.split(param_samples,count_scatter)
        else:
            param_samples_split = None
            
        #scatter parameter samples data
        param_samps = comm.scatter(param_samples_split,root=0)

        # initialize data
        param_samples_dict_rank = {qoi_name:[] for qoi_name in self.funcnames}
        param_samples_diff_dict_rank = {qoi_name:[] for qoi_name in self.funcnames}
        jac_dict_rank = {qoi_name:[] for qoi_name in self.funcnames}
        qoi_dict_rank = {qoi_name:[] for qoi_name in self.funcnames}

        

        # evaluate QoI at random sampling
        for sample in param_samps:  
            qoi_sample, jac_sample = self.jac(sample).values()
            # store output
            for qoi_name in self.funcnames:
                if not (jac_sample[qoi_name] is None):
                    param_samples_dict_rank[qoi_name].append(jac_sample[qoi_name])
                    jac_dict_rank[qoi_name].append(jac_sample[qoi_name])
                    qoi_dict_rank[qoi_name].append(qoi_sample[qoi_name])
                else:
                    param_samples_diff_dict_rank[qoi_name].append(sample)

        # gather data
        param_samples = None
        param_samples_diff_int = None
        jac_dict = None
        qoi_dict= None

        param_samples_dict = comm.gather(param_samples_dict_rank, root=0)
        params_samples_diff_dict = comm.gather(param_samples_diff_dict_rank, root=0)
        jac_dict = comm.gather(jac_dict_rank, root=0)
        qoi_dict = comm.gather(qoi_dict_rank, root=0)

        # format gathered data
        if rank == 0:
            #flatten data
            param_samples_dict_flattened = {qoi_name:[] for qoi_name in self.funcnames}
            param_samples_diff_dict_flattened = {qoi_name: [] for qoi_name in self.funcnames}
            jac_dict_flattened = {qoi_name: [] for qoi_name in self.funcnames}
            qoi_dict_flattened = {qoi_name: [] for qoi_name in self.funcnames}

            for cpurank in range(size):
                for qoi_name in self.funcnames:
                    param_samples_dict_flattened[qoi_name].extend(param_samples_dict[cpurank][qoi_name]) 
                    param_samples_diff_dict_flattened[qoi_name].extend(params_samples_diff_dict[cpurank][qoi_name])
                    jac_dict_flattened[qoi_name].extend(jac_dict[cpurank][qoi_name])
                    qoi_dict_flattened[qoi_name].extend(qoi_dict[cpurank][qoi_name])

            #compute outer product
            jac_outer_dict = {qoi_name: [] for qoi_name in self.funcnames}
            nfuncs_dict = {qoi_name: 0 for qoi_name in self.funcnames}

            for qoi_name in self.funcnames:
                for i in range(len(jac_dict_flattened[qoi_name])):
                    jac_sample = jac_dict_flattened[qoi_name][i]
                    jac_outer_dict[qoi_name].append(np.outer(jac_sample,jac_sample))
                    nfuncs_dict[qoi_name] += 1

            # compute cost matrix and norm convergence
            cost_matrix_dict = {}
            cost_matrix_cumul_dict = {}
            norm_convergence_dict = {}

            for qoi_name in self.funcnames:
                cost_cumsum = np.cumsum(jac_outer_dict[qoi_name],axis=0)/np.arange(1,nfuncs_dict[qoi_name]+1)[:,None,None]
                cost_matrix_cumul_dict[qoi_name] = cost_cumsum
                cost_matrix_dict[qoi_name] = cost_cumsum[-1,:,:]
                norm_convergence_dict[qoi_name] = np.linalg.norm(cost_cumsum,ord='fro',axis=(1,2))

            # compute variance matrix
            variance_matrix_dict = {}
            for qoi_name in self.funcnames:
                variance_mat = np.sum((jac_outer_dict[qoi_name]-cost_matrix_dict[qoi_name])**2/(nfuncs_dict[qoi_name]-1),axis=0)            
                variance_matrix_dict[qoi_name] = variance_mat

            param_results = {"PARAM_SAMPLES": param_samples_dict_flattened,
                             "DIFFICULT_PARAM_SAMPLES": param_samples_diff_dict_flattened}

            fun_results = {"NUMBER_OF_FUNCTION_SUCCESS": nfuncs_dict,
                           "NORM_OF_SEQ_OF_CUMUL_SUMS": norm_convergence_dict,
                           "SEQ_OF_CUMUL_SUMS": cost_matrix_cumul_dict, 
                           "VARIANCE_OF_ENTRIES": variance_matrix_dict,
                           "FINAL_COST_MATRIX":cost_matrix_dict}

            return {'PARAMETER_RESULTS': param_results, 'FUNCTION_RESULTS': fun_results}


def test():
    f = lambda x: np.exp(0.7*x[0] + 0.3*x[1])
    jac = lambda x:  np.array([0.7*f(x),0.3*f(x)])
    f_jac =lambda x: {'QoI_values': {"exp": f(x)}, 'jac_values': {"exp": jac(x)}}
    as_test = ActiveSubspaces(f_jac, 1,["exp"], 2,niters=int(1e3))
    results = as_test.compute_cost_matrix()
    if rank == 0:
        print(np.linalg.eig(results['FUNCTION_RESULTS']["FINAL_COST_MATRIX"]["exp"]))

def dhaB_dhaT_model(argv, arc):
    # get inputs
    enz_ratio_name = argv[1]
    niters = int(float(argv[2]))
    sampling =  argv[3]
    threshold = float(argv[4])
    # initialize variables
    transform = 'log10'
    start_time = (10**(-15))
    final_time = 100*HRS_TO_SECS
    integration_tol = 1e-3
    nsamples = 500
    tolsolve = 10**-10
    enz_ratio_name_split =  enz_ratio_name.split(":")
    enz_ratio = float(enz_ratio_name_split[0])/float(enz_ratio_name_split[1])
    params_values_fixed = {'NAD_MCP_INIT': 0.1,
                          'enz_ratio': enz_ratio,
                          'G_MCP_INIT': 0,
                          'H_MCP_INIT': 0,
                          'P_MCP_INIT': 0,
                          'G_CYTO_INIT': 0,
                          'H_CYTO_INIT': 0,
                          'P_CYTO_INIT': 0 ,
                          'G_EXT_INIT': 200,
                          'H_EXT_INIT': 0,
                          'P_EXT_INIT': 0}
    
    # create object to generate jac
    param_sens_list = list(PARAM_SENS_LOG10_BOUNDS.keys())
    dhaB_dhaT_model_jacobian_as = DhaBDhaTModelJacAS(start_time, final_time, integration_tol,
                                                     nsamples, tolsolve, params_values_fixed,
                                                     param_sens_list, transform = transform)
    def dhaB_dhaT_jac(runif):
        param_sens_dict = {param_name: val for param_name,val in zip(param_sens_list,runif)}
        return dhaB_dhaT_model_jacobian_as.jac_subset(param_sens_dict) 

    # create object to run active subspaces
    as_dhaB_dhaT_mod = ActiveSubspaces(dhaB_dhaT_jac, 3, QOI_NAMES, len(param_sens_list),niters=niters, sampling = sampling)

    # run integration
    start_time = time.time()
    results = as_dhaB_dhaT_mod.compute_cost_matrix()
    end_time = time.time()

    # gather results and output
    if rank == 0:
        param_results = results["PARAMETER_RESULTS"]
        fun_results = results["FUNCTION_RESULTS"]

        date_string = time.strftime("%Y_%m_%d_%H:%M")

        # create folder 
        params_names = param_sens_list
        folder = transform+"/"+date_string
        folder_name = os.path.abspath(os.getcwd()) + '/data/' + enz_ratio_name+ '/'+  folder
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)

        # store results
        file_name_pickle = folder_name + '/sampling_' + sampling + '_N_' + str(niters) + '.pkl'
        with open(file_name_pickle, 'wb') as f:
            pickle.dump(results, f)
        
        # save text output
        generate_txt_output(fun_results["FINAL_COST_MATRIX"], fun_results["NUMBER_OF_FUNCTION_SUCCESS"],
                            fun_results["VARIANCE_OF_ENTRIES"], param_results["DIFFICULT_PARAM_SAMPLES"],
                            folder_name, transform, PARAM_SENS_LOG10_BOUNDS, size, sampling,enz_ratio_name,
                            niters, start_time,end_time)

        # save eigenvalue plots
        generate_eig_plots_QoI(fun_results["FINAL_COST_MATRIX"],PARAM_SENS_LOG10_BOUNDS.keys(),
                               folder,sampling, enz_ratio_name, niters,threshold, save=True)

if __name__ == '__main__':
    test()
    dhaB_dhaT_model(sys.argv, len(sys.argv))

