import matplotlib as mpl
import numpy as np

mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}'] #for \text command
import scipy.optimize as opt
from qoi_dhab_dhaT_model import *
from active_subspaces import FUNCS_TO_NAMES
import seaborn as sns
import scipy.stats as stats

class ReducedQoI(QoI):
    def __init__(self, cost_matrices, n_inactive_samples, 
                start_time,final_time,integration_tol, nintegration_samples, tolsolve,
                params_values_fixed, param_sens_name, 
                max_eig_inds= None,
                external_volume = 9e-6, rc = 0.375e-6, lc = 2.47e-6, rm = 7.e-8, 
                ncells_per_metrecubed =8e14, cellular_geometry = "rod", transform = "identity"):
        """
        :params cost_matrices: dictionary of cost matrices associated with Active Subspaces for each QoI
        :params n_inactive_samples:  number of samples to average over the z direction

        :params start_time: initial time of the system -- cannot be 0
        :params final_time: final time of the system 
        :params integration_tol: integration tolerance
        :params nintegration_samples: number of samples of time samples
        :params tolsolve: tolerance for break condition
        :params params_values_fixed: dictionary parameters whose senstivities are not being studied and 
                                     their values
        :params param_sens_name: bounds of parameters whose sensitivities are being studied
        :params max_eig_inds: index of the eigenvalue partition
        :params external_volume: external volume of the system
        :params rc: radius of system
        :params lc: length of the cylindrical component of cellular_geometry = 'rod'
        :params rm: radius of MCP
        :params ncells_per_metrecubed: number of cells per m^3
        :params cellular_geometry: geometry of the cell, rod (cylinder with hemispherical ends)/sphere
        :params tranform: transformation of the parameters, log2, log10, identity or mixed.   
        """

        # load previous z samples
        self.n_inactive_samples = n_inactive_samples

        # compute and store eigenvectors
        super().__init__(cost_matrices, start_time,final_time, integration_tol, 
                        nintegration_samples, tolsolve, params_values_fixed,
                        param_sens_name, external_volume, rc, lc, rm, 
                        ncells_per_metrecubed, cellular_geometry, transform)

        #dimensions of reduced and full sample for each QoI
        if max_eig_inds is None:
            self.max_eig_inds = {qoi_name:(np.argmax(np.cumsum(eigvals)/eigvals.sum() > 0.9)+1) for qoi_name,eigvals in self.eigenvalues_QoI.items()}
        else:
            self.max_eig_inds = max_eig_inds

        self.param_dim = len(param_sens_name)
        self.active_param_dims = self.max_eig_inds
        self.inactive_param_dims = {func_name: (self.param_dim - max_ind) for func_name,max_ind in self.max_eig_inds.items()}
        self._partition_eigenvectors()
        self._generate_optimization_parameters()

    def _partition_eigenvectors(self):
        """
        Partitions eigenspace by the number of active directions for each QoI
        """
        W1 = {}
        W2 = {}
        for func_name in QOI_NAMES:
            eigvecs = self.eigenvectors_QoI[func_name]
            W1[func_name] = eigvecs[:,:self.active_param_dims[func_name]]
            W2[func_name] = eigvecs[:,self.active_param_dims[func_name]:]
        self.W1 = W1
        self.W2 = W2


    def _generate_optimization_parameters(self):
        """
        generates optimization vectors and matrices to find the inactive coordinate bounds
        given an active coordinate
        """
        # cost fun
        max_c = {}
        min_c = {}

        #slack constraint
        slack_constraint_matrix = {}
        slack_constraint_ub = {}
        #inactive constraint
        inactive_constraint_matrix = {}

        for func_name in QOI_NAMES:

            # get size of z dimension
            z_dim = self.inactive_param_dims[func_name]
            param_dim = self.param_dim
            # store constraints for slack matrix
            slack_constraint_matrix[func_name] = np.concatenate((-np.ones((z_dim,1)),np.eye(z_dim)),axis=1)
            slack_constraint_ub[func_name] = np.zeros((z_dim)) 

            # store constraints for inactive matrix
            W2 = self.W2[func_name]
            inactive_constraint_matrix[func_name] = np.concatenate((np.zeros((param_dim,1)),W2),axis=1)
            
            # cost functions
            max_c[func_name] = np.zeros(z_dim+1)
            max_c[func_name][0] = -1
            min_c[func_name] = np.zeros(z_dim+1)
            min_c[func_name][0] = 1

        self.max_c = max_c
        self.min_c = min_c
        self.slack_constraint_matrix = slack_constraint_matrix
        self.slack_constraint_ub = slack_constraint_ub
        self.inactive_constraint_matrix = inactive_constraint_matrix

    def _generate_inactive_bounds(self,active_vals):
        """
        Generates the inactive bounds given the active coordinates, active_vals
        @param active_vals: dictionary of active coordinates with each QoI
        @return: dictionary of inactive coordinates bounds for each QoI
        """
        inactive_bounds = {func_name: [np.inf,-np.inf] for func_name in QOI_NAMES}

        for func_name in active_vals.keys():

            inactive_dim = self.inactive_param_dims[func_name]

            #generate active component of x
            W1 = self.W1[func_name]
            active_in_param = np.dot(W1.T,active_vals[func_name])

            # upper and lower bound in actove constraint matrix
            inactive_ub_constraint = np.ones(self.param_dim) - active_in_param
            inactive_lb_constraint = np.ones(self.param_dim) + active_in_param
            for j in range(inactive_dim):
                #set equality constraint
                slack_constraint_matrix_eq = np.array([self.slack_constraint_matrix[func_name][j,:]])
                slack_constraint_eq = self.slack_constraint_ub[func_name][j]
                #set inequality constraint for max decision variable
                slack_constraint_matrix_ineq = np.delete(self.slack_constraint_matrix[func_name],j,axis=0)
                slack_constraint_ineq = np.delete(self.slack_constraint_ub[func_name],j,axis=0)

                # final constraints for max problem
                mat_ineq = np.concatenate((slack_constraint_matrix_ineq,
                                         self.inactive_constraint_matrix[func_name],
                                         -self.inactive_constraint_matrix[func_name]))
                ub_ineq = np.concatenate((slack_constraint_ineq,
                                          inactive_ub_constraint,
                                          inactive_lb_constraint))
                bounds_var = [(None,None) for _ in range(inactive_dim+1)]

                # find max
                res_max = opt.linprog(self.max_c[func_name], A_ub=mat_ineq, b_ub=ub_ineq,
                                      A_eq=slack_constraint_matrix_eq, b_eq=slack_constraint_eq,
                                      bounds=bounds_var)
                # inactive upper bound 
                if res_max.status == 0:
                    if inactive_bounds[func_name][1] < -res_max.fun:
                        inactive_bounds[func_name][1] = -res_max.fun

                # final constraints for min problem
                mat_ineq = np.concatenate((-slack_constraint_matrix_ineq,
                                         self.inactive_constraint_matrix[func_name],
                                         -self.inactive_constraint_matrix[func_name]))

                # find min
                res_min = opt.linprog( self.min_c[func_name], A_ub=mat_ineq, b_ub=ub_ineq,
                                        A_eq=slack_constraint_matrix_eq, b_eq=slack_constraint_eq, 
                                        bounds=bounds_var)
                # inactive lower bound
                if res_min.status == 0:
                    if res_min.fun < inactive_bounds[func_name][0]:
                        inactive_bounds[func_name][0] = res_min.fun
        return inactive_bounds

    def _generate_chebyshev_center(active_vals, self):
        """
        Generates the Chebyshev center of inactive coordinate space given an active coordinate
        for each QoI
        @param active_vals: dictionary of active coordinates with each QoI
        @return: dictionary of Chebyshev center for each QoI
        """
        #TODO: Check if this does what i would like it to do
        chebyshev_centers = {}
        for func_name in active_vals.keys():

            inactive_dim = self.inactive_param_dims[func_name]

            # cost function
            c_chebyshev = np.zeros(inactive_dim + 1)
            c_chebyshev[1] = -1

            # create bounds for z space
            inactive_constraint_matrix = self.inactive_constraint_matrix[func_name]
            W1 = self.W1[func_name]
            active_in_param = np.dot(W1.T,active_vals[func_name])

            # upper and lower bound in active constraint matrix
            inactive_ub_constraint = np.ones(self.param_dim) - active_in_param
            inactive_lb_constraint = np.ones(self.param_dim) + active_in_param
            inactive_ineqs_b = np.concatenate((inactive_ub_constraint, inactive_lb_constraint))

            # generate chebyshev inequality matrix
            row_norm_inactive_constraint_matrix = np.linalg.norm(inactive_constraint_matrix[:, 1:], axis=1)
            row_norm_inactive_constraint_matrix = np.array([np.concatenate((row_norm_inactive_constraint_matrix,
                                                                            row_norm_inactive_constraint_matrix))]).T
            inactive_ineqs_mat = np.concatenate((inactive_constraint_matrix[:, 1:], -inactive_constraint_matrix[:, 1:]))
            chebyshev_ineqs_mat = np.concatenate((inactive_ineqs_mat,row_norm_inactive_constraint_matrix),axis=1)

            # bounds
            bounds_var = [(None, None) for _ in range(inactive_dim + 1)]
            res = opt.linprog(c_chebyshev, A_ub=chebyshev_ineqs_mat, b_ub=inactive_ineqs_b,bounds=bounds_var)
            chebyshev_centers = res.x[:-1]
        return chebyshev_centers

    def _inactive_sampler_rejection(self,active_vals,method="rejection"):
        """
        Generates samples of the inactive coordinate samples given an active coordinate
        @param active_vals: dictionary of active coordinates with each QoI
        @return: dictionary of inactive coordinates samples for each QoI
        """
        #generate bounds of hypercube that contains z|y
        inactive_given_active_bounds = self._generate_inactive_bounds(active_vals)
        inactive_samples_dict = {}
        
        for func_name in active_vals.keys():
            inactive_dim = self.inactive_param_dims[func_name]
            min_edge,max_edge = inactive_given_active_bounds[func_name]
            inactive_samples = []
            if not (max_edge == -np.inf) and  not (min_edge == np.inf):
                # create bounds for z space
                inactive_constraint_matrix = self.inactive_constraint_matrix[func_name]
                active_in_param = np.dot(self.W1[func_name].T, active_vals[func_name])
                # upper and lower bound in actove constraint matrix
                inactive_ub_constraint = np.ones(self.param_dim) - active_in_param
                inactive_lb_constraint = np.ones(self.param_dim) + active_in_param

                inactive_ineqs_mat = np.concatenate((inactive_constraint_matrix[:,1:], -inactive_constraint_matrix[:,1:]))
                inactive_ineqs_b = np.concatenate((inactive_ub_constraint,inactive_lb_constraint))

                while(len(inactive_samples) < self.n_inactive_samples):
                    inactive_pot_sample = (max_edge-min_edge)*np.random.uniform(size= inactive_dim) + min_edge
                    if (np.dot(inactive_ineqs_mat,inactive_pot_sample) - inactive_ineqs_b <= 0 ).all():
                        inactive_samples.append(inactive_pot_sample)
            inactive_samples_dict[func_name] = inactive_samples

        return inactive_samples_dict

    def _inactive_sampler_hitandrun(self, active_vals):
        """
        Generates samples of the inactive coordinate samples given an active coordinate
        @param active_vals: dictionary of active coordinates with each QoI
        @return: dictionary of inactive coordinates samples for each QoI
        """
        # generate bounds of hypercube that contains z|y
        inactive_given_active_bounds = self._generate_inactive_bounds(active_vals)
        inactive_samples_dict = {}

        for func_name in active_vals.keys():
            inactive_dim = self.inactive_param_dims[func_name]

            # create bounds for z space
            inactive_constraint_matrix = self.inactive_constraint_matrix[func_name]
            active_in_param = np.dot(self.W1[func_name].T, active_vals[func_name])
            # upper and lower bound in actove constraint matrix
            inactive_ub_constraint = np.ones(self.param_dim) - active_in_param
            inactive_lb_constraint = np.ones(self.param_dim) + active_in_param

            inactive_ineqs_mat = np.concatenate(
                (inactive_constraint_matrix[:, 1:], -inactive_constraint_matrix[:, 1:]))
            inactive_ineqs_b = np.concatenate((inactive_ub_constraint, inactive_lb_constraint))

            #MCMC walk parameters
            samples = []
            maxsamps = 1e4

            # TODO generate inactive current value
            samples.append(inactive_curr)
            # hit and run algorithm
            while (len(samples) < maxsamps):

                # generate direction
                v = np.array([0, 0, 0])
                while np.linalg.norm(v) < 0.001:
                    v = np.random.normal(size=inactive_dim)
                v = v / np.linalg.norm(v)

                # find max chord distance check
                #TODO: check it seems wrong
                chord_max_param_curr = np.inf
                for i in range(2 * inactive_dim):
                    chord_max_param = (inactive_ineqs_b[i] - np.dot(inactive_ineqs_mat[:, i],
                                                                    inactive_curr)) / np.dot(
                        inactive_ineqs_mat[:, i], v)

                    if chord_max_param < chord_max_param_curr:
                        chord_max_param_curr = chord_max_param

                #take step on chord
                random_chord_dist_param = np.random.uniform() * (
                            2 * chord_max_param_curr) - chord_max_param_curr
                inactive_curr = inactive_curr + random_chord_dist_param * v
                samples.append(inactive_curr)

            #sample from list
            inactive_samples = np.random.sample(samples, self.n_inactive_samples)

            # store inactive samples
            inactive_samples_dict[func_name] = inactive_samples
        return inactive_samples_dict

    def generate_reduced_QoI_vals(self,active_vals,gen_histogram=False):
        """
        Generates samples of the QoIs coordinate samples given an active coordinate
        @param active_vals: dictionary of active coordinates with each QoI
        @return: dictionary of  samples of each QoI
        """
        inactive_samples_dict = self._inactive_sampler(active_vals)
        red_qoi_samples_dict = {}
        mean_dict = {}
        std_dict = {}
        skew_dict = {}
        mode_dict = {}
        for func_name in active_vals.keys():
            n_sucessful_solves = 0
            red_qoi_sample_vals = [] 
            inactive_samples = inactive_samples_dict[func_name]
            if len(inactive_samples) != 0:
                for inactive_vals in inactive_samples:
                    param_unif = np.dot(self.W1[func_name].T,active_vals[func_name]) + np.dot(self.W2[func_name],inactive_vals)
                    param_dict = {param_name:param_val for param_name, param_val  in zip(self.params_sens_list,param_unif)}
                    sol_vals = self.generate_QoI_vals(param_dict)
                    if sol_vals[func_name]:
                        n_sucessful_solves += 1
                        red_qoi_sample_vals.append(sol_vals[func_name])

                mean_dict[func_name] = np.mean(red_qoi_sample_vals)
                std_dict[func_name] = np.std(red_qoi_sample_vals)
                skew_dict[func_name] = stats.skew(red_qoi_sample_vals)
                mode_dict[func_name] = stats.mode(red_qoi_sample_vals)
                if gen_histogram:
                    self._generate_histogram(red_qoi_sample_vals,func_name) #save name?
            else: 
                mean_dict[func_name] = None
                sd_dict[func_name] = None
                skew_dict[func_name] = None
                mode_dict[func_name] = None

            red_qoi_samples_dict[func_name] = red_qoi_sample_vals
        red_qoi_stats = {"mean": mean_dict,
                         "std": std_dict,
                         "skew": skew_dict,
                         "mode": mode_dict}

        return red_qoi_samples_dict, red_qoi_stats

    def _generate_histogram(self,qoi_sample_vals,func_name):
        sns.histplot(data=qoi_sample_vals,stat='probability', bins='auto', color='#0504aa',alpha=0.7)
        plt.grid(axis='y', alpha=0.75)
        plt.xlabel(func_name)
        plt.title('Histogram of '+ FUNCS_TO_NAMES[func_name] + ' for a fixed y and '+ r'$M='+str(self.n_inactive_samples) + '$'+ ' samples of z')
        plt.axvline(x=np.mean(qoi_sample_vals), color='red',linewidth=4)
        plt.show()


def main():
    directory = '/home/aarcher/Dropbox/PycharmProjects/MCP/WholeCell/DhaB_DhaT_Model/object_oriented/data/1:3'
    filename = 'log10/2021_05_04_19:41'
    name_pkl = 'sampling_rsampling_N_10000'

    with open(directory + '/'+ filename+'/' +name_pkl + '.pkl', 'rb') as f:
        pk_as = pickle.load(f)
    cost_matrices = pk_as["FUNCTION_RESULTS"]["FINAL_COST_MATRIX"]

    n_inactive_samples = 1e2
    transform = 'log10'
    start_time = (10**(-15))
    final_time = 72*HRS_TO_SECS
    integration_tol = 1e-4
    tolsolve = 1e-5
    nintegration_samples = 500
    enz_ratio_name = "1:3"
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


    red_qoi = ReducedQoI(cost_matrices, n_inactive_samples, 
                        start_time,final_time,integration_tol, nintegration_samples, tolsolve,
                        params_values_fixed, list(PARAM_SENS_LOG10_BOUNDS.keys()), 
                        transform = "log10")

    W1 = red_qoi.W1
    W2 = red_qoi.W2

    params = np.array([0.3084162660909806,-0.49136586580345176,0.5129415947320597,
                       0.3979400086720375,0.7474986422705001,0.4961407271748155,0.,
                       0.3979400086720375,0.,0.,0.3979400086720375,0.34838396084692813,
                       0.39794000867203794])

    active_params = {name:np.dot(W1.T,params.T) for name,W1 in W1.items()}
    print(active_params)
    print(red_qoi.generate_reduced_QoI_vals(active_params,gen_histogram=True))

if __name__ == '__main__':
    main()
