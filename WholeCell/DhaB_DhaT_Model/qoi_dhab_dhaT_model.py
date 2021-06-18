from numpy.linalg import LinAlgError
import matplotlib as mpl
mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}'] #for \text command
import pickle
from active_subspaces_dhaT_dhaB_model import *
from misc import eig_plots
from constants import QOI_NAMES

class QoI(DhaBDhaTModelJacAS):
    def __init__(self, cost_matrices, start_time,final_time,
                integration_tol, nintegration_samples, tolsolve, params_values_fixed,
                param_sens_list, external_volume = 9e-6, rc = 0.375e-6, lc = 2.47e-6,
                rm = 7.e-8, ncells_per_metrecubed =8e14, cellular_geometry = "rod", 
                transform = "identity"):
        """
        :params cost_matrices: cost matrices associated with Active Subspaces
        :params start_time: initial time of the system -- cannot be 0
        :params final_time: final time of the system 
        :params init_conc: inital concentration of the system
        :params integration_tol: integration tolerance
        :params nsamples: number of samples of time samples
        :params params_values_fixed: dictionary parameters whose senstivities are not being studied and 
                                     their values
        :params param_sens_list: bounds of parameters whose sensitivities are being studied
        :params external_volume: external volume of the system
        :params rc: radius of system
        :params lc: length of the cylindrical component of cellular_geometry = 'rod'
        :params rm: radius of MCP
        :params ncells_per_metrecubed: number of cells per m^3
        :params cellular_geometry: geometry of the cell, rod (cylinder with hemispherical ends)/sphere
        :params transform: transformation of the parameters, log2, log10, identity or mixed.   
        """

        self.cost_matrices = cost_matrices
        super().__init__(start_time,final_time, integration_tol, nintegration_samples, 
                        tolsolve, params_values_fixed, param_sens_list, external_volume,
                        rc, lc, rm, ncells_per_metrecubed, cellular_geometry, transform)
        self._generate_eigenspace()


    def _generate_eigenspace(self):
        """
        Generate the eigenspace associated with the cost matrix of each QoI
        """

        eigenvalues_QoI = {}
        eigenvectors_QoI = {}
        for func_name in QOI_NAMES:
            eigs, eigvals = np.linalg.eigh(self.cost_matrices[func_name])
            eigenvalues_QoI[func_name] = np.flip(eigs)
            eigenvectors_QoI[func_name] = np.flip(eigvals, axis=1)
        self.eigenvalues_QoI = eigenvalues_QoI
        self.eigenvectors_QoI = eigenvectors_QoI



    def generate_QoI_vals(self,params_unif_dict):
        """
        Generate the QoI value at parameter value, params_unif_dict

        :params_unif_dict: dictionary of transformed parameters, log2, log10, identity or mixed. 
        """
        sdev = lambda t,x: self._sderiv(t,x,params_unif_dict)
        sdev_jac  = lambda t,x: self.sderiv_jac_conc_fun(t,x,params_unif_dict.values())
        x0 = np.array(self.x0(**params_unif_dict))
        event_stop = lambda t,x: self._event_stop(t,x,params_unif_dict)
        event_stop.terminal = True

        # initialize
        sol_values = {QOI_NAMES[0]: None,QOI_NAMES[1]: None, QOI_NAMES[2]: None}
        # solve ODE
        try:
            sol = solve_ivp(sdev,[0, self.final_time+1], x0, method="BDF",jac=sdev_jac, 
                            t_eval=self.time_orig, atol=self.integration_tol,
                             rtol=self.integration_tol, events=event_stop)
        except ValueError:
            return sol_values

        status, time, sol_sample = [sol.status,sol.t,sol.y.T]

        # get max 3-HPA
        index_3HPA_max = np.argmax(sol_sample[:,self.index_3HPA_cytosol]) 
        # check if derivative is 0 of 3-HPA 
        statevars_maxabs = sol_sample[index_3HPA_max,:self.nvars]
        dev_3HPA = sdev(time[index_3HPA_max],statevars_maxabs)[self.index_3HPA_cytosol]


        if 'nmcps' in self.params_values_fixed.keys():
            nmcps = self.params_values_fixed['nmcps']
        else:
            if self.transform == "log10":
                bound_mcp_a,bound_mcp_b = PARAM_SENS_LOG10_BOUNDS['nmcps']
                nmcps = 10**((params_unif_dict['nmcps'] +1)*(bound_mcp_b - bound_mcp_a)/2. + bound_mcp_a)
            elif self.transform == "log2":
                bound_mcp_a,bound_mcp_b = PARAM_SENS_LOG2_BOUNDS['nmcps']
                nmcps = 2**((params_unif_dict['nmcps'] +1)*(bound_mcp_b - bound_mcp_a)/2. + bound_mcp_a)
            else:
                bound_mcp_a,bound_mcp_b = PARAM_SENS_BOUNDS['nmcps']
                nmcps = (params_unif_dict['nmcps'] +1)*(bound_mcp_b - bound_mcp_a)/2. + bound_mcp_a
        # original mass
        ext_masses_org = x0[(self.nvars-3):self.nvars]* self.external_volume
        cell_masses_org = x0[5:8] * self.cell_volume 
        mcp_masses_org = x0[:5] * self.mcp_volume
        mass_org = ext_masses_org.sum() +  self.ncells*cell_masses_org.sum() +  self.ncells*nmcps*mcp_masses_org.sum()

        # final mass
        ext_masses_fin = sol_sample[-1,(self.nvars-3):self.nvars] * self.external_volume
        cell_masses_fin = sol_sample[-1,5:8] * self.cell_volume
        mcp_masses_fin = sol_sample[-1,:5] * self.mcp_volume
        mass_fin = ext_masses_fin.sum() + self.ncells*cell_masses_fin.sum() + self.ncells*nmcps*mcp_masses_fin.sum()
        relative_diff = mass_fin/mass_org

        # check if integrated correctly
        if (relative_diff > 0.5 and relative_diff < 1.5):

            if abs(dev_3HPA) < 1e-2:
                HPA_max = sol_sample[index_3HPA_max,self.index_3HPA_cytosol]
            else:
                HPA_max = None

            # get sensitivities of Glycerol and 1,3-PDO after 5 hrs
            if status == 0 or (time[-1] > 5*HRS_TO_SECS):
                P_ext = sol_sample[self.first_index_close_enough,self.index_1_3PDO_ext]
                G_ext = sol_sample[self.first_index_close_enough,self.index_Glycerol_ext]
            elif status == 1:
                P_ext = sol_sample[-1,self.index_1_3PDO_ext]
                G_ext = sol_sample[-1,self.index_Glycerol_ext]
            else:
                P_ext = None
                G_ext = None

            sol_values[QOI_NAMES[0]] = HPA_max
            sol_values[QOI_NAMES[1]] =  G_ext
            sol_values[QOI_NAMES[2]] =  P_ext
        return sol_values
            

def main(argv, arc):
    # get inputs
    enz_ratio_name = "1:3"

    # initialize variables
    transform = 'log10'
    start_time = (10**(-15))
    final_time = 72*HRS_TO_SECS
    integration_tol = 1e-4
    tolsolve = 1e-5
    nintegration_samples = 500
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

    params_sens_dict = {'kcatfDhaB': np.log10(660), # /seconds Input
                        'KmDhaBG': np.log10(0.7), # mM Input
                        'kcatfDhaT': np.log10(80), # /seconds
                        'KmDhaTH': np.log10(0.5), # mM
                        'KmDhaTN': np.log10(0.3), # mM
                        'NADH_MCP_INIT': np.log10(0.4),
                        'PermMCPPolar': np.log10(10**-3),
                        'PermMCPNonPolar': np.log10((10**-1)/2.),
                        'PermCellGlycerol': np.log10(1.e-5), 
                        'PermCellPDO': np.log10(1.e-4), 
                        'PermCell3HPA': np.log10((1e-2)/2.), 
                        'dPacking': np.log10(0.5),
                        'nmcps': np.log10(15)}


    params_unif = {}
    for param_name, param_val in params_sens_dict.items():
        bound_a,bound_b = PARAM_SENS_LOG10_BOUNDS[param_name]
        params_unif[param_name] = 2*(param_val - bound_a)/(bound_b - bound_a) - 1
    print(params_unif)
    directory = '/home/aarcher/Dropbox/PycharmProjects/MCP/WholeCell/DhaB_DhaT_Model/object_oriented/data/1:3'
    filename = 'log10/2021_05_04_19:41'
    name_pkl = 'sampling_rsampling_N_10000'
    with open(directory + '/'+ filename+'/' +name_pkl + '.pkl', 'rb') as f:
        pk_as = pickle.load(f)
                                
    cost_matrices = pk_as["FUNCTION_RESULTS"]["FINAL_COST_MATRIX"]
    qoi_ob = QoI(cost_matrices, start_time,final_time, integration_tol, nintegration_samples,
                 tolsolve, params_values_fixed, list(params_sens_dict.keys()), transform=transform)
    print(qoi_ob.generate_QoI_vals(params_unif))

    for func_name in QOI_NAMES:
       eig_plots(qoi_ob.eigenvalues_QoI[func_name], qoi_ob.eigenvectors_QoI[func_name],params_sens_dict.keys(),
                filename,'rsampling',func_name,enz_ratio_name,10000, threshold = 0, save=False)

if __name__ == '__main__':
    main(sys.argv, len(sys.argv))
