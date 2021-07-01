"""
This code generates the average senstivity matrix that 
most affects the model in a bounded region of parameter space.

Programme written by aarcher07
Editing History:
- 25/11/20
"""

import matplotlib as mpl
mpl.rc('text', usetex = True)
from skopt.space import Space
from dhaB_dhaT_model_jac import *
from misc import *

class DhaBDhaTModelJacAS(DhaBDhaTModelJac):
    def __init__(self,start_time,final_time,
                integration_tol, nsamples, tolsolve, params_values_fixed,
                param_sens_list, external_volume = 9e-6, 
                rc = 0.375e-6, lc = 2.47e-6, rm = 7.e-8, 
                ncells_per_metrecubed =8e14, cellular_geometry = "rod", 
                transform = "identity"):
        """
        :params start_time: initial time of the system -- cannot be 0
        :params final_time: final time of the system 
        :params init_conc: inital concentration of the system
        :params integration_tol: integration tolerance
        :params nsamples: number of samples of time samples
        :params params_values_fixed: dictionary parameters whose senstivities are not being studied and 
                                     their values
        :params param_sens_bounds: bounds of parameters whose sensitivities are being studied
        :params external_volume: external volume of the system
        :params rc: radius of system
        :params lc: length of the cylindrical component of cellular_geometry = 'rod'
        :params rm: radius of MCP
        :params ncells_per_metrecubed: number of cells per m^3
        :params cellular_geometry: geometry of the cell, rod (cylinder with hemispherical ends)/sphere
        :params transform: transformation of the parameters, log2, log10, identity or mixed.      
        """

        super().__init__(start_time,final_time,integration_tol, nsamples, tolsolve, params_values_fixed,
                        param_sens_list, external_volume, rc, lc, rm, 
                        ncells_per_metrecubed, cellular_geometry, transform)
        self._set_jacs_fun()
        self._create_jac_sens()

    def _sderiv(self,t,x,params_sens = None):
        """
        Overrides the _sderiv from dhaB_dhaT_model_local_sens_analysis.py
        :param t: time
        :param x: state variables
        :param params_sens: transformed parameter dictionary
        """
        # transform uniform parameters to associated log10,log2 or identity transform parameters
        params = unif_param_to_transform_params(params_sens,self.transform)
        return super()._sderiv(t,x,params)



def main(argv, arc):
    # get inputs
    enz_ratio_name = argv[1]
    
    # initialize variables
    transform = argv[1]
    start_time = (10**(-15))
    final_time = 72*HRS_TO_SECS
    integration_tol = 1e-6
    tolsolve = 1e-5
    nsamples = 500
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

    sample_space = Space([(-1.,1.) for _ in range(len(PARAM_SENS_MIXED_BOUNDS))])
    params_unif = {key:val for key,val in zip(PARAM_SENS_MIXED_BOUNDS.keys(), sample_space.rvs(1)[0])}
    dhaB_dhaT_model_jacobian = DhaBDhaTModelJacAS(start_time, final_time, integration_tol, nsamples, tolsolve,
                                                params_values_fixed,list(PARAM_SENS_MIXED_BOUNDS.keys()), transform = transform)
    

    jacobian_est = np.array(dhaB_dhaT_model_jacobian.jac_subset(params_unif))
    
    print(jacobian_est)


if __name__ == '__main__':
    main(sys.argv, len(sys.argv))
