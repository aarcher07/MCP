'''
The DhaB-DhaT model contains DhaB-DhaT reaction pathway
in the MCP; diffusion in the cell; diffusion from the cell 
in the external volume.

This model is currently in use. The DhaB-DhaT model assumes that there 
are M identical MCPs within the cytosol and N identical cells within the 
external volume. From time scsle analysis, gradients in cell are removed.

Programme written by aarcher07
Editing History:
- 28/10/20
'''

from base_dhaB_dhaT_model.dhaB_dhaT_model import DhaBDhaTModel
from base_dhaB_dhaT_model.misc_functions import *
from base_dhaB_dhaT_model.data_set_constants import *
from .constants import LOG_PARAMETERS_BOUNDS



class DhaBDhaTModelMCMC(DhaBDhaTModel):
    def __init__(self, rc = 0.375e-6, lc = 2.47e-6,
                 external_volume = 0.002, transform = ''):
        """
        Initializes parameters to be used numerial scheme
        :param rc: Radius of cell in metres
        :param lc: length of the cell in metres (needed if assuming cells are rods)
        :param external_volume: external volume containing cells in metres^3
        :param transform: transform of the parameters, log10, log10 and standardized, no transform
        """
        # Integration Parameters
        super().__init__(rc, lc,external_volume)
        self.transform_name = transform
        if transform == 'log_unif':
            self._sderiv = self._sderiv_log_unif
        elif transform == '':
            pass
        else:
            raise ValueError('Unknown transform')
        self._set_fun_sderiv_jac_statevars()

    def _sderiv_log_unif(self,t,x,log_params):
        """
        Computes the spatial derivative of the system at time point, t, with the parameters
        [-1,1] transformed by transforming parameters into their original values in LOG_PARAMETER_BOUNDS
        :param t: time
        :param x: state variables
        :param params_sens: [-1,1] transformed parameter list
        """
        if log_params is None:
            print("Please set the parameter values")
        params = transform_from_log_unif(log_params,LOG_PARAMETERS_BOUNDS)

        return self._sderiv(t,x,params)


    def generate_QoI_time_series(self,params,tsamples=TIME_EVALS,tol = 10**-5):
        """
        Generate QoI time series for all experimental conditions
        @param params: dictionary parameter values to run the model. keys of the dictionary are in model_constants.py
        @param tsamples: time samples to collect external glycerol, external 1,3-PDO and DCW
        @param tol: tolerance at which integrate the DhaBDhaTModel
        @return: glycerol, external 1,3-PDO and DCW sampled at time samples, tsamples (3*|tsamples|*|experimental conditions| vector)
        """

        #CALIBRATION CONSTANT
        if self.transform_name == 'log_unif':
            bound_a,bound_b = LOG_UNIF_PRIOR_PARAMETERS['scalar']
            scalar = 10**((bound_b - bound_a)*params[0] + bound_a)
        elif dhaB_dhaT_model.transform_name == 'log_norm':
            scalar = 10**(params[0])
        else:
            scalar= params[0]

        # PARAMETERS FOR MODEL
        params_to_dict = dict()
        params_to_dict['scalar'] = scalar
        for param,key in zip(params[1:], MODEL_PARAMETER_LIST):
            params_to_dict[key] = param
        f_data = []

        # generate model for each experiemental initial condition
        for conds in INIT_CONDS_GLY_PDO_DCW.values():
            init_conds = {'G_CYTO_INIT': 0,
                          'H_CYTO_INIT': 0,
                          'P_CYTO_INIT': 0,
                          'G_EXT_INIT': conds[0],
                          'H_EXT_INIT': 0,
                          'P_EXT_INIT': conds[1],
                          'CELL_CONC_INIT': DCW_TO_COUNT_CONC*scalar*conds[2]
                        }

            fvals = self.QoI(params_to_dict, init_conds, tsamples, tol)
            # compute difference for loglikelihood
            f_data.append(fvals.flatten('F'))

        return f_data
def main():
    external_volume = 0.002

    params_trans = {'cellperGlyMass': 10**(5.73158464),
                'PermCellGlycerol': 10**(-3.55285234),
                'PermCellPDO': 10**(-3.85344833),
                'PermCell3HPA': 10**(-2.34212333),
                'VmaxfDhaB': 10**(3.26266939), 
                'KmDhaBG': 10**(0.71152905) ,
                'VmaxfDhaT': 10**(2.85561206),
                'KmDhaTH': 10**(0.69665821),
                'VmaxfGlpK':10**(1.99560497) ,
                'KmGlpKG': 10**(-1.24867452)}

    init_conds={'G_CYTO_INIT': 0, 
                'H_CYTO_INIT': 0,
                'P_CYTO_INIT': 0,
                'G_EXT_INIT': INIT_CONDS_GLY_PDO_DCW[50][0], 
                'H_EXT_INIT': INIT_CONDS_GLY_PDO_DCW[50][1],
                'P_EXT_INIT': 0,
                'CELL_CONC_INIT': INIT_CONDS_GLY_PDO_DCW[50][2]*0.5217871564671509*DCW_TO_COUNT_CONC
                }

    ds='log_norm'

    
    if ds == 'log_unif':
        params = transform_to_log_unif(params_trans)
    elif ds == 'log_norm':
        params = transform_to_log_norm(params_trans)
    else:
        params = params_trans

    dhaB_dhaT_model = DhaBDhaTModel(external_volume=external_volume, transform=ds)

    mintime = 10**(-15)
    fintime = 12*60*60

    #################################################
    # Integrate with BDF
    #################################################


    # initial conditions
    n_compounds_cell = 3
    y0 = np.zeros(dhaB_dhaT_model.nvars)
    for i,init_names in enumerate(VARIABLE_INIT_NAMES):
        y0[i] = init_conds[init_names]  

    tol = 1e-7
    nsamples = 500
    timeorig = np.logspace(np.log10(mintime), np.log10(fintime), nsamples)

    ds = lambda t,x: dhaB_dhaT_model.ds(t,x,params)
    ds_jac = lambda t,x: dhaB_dhaT_model.sderiv_jac_state_vars_fun(t,x,params)

    sol = solve_ivp(ds,[0, fintime+1], y0, method = 'BDF', jac = ds_jac, t_eval=timeorig,
                    atol=tol,rtol=tol)

    print(sol.message)

    #################################################
    # Plot solution
    #################################################
    volcell = dhaB_dhaT_model.cell_volume
    colour = ['b','r','y','c','m']

    # rescale the solutions
    ncompounds = 3
    timeorighours = sol.t/HRS_TO_SECS
    print(sol.message)


    # external solution
    for i in range(0,3):
        yext = sol.y[3+i,:].T
        plt.plot(timeorighours,yext, colour[i])

    plt.legend(['Glycerol','3-HPA','1,3-PDO'],loc='upper right')
    plt.title('Plot of external concentration')
    plt.xlabel('time (hr)')
    plt.ylabel('concentration (mM)')
    plt.show()

    #cell solutions
    for i in range(3):
        ymcp = sol.y[i,:].T
        plt.plot(timeorighours,ymcp, colour[i])


    plt.legend(['Glycerol','3-HPA','1,3-PDO'],loc='upper right')
    plt.title('Plot of cytosol concentrations')
    plt.xlabel('time (hr)')
    plt.ylabel('concentration (mM)')
    plt.show()

    plt.plot(timeorighours,sol.y[-1,:].T/((10**-0.3531)*DCW_TO_COUNT_CONC), colour[i])
    plt.title('Plot of cell concentration')
    plt.xlabel('time (hr)')
    plt.ylabel('concentration (cell per m^3)')
    plt.show()


    #check mass balance
    ext_masses_org = y0[3:6]* external_volume
    cell_masses_org = y0[:3] * volcell 


    ext_masses_fin = sol.y[3:6, -1] * external_volume
    cell_masses_fin = sol.y[:3,-1] * volcell
    print(ext_masses_fin)
    print(ext_masses_org.sum() + external_volume*y0[-1]*cell_masses_org.sum())
    print(ext_masses_fin.sum() + external_volume*sol.y[-1,-1]*cell_masses_fin.sum())
    print(sol.y[-1,-1])
if __name__ == '__main__':
    main()
