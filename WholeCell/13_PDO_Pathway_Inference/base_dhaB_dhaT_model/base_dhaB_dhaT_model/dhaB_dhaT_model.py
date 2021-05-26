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


import sys
import sympy as sp
from .model_constants import *
from .data_set_constants import *
from scipy.integrate import solve_ivp

class DhaBDhaTModel:

    def __init__(self, rc = 0.375e-6, lc = 2.47e-6, external_volume = 0.002):
        """
        Initializes parameters to be used numerial scheme
        :param rc: Radius of cell in metres
        :param lc: length of the cell in metres (needed if assuming cells are rods)
        :param external_volume: external volume containing cells in metres^3
        """
        # Integration Parameters
        self.rc = rc
        self.lc = lc
        self.external_volume = external_volume
        self.nvars = 2*3 + 1

        self.cell_volume = (4*np.pi/3)*(self.rc)**3 + (np.pi)*(self.lc - 2*self.rc)*((self.rc)**2)
        self.cell_surface_area = 2*np.pi*self.rc*self.lc
        self.nparams_sens = len(MODEL_PARAMETER_LIST)

        # differential equation parameters
        self._set_param_sp_symbols()
        self._set_sens_vars()
        self._set_symbolic_state_vars()


    def _set_symbolic_state_vars(self):
        """
        Generates the symbol state variables for the model
        """
        self.x_sp = np.array(sp.symbols('x:' + str(self.nvars)))

    def _set_param_sp_symbols(self):
        """
        sets dictionary of parameters to be analyzed using sensitivity analysis
        """
        self.params_sens_sp_dict = {name:sp.symbols(name) for name in MODEL_PARAMETER_LIST}
        self.params_sens_sp = list((self.params_sens_sp_dict).values())

    def _set_sens_vars(self):
        """
        creates a list of sympy symbols for the derivative of each state vector
        wrt to parameters
        """

        self.n_sensitivity_eqs = self.nvars * self.nparams_sens
        #sensitivity variables
        self.sensitivity_sp = np.array(list(sp.symbols('s0:' + str(self.n_sensitivity_eqs))))

    def _sderiv(self, t, x, params):
        """
        Computes the spatial derivative of the system at time point, t
        :param t: time
        :param x: state variables
        :param params: parameter list
        """

        ###################################################################################
        ################################# Initialization ##################################
        ###################################################################################
     

        # Integration Parameters
        assert len(x) == self.nvars
        # differential equation parameters
        d = np.zeros((len(x))).tolist()  # convert to list to allow use of symbolic derivatives
        n_compounds_cell = 3

        #cell growth
        d[6] =  (-params['KmGlpKG']*x[6] - x[6]*x[0] + sp.sqrt(4*params['KmGlpKG']*params['VmaxfGlpK']*params['cellperGlyMass']*x[6]*x[0] +(params['KmGlpKG']*x[6]+x[6]*x[0])**2 ))/(2*params['KmGlpKG'])
        #params['VmaxfGlpK']*ratio*x[0]/(params['KmGlpKG'] + ratio*x[0]) #params['maxGrowthRate'] * x[3] /(params['saturationConstant'] + x[3])
        ratio = 1/(1+d[6]/x[6])
        

        ###################################################################################
        ################################# cytosol reactions ###############################
        ###################################################################################


        PermCellGlycerol = params['PermCellGlycerol']
        PermCell3HPA = params['PermCell3HPA']
        PermCellPDO  = params['PermCellPDO']

        R_DhaB = params['VmaxfDhaB']*ratio*x[0] / (params['KmDhaBG'] + ratio*x[0]) #+ (ratio*x[1])/params['KmDhaBH']) 
        R_DhaT = params['VmaxfDhaT']*ratio*x[1]/(params['KmDhaTH'] + ratio*x[1]) 
        R_GlpK = params['VmaxfGlpK']*ratio*x[0]/(params['KmGlpKG'] + ratio*x[0])

        cell_area_volume = self.cell_surface_area/self.cell_volume

        
        d[0] = -R_DhaB -R_GlpK + cell_area_volume * PermCellGlycerol * (x[0 + n_compounds_cell] - ratio*x[0])  # microcompartment equation for G
        d[1] =  R_DhaB -  R_DhaT + cell_area_volume * PermCell3HPA * (x[1 + n_compounds_cell] - ratio*x[1])  # microcompartment equation for H
        d[2] = R_DhaT + cell_area_volume * PermCellPDO * (x[2 + n_compounds_cell] - ratio*x[2])  # microcompartment equation for P


        #####################################################################################
        ######################### external volume equations #################################
        #####################################################################################
        d[3] = x[-1] * self.cell_surface_area * PermCellGlycerol * (ratio*x[3 - n_compounds_cell] - x[3]) 
        d[4] = x[-1] * self.cell_surface_area * PermCell3HPA * (ratio*x[4 - n_compounds_cell] - x[4]) 
        d[5] = x[-1] * self.cell_surface_area * PermCellPDO * (ratio*x[5 - n_compounds_cell] - x[5]) 
        return d

    def _set_symbolic_sderiv(self):
        """
        Generates the symbol differential equation
        """
        x_sp = getattr(self, 'x_sp', None)
        if x_sp is None:
            self._set_symbolic_state_vars()
        self.sderiv_symbolic = self._sderiv(0, self.x_sp, self.params_sens_sp_dict)


    def _set_symbolic_sderiv_conc_fun(self):
        """
        Generates the symbol jacobian of the differential equation 
        wrt state variables
        """
        sderiv_symbolic = getattr(self, 'sderiv_symbolic', None)
        if sderiv_symbolic is None:
            self._set_symbolic_sderiv()
            sderiv_symbolic = self.sderiv_symbolic
        self.sderiv_jac_conc_sp = sp.Matrix(sderiv_symbolic).jacobian(self.x_sp)
        sderiv_jac_conc_fun_lam = sp.lambdify((self.x_sp,self.params_sens_sp), self.sderiv_jac_conc_sp, 'numpy')
        self._sderiv_jac_conc_fun = lambda t,x,params_sens_dict: sderiv_jac_conc_fun_lam(x,params_sens_dict.values())

    def QoI(self,params,init_conds,tsamples=TIME_EVALS,tol = 10**-5):
        """
        Integrates the DhaB-DhaT model with parameter values, param, and returns external glycerol
         1,3-PDO and cell concentration time samples, tsamples
        @param params: dictionary parameter values to run the model. keys of the dictionary are in model_constants.py
        @param init_conds: dictionary initial conditions to run the model. keys of the dictionary are in model_constants
        @param base_dhaB_dhaT_model: instance of the DhaBDhaTModel class
        @param tsamples: time samples to collect external glycerol, external 1,3-PDO and DCW
        @param tol: tolerance at which integrate the DhaBDhaTModel
        @return: glycerol, external 1,3-PDO and DCW sampled at time samples, tsamples (3 x |tsamples| matrix)
        """
        if not hasattr(self, '_sderiv_jac_conc_fun'):
            self._set_symbolic_sderiv_conc_fun()
        # format inputs
        tsamplessecs = np.array([t*HRS_TO_SECS for t in tsamples])
        model_params = {key: val for key,val in params.items() if key != "scalar" }
        scalar = params['scalar']

        # run ODE
        ds = lambda t,x: self._sderiv(t, x, model_params)
        ds_jac = lambda t,x: self._sderiv_jac_conc_fun(t,x,model_params)
        y0 = np.zeros(len(VARIABLE_INIT_NAMES))
        for i,init_names in enumerate(VARIABLE_INIT_NAMES):
            y0[i] = init_conds[init_names]

        sol = solve_ivp(ds,[0, tsamplessecs[-1]+10], y0, method = 'BDF', jac = ds_jac, t_eval=tsamplessecs,
                        atol=tol,rtol=tol)#, events=event_stop)

        # rescale cell conc
        fdata = sol.y[DATA_COLUMNS,:].T
        fdata[:,2] = fdata[:,2]/(scalar*DCW_TO_COUNT_CONC)
        return fdata
