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


import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve
import scipy.constants as constants
import sympy as sp
import scipy.sparse as sparse
import pdb
import time
import matplotlib.pyplot as plt
from sklearn.metrics import auc
from constants import *
from misc_functions import *
import sys

class DhaBDhaTModel:

    def __init__(self, rc = 0.375e-6, lc = 2.47e-6,
                 external_volume = 0.002, transform = ''):
        """
        Initializes parameters to be used numerial scheme
        :param params:
        :param external_volume:
        :param rc: Radius of cell in metres
        :param lc: length of the cell in metres (needed if assuming cells are rods)
        :param rm: Radius of MCP in metres
        :param ncells: number of cells
        :param cellular geometry: "sphere" or "rod" (cylinders with hemispherical ends)
        """
        # Integration Parameters
        self.rc = rc
        self.lc = lc
        self.external_volume = external_volume
        self.nvars = 2*3 + 1

        self.cell_volume = (4*np.pi/3)*(self.rc)**3 + (np.pi)*(self.lc - 2*self.rc)*((self.rc)**2)
        self.cell_surface_area = 2*np.pi*self.rc*self.lc
        self.nparams_sens = len(MODEL_PARAMETER_LIST)
        self.transform_name = transform
        # differential equation parameters
        self._set_param_sp_symbols()
        self._set_sens_vars()
        self._set_symbolic_state_vars()

        if transform == 'log_unif_prior':
            self.ds = self._sderiv_log_unif_prior
        elif transform == 'log_norm_prior':
            self.ds = self._sderiv_log_norm_prior
        elif transform == 'log_unif_bounds':
            self.ds = self._sderiv_log_unif_bounds
        elif transform == '':
            self.ds = self._sderiv
        else:
            raise ValueError('Unknown transform')
        self._set_fun_sderiv_jac_statevars()
        #self._set_fun_sderiv_jac_params()
        #self._create_jac_sens()

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

    def _sderiv(self,t,x,params=None):
        """
        Computes the spatial derivative of the system at time point, t
        :param t: time
        :param x: state variables
        :param params: parameter list
        """

        if params is None:
            if not self.params:
                print("Parameters have not been set.")
                return
            params = self.params

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

    def _sderiv_log_unif_prior(self,t,x,log_params):
        """
        Computes the spatial derivative of the system at time point, t, with the parameters
        [-1,1] transformed by transforming parameters into their original values in LOG_UNIF_PRIOR_PARAMETERS
        :param t: time
        :param x: state variables
        :param params_sens: [-1,1] transformed parameter list
        """
        if log_params is None:
            print("Please set the parameter values")
        params = transform_from_log_unif_prior(log_params)

        return self._sderiv(t,x,params)

    def _sderiv_log_unif_bounds(self,t,x,log_params):
        """
        Computes the spatial derivative of the system at time point, t, with the parameters
        [-1,1] transformed by transforming parameters into their original values in LOG_PARAMETER_BOUNDS
        :param t: time
        :param x: state variables
        :param params_sens: [-1,1] transformed parameter list
        """
        if log_params is None:
            print("Please set the parameter values")
        params = transform_from_log_unif_bounds(log_params)

        return self._sderiv(t,x,params)

    def _sderiv_log_norm_prior(self,t,x,log_params):
        """
        Computes the spatial derivative of the system at time point, t, with the parameters
        [-1,1] transformed by transforming parameters into their original values in LOG_NORM_PRIOR_PARAMETERS
        :param t: time
        :param x: state variables
        :param params_sens:  transformed normal parameter list
        """
        if log_params is None:
            print("Please set the parameter values")
        params = transform_from_log_norm_prior(log_params)

        return self._sderiv(t,x,params)

    def _set_symbolic_sderiv(self):
        """
        Generates the symbol differential equation
        """
        x_sp = getattr(self, 'x_sp', None)
        if x_sp is None:
            self._set_symbolic_state_vars()
        self.sderiv_symbolic = self.ds(0,self.x_sp,self.params_sens_sp_dict)


    def _set_fun_sderiv_jac_statevars(self):
        """
        Generates the symbol jacobian of the differential equation 
        wrt state variables
        """
        sderiv_symbolic = getattr(self, 'sderiv_symbolic', None)
        if sderiv_symbolic is None:
            self._set_symbolic_sderiv()
            sderiv_symbolic = self.sderiv_symbolic
        self.sderiv_jac_state_vars_sp = sp.Matrix(sderiv_symbolic).jacobian(self.x_sp)
        sderiv_jac_state_vars_fun_lam = sp.lambdify((self.x_sp,self.params_sens_sp), self.sderiv_jac_state_vars_sp, 'numpy')
        self.sderiv_jac_state_vars_fun = lambda t,x,params_sens_dict: sderiv_jac_state_vars_fun_lam(x,params_sens_dict.values())
    

    def _set_fun_sderiv_jac_params(self):
        """
        Computes the jacobian of the spatial derivative wrt concentrations (state variables)
        and parameters from create_param_symbols
        """

        # SDeriv with param vals and symbols
        sderiv_symbolic = getattr(self, 'sderiv_symbolic', None)
        if sderiv_symbolic is None:
            self._set_symbolic_sderiv()
            sderiv_symbolic = self.sderiv_symbolic
        
        # derivative of rhs wrt params
        self.sderiv_jac_params_sp = sp.Matrix(sderiv_symbolic).jacobian(self.params_sens_sp)
        sderiv_jac_params_fun_lam = sp.lambdify((self.x_sp,self.params_sens_sp), self.sderiv_jac_params_sp,'numpy')
        self.sderiv_jac_params_fun = lambda t,x,params_sens_dict: sderiv_jac_params_fun_lam(x,params_sens_dict.values())


    def dsens(self,t,xs,params_sens_dict=None):
        """
        Compute RHS of the sensitivity equation

        :param t: time
        :param xs: state variables and sensitivity variables
        :param params_sens_dict: dictionary of param values whose sensitivities are being studied
        """

        if params_sens_dict is None:
            print("Please set the parameter values for local sensitivity analysis")

        assert set(MODEL_PARAMETER_LIST) == set(list(params_sens_dict.keys()))

        # reorder params_sens_dict if necessary
        if MODEL_PARAMETER_LIST != list(params_sens_dict.keys()):
            sys.exit("The internal parameter sensitivity list and the given parameter list do not correspond")
            # params_sens_dict_sorted = {param_name:params_sens_dict[param_name] for param_name in self.params_sens_list}
            # params_sens_dict = params_sens_dict_sorted


        # get state varible and sensitivities
        x = xs[:self.nvars]
        s = xs[self.nvars:]
        dxs = []
        dxs.extend(self.ds(t, x, params_sens=params_sens_dict))

        # compute rhs of sensitivity equations
        sderiv_jac_params_fun_mat = self.sderiv_jac_params_fun(t,x,params_sens_dict)
        sderiv_jac_conc_mat = self.sderiv_jac_state_vars_fun(t,x,params_sens_dict)

        for i in range(self.nvars):
            for j in range(self.nparams_sens):
                dxs.append(np.dot(sderiv_jac_conc_mat[i,:], s[range(j,self.n_sensitivity_eqs,self.nparams_sens)])
                           + sderiv_jac_params_fun_mat[i,j])
        return dxs


    def _create_jac_sens(self):
        """
        set compute jacobian of the sensitivity equation
        """

        # create state variables
        xs_sp = np.concatenate((self.x_sp,self.sensitivity_sp))
        dsens_sym = sp.Matrix(self.dsens(0,xs_sp,self.params_sens_sp_dict))
        dsens_sym_jac = dsens_sym.jacobian(xs_sp)

        # generate jacobian
        dsens_jac_dense_mat_fun = sp.lambdify((xs_sp,self.params_sens_sp),dsens_sym_jac)
        dsens_jac_sparse_mat_fun = lambda t,xs,params_sens_dict: sparse.csr_matrix(dsens_jac_dense_mat_fun(xs,params_sens_dict.values()))
        self._dsens_jac_sparse_mat_fun = dsens_jac_sparse_mat_fun

    def dsens_jac(self,t,xs,params_sens_dict=None):
        """
        Computes the jacobian of the RHS of the sensitivity equation

        :param t: time
        :param xs: state variables and sensitivity variables
        :param params_sens_dict: dictionary of param values whose sensitivities are being studied
        """

        if params_sens_dict is None:
            print("Please set the parameter values")

        assert set(MODEL_PARAMETER_LIST) == set(list(params_sens_dict.keys()))

        # reorder params_sens_dict if necessary
        if MODEL_PARAMETER_LIST != list(params_sens_dict.keys()):
            sys.exit("The internal parameter sensitivity list and the given parameter list do not correspond")
            # params_sens_dict_sorted = {param_name:params_sens_dict[param_name] for param_name in self.params_sens_list}
            # params_sens_dict = params_sens_dict_sorted

        return self._dsens_jac_sparse_mat_fun(t,xs,params_sens_dict)

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
