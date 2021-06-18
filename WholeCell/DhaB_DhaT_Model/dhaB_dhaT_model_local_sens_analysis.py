'''
Local Sensitivity Analysis of DhaB-DhaT Model with functions.
This module gives the user control over the parameters for 
which they would like to do sensitivity analysis.

Programme written by aarcher07

Editing History:
- 26/10/20
'''

from scipy.integrate import solve_ivp
import sys
from constants import *
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}' #for \text command
import sympy as sp
import scipy.sparse as sparse
import time
import math
from dhaB_dhaT_model import DhaBDhaTModel



# override ComputeEnzymeConcentrations in the original documentation
class DhaBDhaTModelLocalSensAnalysis(DhaBDhaTModel):

    def __init__(self,params_values_fixed, params_sens_list, external_volume = 9e-6,
                rc = 0.375e-6, lc = 2.47e-6, rm = 7.e-8, ncells_per_metrecubed = 8e14,
                cellular_geometry = "rod", transform = "identity"):
        """
        :params params_values_fixed:  non-sensitivity parameters
        :params params_sens_list: parameters for the sensitivity study
        :params external_volume: external voluime
        :params rc: cell radius
        :params lc: cell length
        :params rm: MCP radius
        :params ncells_per_metrecubed: number of cells per volume^3
        :params cellular_geometry: cell geometry ("rod" or "sphere")
        :params transform: parameter input: "log2", "log10", "identity" or "mixed" (identity and log10)     
        """

        # check if parameter list have the correct parameter names

        potential_param_list = []
        potential_param_list.extend(list(params_values_fixed.keys()))
        potential_param_list.extend(params_sens_list)

        assert transform in ["log2", "log10", "identity", "mixed"]    
        assert len(PARAMETER_LIST + VARIABLE_INIT_NAMES) == len(potential_param_list)
        assert set(PARAMETER_LIST + VARIABLE_INIT_NAMES) == set(potential_param_list)

        # create super
        super().__init__(None, external_volume, rc, lc, rm, ncells_per_metrecubed, 
                        cellular_geometry)

        # create parameter sens symbols and values
        self.params_sens_list = params_sens_list
        self.params_values_fixed = params_values_fixed
        self.nparams_sens = len(params_sens_list)
        self.transform = transform
        self._set_param_sp_symbols()
        self._set_sens_vars()


    def _set_param_sp_symbols(self):
        """
        sets dictionary of parameters to be analyzed using sensitivity analysis
        """
        self.params_sens_sp_dict = {name:sp.symbols(name) for name in self.params_sens_list}
        self.params_sens_sp = list((self.params_sens_sp_dict).values())


    def _set_sens_vars(self):
        """
        creates a list of sympy symbols for the derivative of each state vector
        wrt to parameters
        """

        self.n_sensitivity_eqs = self.nvars * self.nparams_sens
        #sensitivity variables
        self.sensitivity_sp = np.array(list(sp.symbols('s0:' + str(self.n_sensitivity_eqs))))

    def _sderiv_id(self,t,x,params_sens = None):
        """
        Computes the spatial derivative of the system at time point, t, with the parameters
        log10 transformed
        :param t: time
        :param x: state variables
        :param params: dictionary of parameter values
        """
        if params_sens is None:
            print("Please set the parameter values")

        params_sens_keys = set(params_sens.keys())

        # transform and set parameters
        if params_sens_keys == set(self.params_sens_list):
            params = {**self.params_values_fixed, **params_sens}
            return super()._sderiv(t,x,params = params)
        else:
            print("Internal list of parameter senstivities and given dictionary do not correspond.")

    def _sderiv_mixed(self,t,x,params_sens):
        """
        Computes the spatial derivative of the system at time point, t, with the parameters
        identity and log10 transformed
        :param t: time
        :param x: state variables
        :param params_sens: dictionary of parameter values
        """
        if params_sens is None:
            print("Please set the parameter values")
        params = {}
        for param_name, param_val in params_sens.items():
            if param_name in PERM_PARAMETER_LIST:
                params[param_name] = 10**param_val
            else:
                params[param_name] = param_val
        params = {**self.params_values_fixed, **params}

        return super()._sderiv(t,x,params)

    def _sderiv_log10(self,t,x,params_sens = None):
        """
        Computes the spatial derivative of the system at time point, t, with the parameters
        log10 transformed
        :param t: time
        :param x: state variables
        :param params: log2 transformed parameter list
        """
        if params_sens is None:
            print("Please set the parameter values")

        params_sens_keys = set(params_sens.keys())

        #transform and set parameters
        if params_sens_keys == set(self.params_sens_list):
            params_sens_log10 = {param: (10**param_val if param in PARAMETER_LIST else param_val) for param,param_val in params_sens.items() }
            params_log10 = {**self.params_values_fixed, **params_sens_log10}
            return super()._sderiv(t,x,params = params_log10)
        else:
            print("Internal list of parameter senstivities and given dictionary do not correspond.")



    def _sderiv_log2(self,t,x,params_sens = None):
        """
        Computes the spatial derivative of the system at time point, t, with the parameters
        log2 transformed
        :param t: time
        :param x: state variables
        :param params_sens: log2 transformed parameter dictionary
        """
        if params_sens is None:
            print("Please set the parameter values")

        params_sens_keys = set(params_sens.keys())
 
        # transform and set parameter
        if params_sens_keys == set(self.params_sens_list):
            params_sens_log2 = {param: (2**param_val if param in PARAMETER_LIST else param_val) for param,param_val in params_sens.items() }
            params_log2 = {**self.params_values_fixed, **params_sens_log2}
            return super()._sderiv(t,x,params = params_log2)
        else:
            print("Internal list of parameter senstivities and given dictionary do not correspond.")

    def _sderiv(self,t,x,params_sens = None):
        """
        Overrides the _sderiv from dhaB_dhaT_model.py
        :param t: time
        :param x: state variables
        :param params_sens: transformed parameter dictionary
        """

        if self.transform == "log2":
            return self._sderiv_log2(t,x,params_sens)
        elif self.transform == "log10":
            return self._sderiv_log10(t,x,params_sens)
        elif self.transform == "mixed":
            return self._sderiv_mixed(t,x,params_sens)
        elif self.transform == "identity":
            return self._sderiv_id(t,x,params_sens)

    def _set_symbolic_sderiv(self):
        """
        Overrides the super and generates the symbol differential equation with symbolic parameters
        """
        x_sp = getattr(self, 'x_sp', None)
        if x_sp is None:
            self._set_symbolic_state_vars()
        self.sderiv_symbolic = self._sderiv(0,self.x_sp,self.params_sens_sp_dict)


    def _set_symbolic_sderiv_conc_sp(self):
        """
        Generates the symbol jacobian of the differential equation 
        wrt state variables
        """
        sderiv_symbolic = getattr(self, 'sderiv_symbolic', None)
        if sderiv_symbolic is None:
            self._set_symbolic_sderiv()
            sderiv_symbolic = self.sderiv_symbolic
        self.sderiv_jac_conc_sp = sp.Matrix(sderiv_symbolic).jacobian(self.x_sp)

    def _set_symbolic_sderiv_conc_fun(self):
        """
        Generates the jacobian function of the differential equation 
        wrt state variables
        """

        sderiv_jac_conc_sp = getattr(self, 'sderiv_jac_sp', None)
        if sderiv_jac_conc_sp is None:
            self._set_symbolic_sderiv_conc_sp()
            sderiv_jac_conc_sp = self.sderiv_jac_conc_sp
        sderiv_jac_conc_fun = sp.lambdify((self.x_sp,self.params_sens_sp), sderiv_jac_conc_sp,'numpy')
        self.sderiv_jac_conc_fun = lambda t,x,params_sens_vals: sderiv_jac_conc_fun(x,params_sens_vals)

    def _set_jacs_fun(self):
        """
        Computes the jacobian of the spatial derivative wrt concentrations (state variables)
        and parameters from create_param_symbols
        """

        # SDeriv with param vals and symbols
        sderiv_symbolic = getattr(self, 'sderiv_jac_sp', None)
        if sderiv_symbolic is None:
            self._set_symbolic_sderiv()
            sderiv_symbolic = self.sderiv_symbolic

        sderiv_jac_conc_fun = getattr(self, 'sderiv_jac_conc_fun', None)
        if sderiv_jac_conc_fun is None:
            self._set_symbolic_sderiv_conc_fun()

        # derivative of rhs wrt Conc
        sderiv_jac_params = sp.Matrix(sderiv_symbolic).jacobian(self.params_sens_sp)
        sderiv_jac_params_fun = sp.lambdify((self.x_sp,self.params_sens_sp), sderiv_jac_params,'numpy')
        self.sderiv_jac_params_fun = lambda t,x,params_sens_vals: sderiv_jac_params_fun(x,params_sens_vals)

    def dsens(self,t,xs,params_sens_dict=None):
        """
        Compute RHS of the sensitivity equation

        :param t: time
        :param xs: state variables and sensitivity variables
        :param params_sens_dict: dictionary of param values whose sensitivities are being studied
        """

        if params_sens_dict is None:
            print("Please set the parameter values for local sensitivity analysis")

        assert set(self.params_sens_list) == set(list(params_sens_dict.keys()))

        # reorder params_sens_dict if necessary
        if self.params_sens_list != list(params_sens_dict.keys()):
            sys.exit("The internal parameter sensitivity list and the given parameter list do not correspond")
            # params_sens_dict_sorted = {param_name:params_sens_dict[param_name] for param_name in self.params_sens_list}
            # params_sens_dict = params_sens_dict_sorted


        # get state varible and sensitivities
        x = xs[:self.nvars]
        s = xs[self.nvars:]
        dxs = []
        dxs.extend(self._sderiv(t, x, params_sens=params_sens_dict))

        # compute rhs of sensitivity equations
        sderiv_jac_params_fun_mat = self.sderiv_jac_params_fun(t,x,params_sens_dict.values())
        sderiv_jac_conc_mat = self.sderiv_jac_conc_fun(t,x,params_sens_dict.values())
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

        assert set(self.params_sens_list) == set(list(params_sens_dict.keys()))

        # reorder params_sens_dict if necessary
        if self.params_sens_list != list(params_sens_dict.keys()):
            sys.exit("The internal parameter sensitivity list and the given parameter list do not correspond")
            # params_sens_dict_sorted = {param_name:params_sens_dict[param_name] for param_name in self.params_sens_list}
            # params_sens_dict = params_sens_dict_sorted

        return self._dsens_jac_sparse_mat_fun(t,xs,params_sens_dict)

def main(nsamples = 500):

    external_volume =  9e-6
    ncells_per_metrecubed = 8e14 # 7e13-8e14 cells per m^3
    mintime = 10**(-15)
    secstohrs = 60*60
    fintime = 20*60*60
    transform = "log10"    # parameter to check senstivity

    params_values_fixed = {'kcatfDhaT': 59.4, # /seconds
                          'KmDhaTH': 0.77, # mM
                          'KmDhaTN': 0.03, # mM
                          'enz_ratio': 1/3,
                          'NADH_MCP_INIT': 0.36,
                          'NAD_MCP_INIT': 1.,
                          'PermMCPPolar': 10**-3, 
                          'PermCellGlycerol': 10**-7,
                          'PermCell3HPA': 10**-2,          
                          'G_MCP_INIT': 0,
                          'H_MCP_INIT': 0,
                          'P_MCP_INIT': 0,
                          'G_CYTO_INIT': 0,
                          'H_CYTO_INIT': 0,
                          'P_CYTO_INIT': 0 ,
                          'G_EXT_INIT': 200,
                          'H_EXT_INIT': 0,
                          'P_EXT_INIT': 0}


    params_sens_list = ['kcatfDhaB', 'KmDhaBG', 'PermMCPNonPolar', 'PermCellPDO', 'dPacking', 'nmcps']

    # parameter to check senstivity
    params_sens_dict = {'kcatfDhaB':400, # /seconds Input
              'KmDhaBG': 0.6, # mM Input
              'PermMCPNonPolar': 10**-2, 
              'PermCellPDO': 10.**-7,
              'dPacking': 0.64,
              'nmcps': 15}
    for key in params_sens_dict.keys():
        if transform == "log2":
            params_sens_dict[key] = np.log2(params_sens_dict[key])
        if transform == "log10":
            params_sens_dict[key] = np.log10(params_sens_dict[key])

    #################################################
    # Setup Senstivity Eqs
    #################################################

    # setup differential eq
    model_local_sens = DhaBDhaTModelLocalSensAnalysis(params_values_fixed, params_sens_list, external_volume = external_volume, 
                                                     rc = 0.375e-6, lc =  2.47e-6, rm = 7.e-8,
                                                     ncells_per_metrecubed = ncells_per_metrecubed, 
                                                     cellular_geometry = "rod", transform = transform)
    model_local_sens._set_jacs_fun()
    model_local_sens._create_jac_sens()
    #parameterize sensitivity functions
    dsens_param = lambda t, xs: model_local_sens.dsens(t,xs,params_sens_dict)
    dsens_param_jac = lambda t, xs: model_local_sens.dsens_jac(t,xs,params_sens_dict)

    # initial conditions
    y0 = np.zeros(model_local_sens.nvars) 
    y0[-3] = params_values_fixed['G_EXT_INIT']  # y0[-5] gives the initial state of the external substrate.
    sens0 = np.zeros(model_local_sens.n_sensitivity_eqs)
    for i,param in enumerate(params_sens_list):
        if param in VARIABLE_INIT_NAMES:
            sens0[i::model_local_sens.nparams_sens] = 1
    ys0 = np.concatenate([y0,sens0])


    #################################################
    # Integrate BDF
    #################################################

    # solution params
    tol = 1e-3
    time_orig = np.logspace(np.log10(mintime),np.log10(fintime),nsamples)
    # terminal event
    # tol_solve = 10**-5
    # def event_stop(t,y):
    #     dSsample = np.array(model_local_sens._sderiv(t,y[:model_local_sens.nvars],
    #                                                    params_sens = params_sens_dict))
    #     dSsample_dot = np.abs(dSsample).sum()
    #     return dSsample_dot - tol_solve 
    # event_stop.terminal = True


    start_time = time.time()

    sol = solve_ivp(dsens_param,[0, fintime+10], ys0, method="BDF", 
                    jac = dsens_param_jac, #events=event_stop,
                    t_eval=time_orig, atol=tol,rtol=tol)
    end_time = time.time()
    index_max_3HPA = np.argmax(sol.y[4,:])

    #################################################
    # Plot solutions
    #################################################

    if 'nmcps' in params_values_fixed.keys():
        nmcps = params_values_fixed['nmcps']
    else:
        nmcps = params_sens_dict['nmcps']
    ncells = model_local_sens.ncells
    volcell = model_local_sens.cell_volume
    volmcp = model_local_sens.mcp_volume
    external_volume = model_local_sens.external_volume
    colour = ['b','r','y','c','m']


    print('solve time: ' + str(end_time-start_time))
    # plot state variables solution
    print(sol.message)

    # rescale the solutions
    ncompounds = 3
    time_orig_hours = sol.t/secstohrs


    #plot parameters
    namesvars = ['Glycerol', '3-HPA', '1,3-PDO']
    sens_vars_names = [r'$kcat_f^{DhaB}$', r'$K_M^{DhaB}$', r'$P_{\text{MCP},\text{Non-Polar}}$', r'$k_{c}$', r'$dPacking$', r'$MCP$']
    colour = ['b','r','y','c','m']

    # cellular solutions
    # for i in range(0,ncompounds):
    #     ycell = sol.y[5+i, :]
    #     plt.plot(sol.t/secstohrs, ycell, colour[i])
    index_max_3HPA = np.argmax(sol.y[4,:])
    print(np.max(sol.y[4,:]))
    i = 1
    ycell = sol.y[3+i, :]
    plt.axvline(x=time_orig_hours[index_max_3HPA],ls='--',ymin=0.05,color='k')

    plt.plot(sol.t/secstohrs, ycell, colour[i])
    plt.title('Plot of cellular 3-HPA concentration')
    plt.legend(['3-HPA'], loc='upper right')
    plt.xlabel('time (hr)')
    plt.ylabel('concentration (mM)')
    plt.grid() 
    plt.show()


    # plot sensitivity variable solutions for MCP variables
    for i in range(0,len(namesvars)):
        figure, axes = plt.subplots(nrows=int(math.ceil(len(params_sens_list)/2)), ncols=2, figsize=(10,10), sharex=True, sharey=True)
        soly = sol.y[(model_local_sens.nvars + i*model_local_sens.nparams_sens):(model_local_sens.nvars + (i+1)*model_local_sens.nparams_sens), :]
        maxy = np.max(soly)
        miny =np.min(soly)
        yub = 1.15*maxy if maxy > 0 else 0.85*maxy
        lub = 0.85*miny if miny > 0 else 1.15*miny
        for j in range(model_local_sens.nparams_sens):
            axes[j // 2, j % 2].plot(time_orig_hours, soly[j,:].T)
            axes[j // 2, j % 2].set_ylabel(r'$\partial (\text{' + namesvars[i] + '})/\partial \log_{10}' + sens_vars_names[j][1:])
            # axes[j // 2, j % 2].set_ylabel(r'$\partial (' + namesvars[i] + ')/\partial ' + sens_vars_names[j][1:])

            axes[j // 2, j % 2].set_title(sens_vars_names[j])
            axes[j // 2, j % 2].set_ylim([lub, yub])
            axes[j // 2, j % 2].grid()
            if j >= (model_local_sens.nparams_sens-2):
                axes[(model_local_sens.nparams_sens-1) // 2, j % 2].set_xlabel('time/hrs')

        figure.suptitle(r'Sensitivity, $\partial (\text{' + namesvars[i]+'})/\partial\log_{10} p_i$, of the MCP concentration of '
                       + namesvars[i] + ' wrt $p_i$', y = 0.92)
        # figure.suptitle(r'Sensitivity, $\partial (' + namesvars[i]+')/\partial p_i$, of the MCP concentration of '
        #                 + namesvars[i] + ' wrt $p_i$', y = 0.92)
        # plt.savefig('/Users/aarcher/PycharmProjects/MCP/WholeCell/plots/Perm_SensitivityInternal_'+ namesvars[i-2] +'.png',
        #             bbox_inches='tight')
        plt.show()


    # plot sensitivity variable solutions for cellular variables
    for i in range(len(namesvars),2*len(namesvars)):
        figure, axes = plt.subplots(nrows=int(math.ceil(len(params_sens_list)/2)), ncols=2, figsize=(10,10), sharex=True, sharey=True)
        soly = sol.y[(model_local_sens.nvars + i*model_local_sens.nparams_sens):(model_local_sens.nvars + (i+1)*model_local_sens.nparams_sens), :]
        maxy = np.max(soly)
        miny = np.min(soly)
        yub = 1.15*maxy if maxy > 0 else 0.85*maxy
        lub = 0.85*miny if miny > 0 else 1.15*miny
        for j in range(model_local_sens.nparams_sens):
            axes[j // 2, j % 2].plot(time_orig_hours, soly[j,:].T)
            axes[j // 2, j % 2].set_ylabel(r'$\partial (\text{' + namesvars[i-len(namesvars)] + '})/\partial \log_{10}' + sens_vars_names[j][1:])
            #axes[j // 2, j % 2].set_ylabel(r'$\partial (' + namesvars[i-len(namesvars)] + ')/\partial ' + sens_vars_names[j][1:])
            axes[j // 2, j % 2].set_title(sens_vars_names[j])
            axes[j // 2, j % 2].grid()
            axes[j // 2, j % 2].set_ylim([lub, yub])
            axes[j // 2, j % 2].axvline(x=time_orig_hours[index_max_3HPA],ls='--',ymin=0.05,color='k')

            if j >= (model_local_sens.nparams_sens-2):
                axes[(model_local_sens.nparams_sens-1) // 2, j % 2].set_xlabel('time/hrs')


        figure.suptitle(r'Sensitivity, $\partial (\text{' + namesvars[i-len(namesvars)]+'})/\partial \log_{10} p_i$, of the cellular concentration of '
                        + namesvars[i-len(namesvars)] + ' wrt $p_i$', y = 0.92)
        # figure.suptitle(r'Sensitivity, $\partial (' + namesvars[i-len(namesvars)]+')/\partial p_i$, of the cellular concentration of '
        #                 + namesvars[i-len(namesvars)] + ' wrt $p_i$', y = 0.92)
        # plt.savefig('/Users/aarcher/PycharmProjects/MCP/WholeCell/plots/Perm_SensitivityInternal_'+ namesvars[i-2] +'.png',
        #             bbox_inches='tight')
        plt.show()


    # sensitivity variables
    for i in range(-len(namesvars),0):
        figure, axes = plt.subplots(nrows=int(math.ceil(len(params_sens_list)/2)), ncols=2, figsize=(10,10), sharex=True,sharey=True)
        if i == -3:
            soly = sol.y[-(model_local_sens.nparams_sens):,:]
        else:
            soly = sol.y[-(i+4)*model_local_sens.nparams_sens:-(i+3)*model_local_sens.nparams_sens, :]
        maxy = np.max(soly)
        miny = np.min(soly)
        yub = 1.15*maxy if maxy > 0 else 0.85*maxy
        lub = 0.85*miny if miny > 0 else 1.15*miny
        for j in range(model_local_sens.nparams_sens):
            axes[j // 2, j % 2].plot(time_orig_hours, soly[j,:].T)
            axes[j // 2, j % 2].set_ylabel(r'$\partial (\text{' + namesvars[i+3] + '})/\partial \log_{10}' + sens_vars_names[j][1:])
            # axes[j // 2, j % 2].set_ylabel(r'$\partial (' + namesvars[i+3] + ')/\partial ' + sens_vars_names[j][1:])
            axes[j // 2, j % 2].set_ylim([lub, yub])
            axes[j // 2, j % 2].set_title(sens_vars_names[j])
            axes[j // 2, j % 2].grid()
            if j >= (model_local_sens.nparams_sens-2):
                axes[(model_local_sens.nparams_sens-1) // 2, j % 2].set_xlabel('time/hrs')

        figure.suptitle(r'Sensitivity, $\partial (\text{' + namesvars[i + 3]+'})/\partial \log_{10} p_i$, of the external concentration of '
                        + namesvars[i + 3] + ' wrt $p_i$', y = 0.92)
        # figure.suptitle(r'Sensitivity, $\partial (' + namesvars[i + 3]+')/\partial \log_{10} p_i$, of the external concentration of '
        #                 + namesvars[i + 3] + ' wrt $p_i$', y = 0.92)
        # plt.savefig('/Users/aarcher/PycharmProjects/MCP/WholeCell/plots/Perm_SensitivityExternal_'+ namesvars[i+3]  +'.png',
        #             bbox_inches='tight')
        plt.show()


    #check mass balance
    ext_masses_org = y0[(model_local_sens.nvars-3):model_local_sens.nvars]* model_local_sens.external_volume
    cell_masses_org = y0[5:8] * volcell 
    mcp_masses_org = y0[:5] * volmcp
    ext_masses_fin = sol.y[(model_local_sens.nvars-3):model_local_sens.nvars, -1] * model_local_sens.external_volume
    cell_masses_fin = sol.y[5:8,-1] * volcell
    mcp_masses_fin = sol.y[:5, -1] * volmcp
    mass_org = ext_masses_org.sum() + ncells*cell_masses_org.sum() + ncells*nmcps*mcp_masses_org.sum()
    mass_fin = ext_masses_fin.sum() + ncells*cell_masses_fin.sum() + ncells*nmcps*mcp_masses_fin.sum()
    print(mass_org)
    print(mass_fin)


if __name__ == '__main__':
    main()


