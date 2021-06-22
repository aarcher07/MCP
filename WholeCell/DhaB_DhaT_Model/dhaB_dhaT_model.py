'''
The DhaB-DhaT model contains DhaB-DhaT reaction pathway
in the MCP; diffusion in the cell; diffusion from the cell 
in the external volume.

This model is currently in use. The DhaB-DhaT model assumes that there 
are M identical MCPs within the cytosol and N identical cells within the 
external volume. From time scale analysis, gradients in the cell are removed.

Programme written by aarcher07
Editing History: See github history
'''


import numpy as np
from scipy.integrate import solve_ivp
import scipy.constants as constants
import sympy as sp
import scipy.sparse as sparse
import time
import matplotlib.pyplot as plt
from constants import HRS_TO_SECS

class DhaBDhaTModel:
    def __init__(self,params,external_volume = 9e-6, rc = 0.68e-6,
                lc = 2.47e-6, rm = 7.e-8, ncells_per_metrecubed = 8e14, 
                cellular_geometry = "rod"):
        """
        Initializes parameters to be used numerial scheme
        :param params: None or dictionary of parameters with keys, PARAMETER_LIST in constants.py
        :param external_volume: external volume amount
        :param rc: Radius of cell in metres
        :param lc: length of the cell in metres (needed if assuming cells are rods)
        :param rm: Radius of MCP in metres
        :param ncells_per_metrecubed: number of cells per m^3
        :param cellular geometry: "sphere" or "rod" (cylinders with hemispherical ends)
        """
        # Integration Parameters
        self.external_volume = external_volume 
        self.rc = rc
        self.lc = lc
        self.rm = rm
        self.mcp_surface_area = 4*np.pi*(self.rm**2)
        self.mcp_volume = (4./3.)*np.pi*(self.rm**3)
        self.ncells = ncells_per_metrecubed*external_volume
        self.cellular_geometry = cellular_geometry
        self.nvars = 3*3

        if cellular_geometry == "sphere":
            self.cellular_geometry = "sphere"
            self.cell_volume = 4*np.pi*(self.rc**3)/3
            self.cell_surface_area = 4*np.pi*(self.rc**2)
        elif cellular_geometry == "rod":
            self.cellular_geometry = "rod"
            self.cell_volume = (4*np.pi/3)*(self.rc)**3 + (np.pi)*(self.lc - 2*self.rc)*((self.rc)**2)
            self.cell_surface_area = 2*np.pi*self.rc*self.lc
        self.vratio = self.cell_surface_area/external_volume 

        # differential equation parameters
        self.params = params
        self._set_symbolic_state_vars()
        if params:
            self._set_symbolic_sderiv_conc_fun()


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

        rdhaB = 8/2.
        rdhaT = 5/2.
        AVOGADRO_CONSTANT = constants.Avogadro
        rmcp =140/2.
        vol = lambda r: 4*np.pi*(r**3)/3
        nDhaT =  vol(rmcp)*params['dPacking']/(vol(rdhaT) + params['enz_ratio']*vol(rdhaB))
        dhaT_conc = nDhaT/(AVOGADRO_CONSTANT*vol(rmcp*(1e-9)))
        dhaB_conc = params['enz_ratio']*dhaT_conc

        ###################################################################################
        ################################# Initialization ##################################
        ###################################################################################
     

        # Integration Parameters
        assert len(x) == self.nvars
        n_compounds_cell = 3
        # differential equation parameters
        ncells = self.ncells 
        nmcps = params['nmcps']
        d = np.zeros((len(x))).tolist()  # convert to list to allow use of symbolic derivatives

        ###################################################################################
        ################################## MCP reactions ##################################
        ###################################################################################

        R_DhaB = dhaB_conc*params['kcatfDhaB']*x[0]/ (params['KmDhaBG'] + x[0])
        R_DhaT = dhaT_conc*params['kcatfDhaT']*x[1] * params['NADH_MCP_INIT']  / (params['KmDhaTH']*params['KmDhaTN'] + x[1] * params['NADH_MCP_INIT'])

        d[0] = -R_DhaB + (3*params['PermMCPPolar']/self.rm)*(x[0 + n_compounds_cell] - x[0])  # microcompartment equation for G
        d[1] =  R_DhaB -  R_DhaT + (3*params['PermMCPNonPolar']/self.rm)*(x[1 + n_compounds_cell] - x[1])  # microcompartment equation for H
        d[2] = R_DhaT + (3*params['PermMCPPolar']/self.rm)*(x[2 + n_compounds_cell] - x[2])  # microcompartment equation for P

        ####################################################################################
        ##################################### cytosol of cell ##############################
        ####################################################################################

        index = 3

        for i in range(index, index + n_compounds_cell):
            # cell equations for ith compound in the cell
            if i % 3 == 1:
                Pm = params['PermMCPNonPolar']
            else:
                Pm = params['PermMCPPolar']

            if i % 3 == 0:
                Pc = params['PermCellGlycerol']
            elif i % 3 == 1:
                Pc = params['PermCell3HPA']
            else:
                Pc = params['PermCellPDO']

            d[i] = -Pc*(self.cell_surface_area/self.cell_volume) * (x[i] - x[i + n_compounds_cell]) - nmcps*Pm*(self.mcp_surface_area/self.cell_volume)*(x[i] - x[i- n_compounds_cell]) 

        #####################################################################################
        ######################### external volume equations #################################
        #####################################################################################
        for i in reversed(range(-1, -1-n_compounds_cell, -1)):
            if i % 3 == 0:
                Pc = params['PermCellGlycerol']
            elif i % 3 == 1:
                Pc = params['PermCell3HPA']
            else:
                Pc = params['PermCellPDO']

            d[i] = self.vratio* Pc * ncells * (x[i - n_compounds_cell] - x[i])  # external equation for concentration
        return d


    def _set_symbolic_state_vars(self):
        """
        Generates the symbol state variables for the model
        """
        self.x_sp = np.array(sp.symbols('x:' + str(self.nvars)))


    def _set_symbolic_sderiv(self):
        """
        Generates the symbol differential equation
        """
        x_sp = getattr(self, 'x_sp', None)
        if x_sp is None:
            self._set_symbolic_state_vars()
        self.sderiv_symbolic = self._sderiv(0,self.x_sp)
        

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
        sderiv_jac_conc_lam = sp.lambdify(self.x_sp, sderiv_jac_conc_sp, 'numpy')
        self.sderiv_jac_conc_fun = lambda t,x: sparse.csr_matrix(sderiv_jac_conc_lam(*x))


def main():
    external_volume =  9e-6
    ncells_per_metrecubed = 8e14 # 7e13-8e14 cells per m^3
    
    params = {'kcatfDhaB': 630, # /seconds Input
          'KmDhaBG': 0.85, # mM Input
          'kcatfDhaT': 70., # /seconds
          'KmDhaTH': 0.55, # mM
          'KmDhaTN': 0.245, # mM
          'PermMCPPolar': 10**-3, 
          'PermMCPNonPolar': 10**-2, 
          'PermCellGlycerol': 10**-7,
          'PermCellPDO':  10**-5,
          'PermCell3HPA': 10**-2,
          'dPacking': 0.64,
          'enz_ratio': 1/18,
          'nmcps': 15.,
          'NADH_MCP_INIT': .36, # mM
          'NAD_MCP_INIT': 1.} # mM

    init_conditions = {'G_EXT_INIT': 200, #  2 * 10^(-4) mol/cm3 = 200 mM. 
                        }

    nmcps = params['nmcps']

    # make model
    dhaB_dhaT_model = DhaBDhaTModel(params, external_volume = external_volume, 
                                    ncells_per_metrecubed =ncells_per_metrecubed,cellular_geometry="rod",
                                    rc = 0.375e-6, lc = 2.47e-6)
    

    # event functions
    tolG = 0.5*init_conditions['G_EXT_INIT']
    tolsolve = 10**-5

    def event_Gmin(t,y):
        return y[-3] - tolG
    def event_Pmax(t,y):
        return y[-1] - tolG
    def event_stop(t,x):
        dSsample = sum(np.abs(dhaB_dhaT_model._sderiv(t,x)))
        return dSsample - tolsolve 
    event_stop.terminal = True

    # integration parameters


    mintime = 10**(-15)
    fintime = 72*60*60

    #################################################
    # Integrate with BDF
    #################################################


    # initial conditions
    n_compounds_cell = 3
    y0 = np.zeros(dhaB_dhaT_model.nvars)
    y0[-3] = init_conditions['G_EXT_INIT']  # y0[-5] gives the initial state of the external substrate.

    # time samples

    tol = 1e-3
    nsamples = 500
    timeorig = np.logspace(np.log10(mintime), np.log10(fintime), nsamples)

    time_1 = time.time()
    try:
        sol = solve_ivp(dhaB_dhaT_model._sderiv,[0, fintime+1], y0, method="BDF",jac=dhaB_dhaT_model.sderiv_jac_conc_fun, t_eval=timeorig,
                        atol=tol,rtol=tol, events=event_stop)
    except ValueError:
        return
    time_2 = time.time()

    print('time to integrate: ' + str(time_2 - time_1))
    print(sol.message)

    #################################################
    # Plot solution
    #################################################
    ncells = dhaB_dhaT_model.ncells
    volcell = dhaB_dhaT_model.cell_volume
    volmcp = dhaB_dhaT_model.mcp_volume
    external_volume = dhaB_dhaT_model.external_volume
    colour = ['b','r','y','c','m']

    # rescale the solutions
    ncompounds = 3
    timeorighours = sol.t/HRS_TO_SECS
    print(sol.message)

    # cellular solutions
    for i in range(0,ncompounds):
        ycell = sol.y[3+i, :]
        plt.plot(timeorighours,ycell, colour[i])
    plt.title('Plot of cellular concentration')
    plt.legend(['Glycerol', '3-HPA', '1,3-PDO'], loc='upper right')
    plt.xlabel('time (hr)')
    plt.ylabel('concentration (mM)')
    plt.show()


    # external solution
    for i in range(0,3):
        yext = sol.y[-3+i,:].T
        plt.plot(timeorighours,yext, colour[i])

    plt.legend(['Glycerol','3-HPA','1,3-PDO'],loc='upper right')
    plt.title('Plot of external concentration')
    plt.xlabel('time (hr)')
    plt.ylabel('concentration (mM)')
    plt.show()

    #MCP solutions
    minval = np.inf
    maxval = -np.inf
    for i in range(3):
        ymcp = sol.y[2+i,:].T
        plt.plot(timeorighours,ymcp, colour[i])


    plt.legend(['Glycerol','3-HPA','1,3-PDO'],loc='upper right')
    plt.title('Plot of MCP concentration')
    plt.xlabel('time (hr)')
    plt.ylabel('concentration (mM)')
    plt.show()

    #check mass balance
    ext_masses_org = y0[-3:]* external_volume
    cell_masses_org = y0[5:8] * volcell 
    mcp_masses_org = y0[:5] * volmcp
    ext_masses_fin = sol.y[-3:, -1] * external_volume
    cell_masses_fin = sol.y[5:8,-1] * volcell
    mcp_masses_fin = sol.y[:5, -1] * volmcp
    print(ext_masses_org.sum() + ncells*cell_masses_org.sum() + ncells*nmcps*mcp_masses_org.sum())
    print(ext_masses_fin.sum() + ncells*cell_masses_fin.sum() + ncells*nmcps*mcp_masses_fin.sum())
    print((sol.y[-3:, -1]).sum()*(external_volume+ncells*nmcps*volmcp+ncells*volcell))

if __name__ == '__main__':
    main()
