'''
The wild-type model contains native reaction pathway
in the MCP; diffusion in the cell; diffusion from the cell 
in the external volume.

This model is currently in use. The DhaB-DhaT model assumes that there 
are M identical MCPs within the cytosol and N identical growing cells within the
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

class WildType:
    def __init__(self,params, ncells_per_metrecubed, dncells_per_metrecubed, external_volume = 9e-6,
                 mcp_volume, mcp_surface_area, cell_volume, cell_surface_area):
        """
        Initializes parameters to be used numerial scheme
        :param params: None or dictionary of parameters with keys, PARAMETER_LIST in constants.py
        :param ncells_per_metrecubed: ncells_per_metrecubed at any given time during experiment
        :param external_volume: external volume amount in metres^3
        :param rc: Radius of cell in metres
        :param lc: length of the cell in metres (needed if assuming cells are rods)
        :param rm: Radius of MCP in metres
        :param cellular geometry: "sphere" or "rod" (cylinders with hemispherical ends)
        """
        # geometric parameters
        self.external_volume = external_volume
        self.mcp_surface_area = mcp_surface_area
        self.mcp_volume = mcp_volume
        self.cell_volume = cell_volume
        self.cell_surface_area = cell_surface_area
        self.vratio = self.cell_surface_area/external_volume 

        # differential equation parameters
        self.nvars = 5*3
        self.ncells = lambda t: ncells_per_metrecubed(t)*external_volume
        self.ncells_prev = self.ncells(0)
        self.dncells = lambda t: dncells_per_metrecubed(t)*external_volume
        self.params = params

        # set integration
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
            if self.params is None:
                print("Parameters have not been set.")
                return
            params = self.params

        ###################################################################################
        ################################# Initialization ##################################
        ###################################################################################
     

        # Integration Parameters
        assert len(x) == self.nvars
        n_compounds_cell = 5
        # differential equation parameters
        ncells = self.ncells(t)
        fac_change_conc = 1 / (1 + self.dncells(t)/ncells)
        nmcps = params['nmcps']
        d = np.zeros((len(x))).tolist()  # convert to list to allow use of symbolic derivatives

        ###################################################################################
        ################################## MCP reactions ##################################
        ###################################################################################

        R_CDE =
        R_PQf =
        R_PQr =
        MCP_geo_fac = self.mcp_surface_area/self.mcp_volume

        d[0] = -R_CDE + MCP_geo_fac*params['PermMCPPropanedial']*(x[0 + n_compounds_cell] - x[0])  # microcompartment equation for G
        d[1] =  R_CDE -  R_PQf + R_PQr + MCP_geo_fac*params['PermMCPPropionaldehyde']*(x[1 + n_compounds_cell] - x[1])  # microcompartment equation for H
        d[2] = R_PQf - R_PQr + MCP_geo_fac*params['PermMCPPropanol']*(x[2 + n_compounds_cell] - x[2])  # microcompartment equation for P
        d[3] = R_PQf - R_PQr + MCP_geo_fac*params['PermMCPPropionyl']*(x[3 + n_compounds_cell] - x[3])  # microcompartment equation for P
        d[4] = MCP_geo_fac*params['PermMCPPropionate']*(x[4 + n_compounds_cell] - x[4])  # microcompartment equation for P

        ####################################################################################
        ##################################### cytosol of cell ##############################
        ####################################################################################

        index = 4

        for i in range(index, index + n_compounds_cell):
            # cell equations for ith compound in the cell
            if i % n_compounds_cell == 0:
                Pm = params['PermMCPPropanedial']
            elif i % n_compounds_cell == 1:
                Pm = params['PermMCPPropionaldehyde']
            elif i % n_compounds_cell == 2:
                Pm = params['PermMCPlPropanol']
            elif i % n_compounds_cell == 3:
                Pm = params['PermMCPPropionyl']
            else:
                Pm = params['PermMCPPropionate']

            if i % n_compounds_cell == 0:
                Pc = params['PermCellPropanedial']
            elif i % n_compounds_cell == 1:
                Pc = params['PermCellPropionaldehyde']
            elif i % n_compounds_cell == 2:
                Pc = params['PermCellPropanol']
            elif i % n_compounds_cell == 3:
                Pc = params['PermCellPropionyl']
            else:
                Pc = params['PermCellPropionate']

            d[i] = -Pc*(self.cell_surface_area/self.cell_volume) * (x[i] - x[i + n_compounds_cell]) - nmcps*Pm*(self.mcp_surface_area/self.cell_volume)*(x[i] - x[i- n_compounds_cell]) 

        #####################################################################################
        ######################### external volume equations #################################
        #####################################################################################
        for i in reversed(range(-1, -1-n_compounds_cell, -1)):
            if i % n_compounds_cell == 0:
                Pc = params['PermCellPropanedial']
            elif i % n_compounds_cell == 1:
                Pc = params['PermCellPropionaldehyde']
            elif i % n_compounds_cell == 2:
                Pc = params['PermCellPropanol']
            elif i % n_compounds_cell == 3:
                Pc = params['PermCellPropionyl']
            else:
                Pc = params['PermCellPropionate']

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
