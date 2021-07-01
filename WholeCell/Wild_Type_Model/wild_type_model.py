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
        :param ncells_per_metrecubed: number of cells per metrecubed at any given time during experiment
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
        dncells = self.dncells(t)

        fac_change_conc = 1 / (1 + dncells/ncells)

        nmcps = params['nmcps']
        d = np.zeros((len(x))).tolist()  # convert to list to allow use of symbolic derivatives

        ###################################################################################
        ################################## MCP reactions ##################################
        ###################################################################################

        R_CDE = params["VmaxCDEf"]*x[0]/(x[0] + params["KmCDEPropanediol"])
        R_Pf = params["VmaxPf"]*x[1]/(x[1] + params["KmPfPropionaldehyde"])
        R_Pr = params["VmaxPr"]*x[3]/(x[3] + params["KmPrPropionyl"])
        R_Qf = params["VmaxQf"]*x[1]/(x[1] + params["KmPfPropionaldehyde"])
        R_Qr = params["VmaxQr"]*x[2]/(x[2] + params["KmPfPropanol"])
        R_Lf = params["VmaxLf"]*x[8]/(x[8] + params["KmLPropionyl"])

        MCP_geo_fac = self.mcp_surface_area/self.mcp_volume

        d[0] = -R_CDE + MCP_geo_fac*params['PermMCPPropanediol']*(x[0 + n_compounds_cell] - x[0])*fac_change_conc  # microcompartment equation for G
        d[1] =  R_CDE -  R_Pf - R_Qf + R_Pr + R_Qr +MCP_geo_fac*params['PermMCPPropionaldehyde']*(x[1 + n_compounds_cell] - x[1])*fac_change_conc  # microcompartment equation for H
        d[2] = R_Qf - R_Qr + MCP_geo_fac*params['PermMCPPropanol']*(x[2 + n_compounds_cell] - x[2])*fac_change_conc  # microcompartment equation for P
        d[3] = R_Pf - R_Pr + MCP_geo_fac*params['PermMCPPropionyl']*(x[3 + n_compounds_cell] - x[3])*fac_change_conc  # microcompartment equation for P
        d[4] = MCP_geo_fac*params['PermMCPPropionate']*(x[4 + n_compounds_cell] - x[4])*fac_change_conc  # microcompartment equation for P

        ####################################################################################
        ##################################### cytosol of cell ##############################
        ####################################################################################

        index = 4

        for i in range(index, index + n_compounds_cell):
            # cell equations for ith compound in the cell
            if i % n_compounds_cell == 0:
                Pm = params['PermMCPPropanediol']
            elif i % n_compounds_cell == 1:
                Pm = params['PermMCPPropionaldehyde']
            elif i % n_compounds_cell == 2:
                Pm = params['PermMCPlPropanol']
            elif i % n_compounds_cell == 3:
                Pm = params['PermMCPPropionyl']
            else:
                Pm = params['PermMCPPropionate']

            if i % n_compounds_cell == 0:
                Pc = params['PermCellPropanediol']
            elif i % n_compounds_cell == 1:
                Pc = params['PermCellPropionaldehyde']
            elif i % n_compounds_cell == 2:
                Pc = params['PermCellPropanol']
            elif i % n_compounds_cell == 3:
                Pc = params['PermCellPropionyl']
            else:
                Pc = params['PermCellPropionate']

            d[i] = -Pc*(self.cell_surface_area/self.cell_volume) * (fac_change_conc*x[i] - x[i + n_compounds_cell]) - nmcps*Pm*(self.mcp_surface_area/self.cell_volume)*(x[i] - x[i- n_compounds_cell])*fac_change_conc
            if i % n_compounds_cell == 4:
                d[i] + = R_Lf

        #####################################################################################
        ######################### external volume equations #################################
        #####################################################################################
        for i in reversed(range(-1, -1-n_compounds_cell, -1)):
            if i % n_compounds_cell == 0:
                Pc = params['PermCellPropanediol']
            elif i % n_compounds_cell == 1:
                Pc = params['PermCellPropionaldehyde']
            elif i % n_compounds_cell == 2:
                Pc = params['PermCellPropanol']
            elif i % n_compounds_cell == 3:
                Pc = params['PermCellPropionyl']
            else:
                Pc = params['PermCellPropionate']

            d[i] = self.vratio* Pc * ncells * (fac_change_conc*x[i - n_compounds_cell] - x[i])  # external equation for concentration
        return d


    def _reset_t_prev(self):
        """
        Resets t_prev to 0
        """
        self.t_prev = 0

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