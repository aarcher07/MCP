import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve
import scipy.constants as constants
import sympy as sp
import scipy.sparse as sparse
import pdb
from mpi4py import MPI
import time
import matplotlib.pyplot as plt
from sklearn.metrics import auc

def initialize_integration_params(external_volume = 9e-6, Rc = 0.68e-6, Lc = None, Rm = 7.e-8, 
                                  Ncells =9*10**9, cellular_geometry = "sphere"):
    """

    Initializes parameters to be used numerial scheme

    :param Vratio: external volume in metres^3
    :param Rc: Radius of cell in metres
    :param Lc: length of the cell in metres (needed if assuming cells are rods)
    :param Rm: Radius of MCP in metres
    :param Ncells: number of cells
    :param cellular geometry: "sphere" or "rod" (cylinders with hemispherical ends)

    :return integration_params: dictionary of integration constants
    """


    # Integration Parameters
    integration_params = dict()
    integration_params['external_volume'] = external_volume 
    integration_params['Rc'] = Rc
    integration_params['Lc'] = Lc
    integration_params['Rm'] = Rm
    integration_params['MCP surface area'] =  4*np.pi*(integration_params['Rm']**2)

    integration_params['Ncells'] = Ncells
    integration_params['cellular_geometry'] = cellular_geometry
    integration_params['nVars'] = 3*3 + 2

    if cellular_geometry == "sphere":
        integration_params["cellular geometry"] = "sphere"
        integration_params['cell volume'] = 4*np.pi*(integration_params['Rc']**3)/3
        integration_params['cell surface area'] = 4*np.pi*(integration_params['Rc']**2)
        integration_params['Vratio'] = integration_params['cell surface area']/external_volume
    elif cellular_geometry == "rod":
        integration_params["cellular geometry"] = "cylinder"
        integration_params['cell volume'] = (4*np.pi/3)*(integration_params['Rc'])**3 + (np.pi)*(integration_params['Lc'] - 2*integration_params['Rc'])*((integration_params['Rc'])**2)
        integration_params['cell surface area'] = 2*np.pi*integration_params['Rc']*integration_params['Lc']
        integration_params['Vratio'] = integration_params['cell surface area']/external_volume 

    return integration_params

def SDeriv(*args):
    """
    Computes the spatial derivative of the system at time point, t
    :param t: time
    :param x: state variables
    :param diffeq_params: differential equation parameters
    :param param_vals: integration parameters
    :return: d: a list of values of the spatial derivative at time point, t
    """

    ###################################################################################
    ################################# Initialization ##################################
    ###################################################################################
    t = args[0]
    x = args[1]
    integration_params = args[2]
    param_vals =  args[3]

    # compute concentration of enzymes if concentrations not available
    if not ('SigmaDhaB' in param_vals.keys()) or not ('SigmaDhaT' in param_vals.keys()):
        if 'enz_ratio' in param_vals.keys():
            enz_ratio = param_vals['enz_ratio']
            if not 'dPacking' in param_vals.keys():
                param_vals['dPacking'] = 0.64
            dPacking = param_vals['dPacking']
            param_vals['SigmaDhaB'],  param_vals['SigmaDhaT'] = ComputeEnzymeConcentrations(enz_ratio, dPacking)
            #TODO: error
 

    # Integration Parameters
    n_compounds_cell = 3
    # differential equation parameters
    param_vals = param_vals.copy()
    param_vals['Rm'] = integration_params['Rm']
    Ncells = integration_params['Ncells'] 
    Nmcps = param_vals['Nmcps'] 
    assert len(x) == n_compounds_cell* 3 + 2
    d = np.zeros((len(x))).tolist()  # convert to list to allow use of symbolic derivatives

    ###################################################################################
    ################################## MCP reactions ##################################
    ###################################################################################

    R_DhaB = param_vals['SigmaDhaB']*param_vals['kcatfDhaB']*x[2]/ (param_vals['KmDhaBG'] + x[2])
    R_DhaT = param_vals['SigmaDhaT']*param_vals['kcatfDhaT']*x[3] * x[0]  / (param_vals['KmDhaTH']*param_vals['KmDhaTN'] + x[3] * x[0])

    d[0] = 0  # microcompartment equation for N
    d[1] =  0  # microcompartment equation for D
    d[2] = -R_DhaB + (3*param_vals['km']/integration_params['Rm'])*(x[2 + n_compounds_cell] - x[2])  # microcompartment equation for G
    d[3] =  R_DhaB -  R_DhaT + (3*param_vals['km']/integration_params['Rm'])*(x[3 + n_compounds_cell] - x[3])  # microcompartment equation for H
    d[4] = R_DhaT + (3*param_vals['km']/integration_params['Rm'])*(x[4 + n_compounds_cell] - x[4])  # microcompartment equation for P

    ####################################################################################
    ##################################### cytosol of cell ##############################
    ####################################################################################

    index = 5

    for i in range(index, index + n_compounds_cell):
        # cell equations for ith compound in the cell
        d[i] = -param_vals['kc']*(integration_params['cell surface area']/integration_params['cell volume']) * (x[i] - x[i + n_compounds_cell]) - Nmcps*param_vals['km']*(integration_params['MCP surface area']/integration_params['cell volume'])*(x[i] - x[i- n_compounds_cell]) 

    #####################################################################################
    ######################### external volume equations #################################
    #####################################################################################
    for i in reversed(range(-1, -1-n_compounds_cell, -1)):
        d[i] = integration_params['Vratio']*param_vals['kc'] * Ncells * (x[i - n_compounds_cell] - x[i])  # external equation for concentration
    return d


def ComputeEnzymeConcentrations(ratio, dPacking):
    '''
    Computes the enzyme concentrations from the relative enzyme expressions and
    the packing efficiency
    '''
    rDhaB = 8/2.
    rDhaT = 5/2.
    AvogadroConstant = constants.Avogadro
    rMCP =140/2.
    Vol = lambda r: 4*np.pi*(r**3)/3;  
    SigmaDhaTExpression = lambda NDhaT: (NDhaT*Vol(rDhaT) + ratio*NDhaT*Vol(rDhaB))/Vol(rMCP) - dPacking
    SigmaDhaT = fsolve(SigmaDhaTExpression,1)[0]/(AvogadroConstant*Vol(rMCP*(1e-9)))
    SigmaDhaB = ratio*SigmaDhaT
    return [SigmaDhaB, SigmaDhaT]

if __name__ == '__main__':
    external_volume =  9e-6
    NcellsPerMCubed = 8e14 # 7e13-8e14 cells per m^3
    Ncells = external_volume*NcellsPerMCubed   
    mintime = 10**(-15)
    secstohrs = 60*60
    fintime = 20*60*60
    integration_params = initialize_integration_params(external_volume = external_volume, 
                                                       Ncells =Ncells,cellular_geometry="rod",
                                                       Rc = 0.375e-6, Lc = 2.47e-6)
    params = {'KmDhaTH': 0.77, # mM
          'KmDhaTN': 0.03, # mM
          'kcatfDhaT': 59.4, # /seconds
          'kcatfDhaB':400, # /seconds Input
          'KmDhaBG': 0.6, # mM Input
          'km': 10**-7, 
          'kc': 10.**-5,
          'dPacking': 0.64,
          'enz_ratio': 1/1.33,
          'Nmcps': 10.,
          'GInit': 200, #  2 * 10^(-4) mol/cm3 = 200 mM. 
          'NInit': 1., # mM
          'DInit': 1.} # mM
    Nmcps = params['Nmcps']

    tolG = 0.5*params['GInit']
    def event_Gmin(t,y):
        return y[-3] - tolG
    def event_Pmax(t,y):
        return y[-1] - tolG
    # spatial derivative
    SDerivParameterized = lambda t,x: SDeriv(t,x,integration_params,params)
    nVars = integration_params['nVars']
    x_list_sp = np.array(sp.symbols('x:' + str(nVars)))

    #jacobian
    SDerivSymbolic = SDerivParameterized(0,x_list_sp)
    SDerivGrad = sp.Matrix(SDerivSymbolic).jacobian(x_list_sp)
    SDerivGradFun = sp.lambdify(x_list_sp, SDerivGrad, 'numpy')
    SDerivGradFunSparse = lambda t,x: sparse.csr_matrix(SDerivGradFun(*x))

    #################################################
    # Integrate with BDF
    #################################################

    # initial conditions
    n_compounds_cell = 3
    y0 = np.zeros(3* n_compounds_cell + 2)
    y0[-3] = params['GInit']  # y0[-5] gives the initial state of the external substrate.
    y0[0] = params['NInit']  # y0[5] gives the initial state of the external substrate.
    y0[1] = params['DInit']  # y0[6] gives the initial state of the external substrate.

    # time samples

    tol = 1e-10
    nsamples = 500
    timeorig = np.logspace(np.log10(mintime), np.log10(fintime), nsamples)


    time_1 = time.time()
    sol = solve_ivp(SDerivParameterized,[0, fintime+1], y0, method="BDF",jac=SDerivGradFunSparse, t_eval=timeorig,
                    atol=tol,rtol=tol, events=[event_Gmin,event_Pmax])

    time_2 = time.time()
    print('time: ' + str(time_2 - time_1))

    #################################################
    # Plot solution
    #################################################
    volcell = integration_params['cell volume']
    volmcp = 4 * np.pi * (integration_params['Rm'] ** 3) / 3
    external_volume = integration_params['external_volume']
    colour = ['b','r','y','c','m']

    # rescale the solutions
    ncompounds = 3
    timeorighours = timeorig/secstohrs


    # cellular solutions
    for i in range(0,ncompounds):
        ycell = sol.y[5+i, :]
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
    print(ext_masses_org.sum() + Ncells*cell_masses_org.sum() + Ncells*Nmcps*mcp_masses_org.sum())
    print(ext_masses_fin.sum() + Ncells*cell_masses_fin.sum() + Ncells*Nmcps*mcp_masses_fin.sum())
    print((sol.y[-3:, -1]).sum()*(external_volume+Ncells*Nmcps*volmcp+Ncells*volcell))

