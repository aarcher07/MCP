import numpy as np
from scipy.integrate import solve_ivp
import sympy as sp
import scipy.sparse as sparse
import pdb
from mpi4py import MPI
import time
import matplotlib.pyplot as plt

def initialize_integration_params(Vratio = 0.46 * 10**(-12), Rc =  0.68e-6, Rm = 3.95e-7,Diff = 1.e-8, Ncells =9*10**9,
                                  ngrid = 25):
    """

    Initializes parameters to be used numerial scheme

    :param Vratio: Ratio of cell volume to external volume
    :param Rm: Radius of compartment (cm)
    :param Diff: Diffusion coefficient m^2s-1
    :param Rc: Effective Radius of cell (cm)
    :param N: number of cells
    :param ngrid: number of spatial grid points


    :return integration_params: dictionary of integration constants
    """


    # Integration Parameters
    integration_params = dict()
    integration_params['Vratio'] = Vratio
    integration_params['Rm'] = Rm
    integration_params['Diff'] = Diff
    integration_params['Rc'] = Rc
    integration_params['ngrid'] = ngrid
    integration_params['m_m'] = (Rm**3)/3
    integration_params['m_c'] = (Rc**3)/3
    integration_params['nVars'] = 3*(2+int(ngrid)) + 2
    integration_params['Ncells'] = Ncells


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

    # Integration Parameters
    n_compounds_cell = 3
    # differential equation parameters
    param_vals = param_vals.copy()
    param_vals['Rm'] = integration_params['Rm']
    param_vals['Diff'] = integration_params['Diff']

    # rescaling
    M_cell = (integration_params['Rc'] / integration_params['Rm']) ** 3
    M_mcp = 1.
    DeltaM = np.divide((M_cell - M_mcp), (integration_params['ngrid']-1))

    pdecoef = integration_params['Diff']*((3.**4.)**(1/3))/((integration_params['m_m']**2.)**(1./3.))
    bc1coef = integration_params['Diff']*((3.**2.)**(1/3))/(integration_params['m_m']**(1./3.))
    bc2coef = integration_params['Diff']*(((3.*integration_params['m_c'])**2)**(1./3.))/integration_params['m_m']
    ode2coef = integration_params['Vratio']*3.*param_vals['kc']/integration_params['Rc']
    Ncells = integration_params['Ncells'] 

    assert len(x) == n_compounds_cell* (2 + (integration_params['ngrid'])) + 2
    d = np.zeros((len(x))).tolist()  # convert to list to allow use of symbolic derivatives

    ###################################################################################
    ################################## MCP reactions ##################################
    ###################################################################################

    R_DhaB = param_vals['SigmaDhaB']*param_vals['kcatfDhaB']*x[2]/ (param_vals['KmDhaBG'] + x[2])
    R_DhaT = param_vals['SigmaDhaT']*param_vals['kcatfDhaT']*x[3] * x[0]  / (param_vals['KmDhaTH']*param_vals['KmDhaTN'] + x[3] * x[0])
    d[0] = - R_DhaT  # microcompartment equation for N
    d[1] =  R_DhaT  # microcompartment equation for D
    d[2] = -R_DhaB + (3*param_vals['km']/integration_params['Rm'])*(x[2 + n_compounds_cell] - x[2])  # microcompartment equation for G
    d[3] =  R_DhaB -  R_DhaT + (3*param_vals['km']/integration_params['Rm'])*(x[3 + n_compounds_cell] - x[3])  # microcompartment equation for H
    d[4] = R_DhaT + (3*param_vals['km']/integration_params['Rm'])*(x[4 + n_compounds_cell] - x[4])  # microcompartment equation for P

    ####################################################################################
    ##################################### boundary of MCP ##############################
    ####################################################################################

    M = M_mcp
    first_coef = pdecoef * (((M + 0.5 * DeltaM) ** 4.) ** (1. / 3.)) / (DeltaM ** 2)
    second_coef = pdecoef * param_vals['km'] * (((M - 0.5 * DeltaM) ** 4.) ** (1. / 3.)) / (bc1coef * DeltaM)

    index = 5
    for i in range(index, index + n_compounds_cell):
        # BC at MCP for the ith compound in the cell
        d[i] = first_coef * (x[i + n_compounds_cell] - x[i]) - second_coef * (x[i] - x[i - n_compounds_cell])

    ####################################################################################
    ##################################### interior of cell #############################
    ####################################################################################
    for k in range(2, (integration_params['ngrid'])):
        start_ind = index + (k-1)*n_compounds_cell
        end_ind = index + k*n_compounds_cell
        M += DeltaM  # update M
        # cell equations for ith compound in the cell
        d[start_ind:end_ind] = (pdecoef/(DeltaM**2)) * ((((M + 0.5 * DeltaM) ** 4.) ** (1. / 3.)) * (x[(start_ind + n_compounds_cell):(end_ind + n_compounds_cell)] - x[start_ind:end_ind])
                                          - (((M - 0.5 * DeltaM) ** 4.) ** (1. / 3.)) * (x[start_ind:end_ind] -x[(start_ind - n_compounds_cell):(end_ind- n_compounds_cell)]))



    ####################################################################################
    ###################### boundary of cell with external volume #######################
    ####################################################################################

    M = M_cell

    first_coef = pdecoef * param_vals['kc'] * (((M + 0.5 * DeltaM) ** 4) ** (1 / 3.))/ (bc2coef * DeltaM)
    second_coef = pdecoef * (((M - 0.5 * DeltaM) ** 4) ** (1 / 3.))/(DeltaM**2)

    for i in reversed(range(-1-n_compounds_cell, -1-2*n_compounds_cell, -1)):
        # BC at ext volume for the ith compound in the cell
        d[i] = first_coef * (x[i + n_compounds_cell] - x[i]) - second_coef * (x[i] - x[i - n_compounds_cell]) 

    #####################################################################################
    ######################### external volume equations #################################
    #####################################################################################
    for i in reversed(range(-1, -1-n_compounds_cell, -1)):
        d[i] = ode2coef * Ncells * (x[i - n_compounds_cell] - x[i])  # external equation for concentration
    return d




if __name__ == '__main__':
    ngrid= 100
    Ncells = 9 * (10**9)  
    secstohrs = 60*60
    secstomins = 60
    fintime = 1*60*60
    mintime = 10**-10

    integration_params = initialize_integration_params(ngrid=ngrid, Ncells =Ncells)

    params = {'KmDhaTH': 0.77, # mM
          'KmDhaTN': 0.03, # mM
          'kcatfDhaT': 59.4, # seconds
          'kcatfDhaB': 400., # Input
          'KmDhaBG': 0.6, # Input
          'km': 10**-4, 
          'kc': 10.**-4,
          'SigmaDhaB': 1, # Input
          'SigmaDhaT': 1, # Input
          'GInit': 200, #  2 * 10^(-4) mol/cm3 = 200 mM. 
          'NInit': 0.5, # mM
          'DInit': 0.5} # mM

    tolG = 0.05*params['GInit']
    def event_Gmin(t,y):
        return y[-3] - tolG

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
    y0 = np.zeros((integration_params['ngrid'] + 2) * n_compounds_cell + 2)
    y0[-3] = params['GInit']  # y0[-5] gives the initial state of the external substrate.
    y0[0] = params['NInit']  # y0[5] gives the initial state of the external substrate.
    y0[1] = params['DInit']  # y0[6] gives the initial state of the external substrate.

    # time samples
    tol = 1e-12
    nsamples = 500
    timeorig = np.logspace(np.log10(mintime), np.log10(fintime), nsamples)


    time_1 = time.time()
    sol = solve_ivp(SDerivParameterized,[0, fintime+1], y0, method="BDF",jac=SDerivGradFunSparse, t_eval=timeorig,
                    atol=tol,rtol=tol, events=event_Gmin)

    time_2 = time.time()
    print('time: ' + str(time_2 - time_1))
    # plot entire grid


    #create grid
    M_cell = (integration_params['Rc'] / integration_params['Rm']) ** 3  # why?
    M_mcp = 1.
    Mgrid = np.linspace(M_mcp, M_cell, integration_params['ngrid'])
    DeltaM = np.divide((M_cell - M_mcp), (integration_params['ngrid']-1))
    Mgridfull = np.concatenate(([M_mcp- DeltaM],Mgrid, [M_cell+ DeltaM]))

    print(sol.message)
    print(sol.t_events)
    #################################################
    # Plot solution
    #################################################
    volcell = 4*np.pi*(integration_params['Rc']**3)/3
    volmcp = 4 * np.pi * (integration_params['Rm'] ** 3) / 3
    volratio = integration_params['Vratio']


    # rescale the solutions
    numeachcompound = 2 + integration_params['ngrid']
    ncompounds = 3
    timeorighours = timeorig/secstohrs
    timeorigmins = timeorig/secstomins
    # cellular solutions
    minval = np.inf
    maxval = -np.inf
    for i in range(0,ncompounds):
        ycell = sol.y[8+i, :]
        plt.plot(timeorighours,ycell)


    plt.title('Plot of cellular masses')
    plt.legend(['G', 'H', 'P'], loc='upper right')
    #plt.legend(['H','P'],loc='upper right')
    plt.xlabel('time (hr)')
    plt.ylabel('concentration (mM)')
    plt.show()


    meancellularconcH = (4*np.pi*integration_params['m_m']*sol.y[range(6,nVars-3,ncompounds), -1]*DeltaM).sum(axis=0)/(volcell-volmcp)
    print(meancellularconcH)
    # external solutions
    minval = np.inf
    maxval = -np.inf
    for i in reversed(range(0,3)):
        yext = sol.y[-i-1,:].T
        plt.plot(timeorighours,yext)
    plt.legend(['G','H','P'],loc='upper right')
    plt.title('Plot of external masses')
    plt.xlabel('time (hr)')
    plt.ylabel('concentration (mM)')
    plt.show()

    #MCP solutions
    minval = np.inf
    maxval = -np.inf
    for i in range(5):
        ymcp = sol.y[i,:].T
        plt.plot(timeorighours,ymcp)
    plt.legend(['N','D','G','H','P'],loc='upper right')
    plt.title('Plot of MCP masses')
    plt.xlabel('time (hr)')
    plt.ylabel('concentration (mM)')
    plt.show()

    #check mass balance
    ext_masses_org = y0[-3:]* (volcell/volratio)
    cell_masses_org = y0[8:11] * (volcell - volmcp)
    mcp_masses_org = y0[:5] * volmcp


    ext_masses_fin = sol.y[-3:, -1] * (volcell/volratio)
    cell_masses_fin = sol.y[8:11,-1] * (volcell - volmcp)
    mcp_masses_fin = sol.y[:5, -1] * volmcp
    print(ext_masses_org.sum() + Ncells*cell_masses_org.sum() + Ncells*mcp_masses_org.sum())
    print(ext_masses_fin.sum() + Ncells*cell_masses_fin.sum() + Ncells*mcp_masses_fin.sum())
    print((sol.y[-3:, -1]).sum()*(volcell/volratio+Ncells*volmcp+Ncells*volcell))

