import numpy as np
from scipy.integrate import solve_ivp
import sympy as sp
import scipy.sparse as sparse
import pdb
from mpi4py import MPI
import time
import matplotlib.pyplot as plt


def initialize_dim_scaling(**kwargs):
    """
    Computes non-dimensional scalings for each state variable

    :param kwargs:
    kwargs[kcatfDhaB]: k_cat_f for G <-> H
    kwargs[SigmaDhaB]: E_T for DhaB_1 + DhaB_2
    kwargs[KmDhaBG]: Km for G in G <-> H
    kwargs[KiDhaBH]: Ki for H in G <-> H

    kwargs[kcatfDhaT]: k_cat_f for H + N <-> D + P
    kwargs[SigmaDhaT]: E_T for H + N <-> D + P 
    kwargs[KmDhaBH]: Km for H in H + N <-> D + P
    kwargs[KmDhaBN]: Km for N in H + N <-> D + P
    kwargs[KiDhaTP]: Ki for P in H + N <-> D + P
    kwargs[KiDhaTD]: Ki for D in H + N <-> D + P


    :return dimscalings:
    dimscalings['N0'] - list of dimensional scalings for NAD+
    dimscalings['D0'] - list of dimensional scalings for NADH
    dimscalings['iDhaB0'] - list of dimensional scalings for inactive DhaB_1
    dimscalings['G0'] - list of dimensional scalings for Glycerol
    dimscalings['H0'] - list of dimensional scalings for 3-HPA
    dimscalings['P0'] - list of dimensional scalings for 1,3 - PDO
    """

    dimscalings = dict()
    dimscalings['N0'] = kwargs['KmDhaTN']
    dimscalings['D0'] = kwargs['KiDhaTD'] 
    dimscalings['iDhaB0'] = kwargs['SigmaDhaB']
    dimscalings['G0'] = kwargs['KmDhaBG']
    dimscalings['H0'] = kwargs['KmDhaTH']
    dimscalings['P0'] = kwargs['KiDhaTP']

    return dimscalings

def initialize_dimless_param(**kwargs):
    """
    Computes non-dimensional parameters
    :param kwargs:
    kwargs[kcatfDhaB]: k_cat_f for G <-> H
    kwargs[SigmaDhaB]: E_T for DhaB_1 + DhaB_2
    kwargs[KmDhaBG]: Km for G in G <-> H
    kwargs[KiDhaBH]: Ki for H in G <-> H

    kwargs['k-1']: k_r for iDhaB_1 + DhaB_2 <-> aDhaB_1
    kwargs['k1']: k_f for iDhaB_1 + DhaB_2 <-> aDhaB_1
    kwargs['Dha2Exp']: expression level for DhaB_2
    kwargs[iDhaB1Exp]: expression level for iDhaB_1

    kwargs[kcatfDhaT]: k_cat_f for H + N <-> D + P
    kwargs[SigmaDhaT]: E_T for H + N <-> D + P 
    kwargs[KmDhaBH]: Km for H in H + N <-> D + P
    kwargs[KmDhaBN]: Km for N in H + N <-> D + P
    kwargs[KiDhaTP]: Ki for P in H + N <-> D + P
    kwargs[KiDhaTD]: Ki for D in H + N <-> D + P

    kwargs['km']: permeability of the MCP
    kwargs['kc']: permeability of the cell

    kwargs['GInit']: initial conditions of glycerol
    kwargs['NInit']: initial conditions of NAD+
    kwargs['DInit']: initial conditions of NADH
    kwargs['Rm']: radius of the MCP
    kwargs['Diff']: diffusive constant 

    :return param_list: list of non-dimensional parameters
    """

    param_name = ['kcatfDhaB','SigmaDhaB','KmDhaBG',
                  'k-1','k1', 'DhaB2Exp','iDhaB1Exp',
                  'kcatfDhaT', 'SigmaDhaT', 'KiDhaBH','KmDhaTH', 'KmDhaTN', 'KiDhaTP','KiDhaTD', 
                  'km', 'kc', 
                  'GInit', 'NInit','DInit','Rm','Diff']


    for key in kwargs.keys():
        assert key in param_name

    # constants
    RT =  2.479 # constant in kJ/mol
    DeltaGDhaT = -15.1 / RT  # using Ph 7.8 since the DhaT reaction is forward processing
    DeltaGDhaB = -18.0 / RT # using Ph 7.8 since the DhaB reaction is forward processing

    # time scale
    t0 =  kwargs['Rm'] /(3*kwargs['km'] * kwargs['Diff'])


    # non-dimensional parameters

    param_dict = dict()

    param_dict['alpha0'] = t0*kwargs['SigmaDhaB']*kwargs['kcatfDhaB']/kwargs['KmDhaBG']
    param_dict['alpha1'] = t0*kwargs['SigmaDhaB']*kwargs['kcatfDhaB']/kwargs['KmDhaTH']
    param_dict['alpha2'] = t0*kwargs['SigmaDhaT']*kwargs['kcatfDhaT']/kwargs['KmDhaTH']
    param_dict['alpha3'] = t0*kwargs['SigmaDhaT']*kwargs['kcatfDhaT']/kwargs['KmDhaTN']
    param_dict['alpha4'] = t0*kwargs['SigmaDhaT']*kwargs['kcatfDhaT']/kwargs['KiDhaTP']
    param_dict['alpha5'] = t0*kwargs['SigmaDhaT']*kwargs['kcatfDhaT']/kwargs['KiDhaTD'] 

    param_dict['alpha10'] = t0*kwargs['k1']
    param_dict['alpha11'] = t0*kwargs['k-1']

    param_dict['beta0'] = kwargs['KmDhaTH']/kwargs['KiDhaBH']

    param_dict['gamma0'] = (kwargs['KmDhaTH']/kwargs['KmDhaBG'])*np.exp(DeltaGDhaB)
    param_dict['gamma1'] = (kwargs['KiDhaTP']*kwargs['KiDhaTD']/(kwargs['KmDhaTH']*kwargs['KmDhaTN']))*np.exp(DeltaGDhaT) 

    param_dict['km'] = kwargs['km']
    param_dict['kc'] = kwargs['kc']
    param_dict['t0'] = t0

    param_dict['DhaBT'] = kwargs['iDhaB1Exp']/( kwargs['DhaB2Exp'] +  kwargs['iDhaB1Exp'])
    param_dict['DeltaDhaB'] = (1 - kwargs['DhaB2Exp']/kwargs['iDhaB1Exp'])/(1 + kwargs['DhaB2Exp']/kwargs['iDhaB1Exp'])

    return param_dict

def initialize_integration_params(Vratio = 0.01, Rm = 1.e-5,Rc = 5.e-5,Diff = 1.e-4, N =10,
                                  ngrid = 25):
    """

    Initializes parameters to be used numerial scheme

    :param Vratio: Ratio of cell volume to external volume
    :param Rm: Radius of compartment (cm)
    :param Diff: Diffusion coefficient
    :param Rc: Effective Radius of cell (cm)
    :param N: number of cells
    :param ngrid: number of spatial grid points


    :return integration_params: dictionary of integration constants
    """


    # Integration Parameters
    integration_params = dict()
    integration_params['Vratio'] = Vratio
    integration_params['Rm'] = ((N)**(1./3.))*Rm
    integration_params['Diff'] = Diff
    integration_params['Rc'] = ((N)**(1./3.))*Rc
    integration_params['ngrid'] = ngrid
    integration_params['m_m'] = N*(Rm**3)/3
    integration_params['m_c'] = N*(Rc**3)/3
    integration_params['nVars'] = 3*(2+int(ngrid)) + 3
    integration_params['N'] = N


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
    param_dict = initialize_dimless_param(**param_vals)

    # coefficients for the diffusion equation

    pdecoef = param_dict['t0']*integration_params['Diff']*((3.**4.)**(1/3))/((integration_params['m_m']**2.)**(1./3.))
    bc1coef = ((3.**2.)**(1/3))/(integration_params['m_m']**(1./3.))
    bc2coef = (((3.*integration_params['m_c'])**2)**(1./3.))/integration_params['m_m']
    ode2coef = param_dict['t0']*integration_params['Diff']*integration_params['Vratio']*3.*param_dict['kc']/integration_params['Rc']

    # rescaling
    M_cell = (integration_params['Rc'] / integration_params['Rm']) ** 3
    M_mcp = 1.
    DeltaM = np.divide((M_cell - M_mcp), (integration_params['ngrid']-1))

    assert len(x) == n_compounds_cell* (2 + (integration_params['ngrid'])) + 3
    d = np.zeros((len(x))).tolist()  # convert to list to allow use of symbolic derivatives

    ###################################################################################
    ################################## MCP reactions ##################################
    ###################################################################################

    R_DhaB = (param_dict['DhaBT'] - x[2])*(x[3] - param_dict['gamma0'] * x[4]) / (1 + x[3] + x[4] * param_dict['beta0'])
    R_DhaT = (x[4] * x[0] - param_dict['gamma1'] * x[5] * x[1]) / (1 + x[4] * x[0] + x[5] * x[1])

    d[0] = -param_dict['alpha3'] * R_DhaT  # microcompartment equation for N
    d[1] = param_dict['alpha5'] * R_DhaT  # microcompartment equation for D
    d[2] = -param_dict['alpha10'] * x[2]*(x[2] - param_dict['DeltaDhaB']) + param_dict['alpha11']*(param_dict['DhaBT'] - x[2])/ (1 + x[3] + x[4] * param_dict['beta0']) # iDhaB_1

    d[3] = -param_dict['alpha0'] * R_DhaB + x[3 + n_compounds_cell] - x[3]  # microcompartment equation for G
    d[4] = param_dict['alpha1'] * R_DhaB - param_dict['alpha2'] * R_DhaT + x[4 + n_compounds_cell] - x[4]  # microcompartment equation for H
    d[5] = param_dict['alpha4'] * R_DhaT + x[5 + n_compounds_cell] - x[5]  # microcompartment equation for P

    ####################################################################################
    ##################################### boundary of MCP ##############################
    ####################################################################################

    M = M_mcp

    first_coef = pdecoef * (((M + 0.5 * DeltaM) ** 4.) ** (1. / 3.)) / (DeltaM ** 2)
    second_coef = pdecoef * param_dict['km'] * (((M - 0.5 * DeltaM) ** 4.) ** (1. / 3.)) / (bc1coef * DeltaM)

    index = 6
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

    first_coef = pdecoef * param_dict['kc'] * (((M + 0.5 * DeltaM) ** 4) ** (1 / 3.))/ (bc2coef * DeltaM)
    second_coef = pdecoef * (((M - 0.5 * DeltaM) ** 4) ** (1 / 3.))/(DeltaM**2)

    for i in reversed(range(-1-n_compounds_cell, -1-2*n_compounds_cell, -1)):
        # BC at ext volume for the ith compound in the cell
        d[i] = first_coef * (x[i + n_compounds_cell] - x[i]) - second_coef * (x[i] - x[i - n_compounds_cell])

    #####################################################################################
    ######################### external volume equations #################################
    #####################################################################################
    for i in reversed(range(-1, -1-n_compounds_cell, -1)):
        d[i] = ode2coef * (x[i - n_compounds_cell] - x[i])  # external equation for concentration

    return d




if __name__ == '__main__':
    ngrid= 10
    fintime = 10**1
    integration_params = initialize_integration_params(ngrid=ngrid)

    params = {'KmDhaTH': 1.,
          'KmDhaTN': 1.,
          'KiDhaTD': 1.,
          'KiDhaTP': 1.,
          'kcatfDhaT': 1.,
          'kcatfDhaB': 1.,
          'KmDhaBG': 1.,
          'KiDhaBH': 1.,
          'km': 10.**-1,
          'kc': 10.**-1,
          'k1': 1.,
          'k-1': 1.,
          'DhaB2Exp': 1.,
          'iDhaB1Exp': 1.,
          'SigmaDhaB': 1,
          'SigmaDhaT': 1,
          'GInit': 1.,
          'NInit': 1.,
          'DInit': 1.}

    t0 = 3*integration_params['Rm']/params['km']
    print(t0)
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
    dimscalings = initialize_dim_scaling(**params)
    n_compounds_cell = 3
    y0 = np.zeros((integration_params['ngrid'] + 2) * n_compounds_cell + 3)
    y0[-3] = params['GInit']/ dimscalings['G0']  # y0[-5] gives the initial state of the external substrate.
    y0[0] = params['NInit'] / dimscalings['N0']  # y0[5] gives the initial state of the external substrate.
    y0[1] = params['DInit'] / dimscalings['D0']  # y0[6] gives the initial state of the external substrate.
    y0[2] = params['iDhaB1Exp']/ (params['iDhaB1Exp'] + params['DhaB2Exp'])
    #y0[-5] = 100

    # time samples
    mintime = 1
    tol = 1e-12
    nsamples = 500
    timeorig = np.logspace(np.log10(mintime), np.log10(fintime), nsamples)


    time_1 = time.time()
    sol = solve_ivp(SDerivParameterized,[0, fintime+1], y0, method="BDF",jac=SDerivGradFunSparse, t_eval=timeorig,
                    atol=1.0e-5,rtol=1.0e-5)

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

    #################################################
    # Plot solution
    #################################################
    scalings = list(dimscalings.values())
    volcell = 4*np.pi*(integration_params['Rc']**3)/3
    volmcp = 4 * np.pi * (integration_params['Rm'] ** 3) / 3
    volratio = integration_params['Vratio']


    # rescale the solutions
    numeachcompound = 2 + integration_params['ngrid']
    ncompounds = 3

    sol.y[:3, :] = (np.multiply(sol.y[:3, :].T, scalings[:3])).T

    for i in range(numeachcompound):
        j = range(3+i*ncompounds, 3+(i+1)*ncompounds)
        sol.y[j,:] = (np.multiply(sol.y[j,:].T,scalings[3:])).T



    # cellular solutions
    minval = np.inf
    maxval = -np.inf
    for i in range(ncompounds):
        logy = np.log10((4*np.pi*integration_params['m_m']*sol.y[range(5+i,nVars-3,ncompounds), :]*DeltaM).sum(axis=0))
        plt.plot(timeorig*t0,logy)
        logminy = int(round(np.min(logy)))
        logmaxy = int(round(np.max(logy)))
        minval = logminy if logminy < minval else minval
        maxval = logmaxy if logmaxy > maxval else maxval

    plt.title('Plot of cellular masses')
    plt.legend(['G', 'H', 'P'], loc='upper right')
    plt.xlabel('time')
    plt.ylabel('mass')
    # xvalslogtimeticks = list(range(int(np.log10(fintime*t0))+1))
    # xtexlogtimeticks = [r'$10^{' + str(i) + '}$' for i in range(min(),int(np.log10(fintime*t0)))]
    # plt.xticks(xvalslogtimeticks, xtexlogtimeticks)
    yvalslogtimeticks = list(range(minval,maxval))
    ytexlogtimeticks = [r'$10^{' + str(i) + '}$' for i in range(minval,maxval)]
    plt.yticks(yvalslogtimeticks, ytexlogtimeticks)
    # plt.savefig('/Users/aarcher/PycharmProjects/MCP/WholeCell/plots/CellularDynamics_kc_' +
    #            str(params['kc']).replace('.',',') + '_km_' + str(params['km']).replace('.',',') + '.png')
    plt.show()

    # external solutions
    minval = np.inf
    maxval = -np.inf
    for i in range(3):
        logy = np.log10((volcell/volratio)*sol.y[-i-1,:].T)
        plt.plot(timeorig*t0,logy)
        logminy = int(round(np.min(logy)))
        logmaxy = int(round(np.max(logy)))
        minval = logminy if logminy < minval else minval
        maxval = logmaxy if logmaxy > maxval else maxval
    plt.legend(['P','H','G'],loc='upper right')
    plt.title('Plot of external masses')
    plt.xlabel('log(time)')
    plt.ylabel('log(mass)')
    # plt.xticks(xvalslogtimeticks, xtexlogtimeticks)
    yvalslogtimeticks = list(range(minval,maxval))
    ytexlogtimeticks = [r'$10^{' + str(i) + '}$' for i in range(minval,maxval)]
    plt.yticks(yvalslogtimeticks, ytexlogtimeticks)
    # plt.savefig('/Users/aarcher/PycharmProjects/MCP/WholeCell/plots/ExternalDynamics_kc_' +
    #            str(params['kc']).replace('.', ',') + '_km_' + str(params['km']).replace('.', ',') + '.png')
    plt.show()

    #MCP solutions
    minval = np.inf
    maxval = -np.inf
    for i in range(6):
        logy = np.log10(volmcp*sol.y[i,:].T)
        plt.plot(timeorig*t0,logy)
        logminy = int(round(np.min(logy)))
        logmaxy = int(round(np.max(logy)))
        minval = logminy if logminy < minval else minval
        maxval = logmaxy if logmaxy > maxval else maxval

    plt.legend(['N','D','iDhaB1Exp','G','H','P'],loc='upper right')
    plt.title('Plot of MCP masses')
    plt.xlabel('log(time)')
    plt.ylabel('log(mass)')
    # plt.xticks(xvalslogtimeticks, xtexlogtimeticks)
    yvalslogtimeticks = list(range(minval,maxval))
    ytexlogtimeticks = [r'$10^{' + str(i) + '}$' for i in range(minval,maxval)]
    plt.yticks(yvalslogtimeticks, ytexlogtimeticks)
    # plt.savefig('/Users/aarcher/PycharmProjects/MCP/WholeCell/plots/MCPDynamics_kc_' +
    #            str(params['kc']).replace('.',',') + '_km_' + str(params['km']).replace('.',',') + '.png')
    plt.show()

    #check mass balance
    ext_masses_org = y0[-3:]* (volcell/volratio)
    cell_masses_org = y0[9:12] * (volcell - volmcp)
    mcp_masses_org = y0[:6] * volmcp


    ext_masses_fin = sol.y[-3:, -1] * (volcell/volratio)
    cell_masses_fin = sol.y[9:12,-1] * (volcell - volmcp)
    mcp_masses_fin = sol.y[:6, -1] * volmcp
    print(ext_masses_org.sum() + cell_masses_org.sum() + mcp_masses_org.sum())
    print(ext_masses_fin.sum() + cell_masses_fin.sum() + mcp_masses_fin.sum())
    print((sol.y[-3:, -1]).sum()*(volcell/volratio+volmcp+volcell))

