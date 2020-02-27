import numpy as np
from scipy.integrate import solve_ivp

import matplotlib.pyplot as plt

import sympy as sp
import scipy.sparse as sparse


def initialize_dim_scaling(**kwargs):
    """
    Computes non-dimensional scalings for each state variable
    :param kwargs:
    kwargs[VfDhaB]: V_f for G <-> H
    kwargs[KmDhaBG]: Km for G in G <-> H
    kwargs[KmDhaBH]: Km for H in G <-> H

    kwargs[VfIcdE]: V_f for D + I <-> N + A
    kwargs[KmIcdED]: K_m for D in D + I <-> N + A
    kwargs[KmIcdEI]: K_m for I in D + I <-> N + A
    kwargs[KiIcdEN]: K_m for N in D + I <-> N + A
    kwargs[KiIcdEA]: K_m for A in D + I <-> N + A

    :return dimscalings: list of dimensional scalings
    """
    #scalings
    dimscalings = dict()
    dimscalings['N0'] = kwargs['KmDhaTN']
    dimscalings['D0'] = kwargs['KmIcdED']

    dimscalings['G0'] = kwargs['KmDhaBG']
    dimscalings['H0'] = kwargs['KmDhaTH']
    dimscalings['P0'] = kwargs['KiDhaTP']
    dimscalings['A0'] = kwargs['KiIcdEA']
    dimscalings['I0'] = kwargs['KmIcdEI']

    return dimscalings

def initialize_dimless_param(**kwargs):
    """
    Computes non-dimensional parameters
    :param kwargs:
    kwargs[VfDhaB]: V_f for G <-> H
    kwargs[KmDhaBG]: Km for G in G <-> H
    kwargs[KmDhaBH]: Km for H in G <-> H

    kwargs[VfIcdE]: V_f for D + I <-> N + A
    kwargs[KmIcdED]: K_m for D in D + I <-> N + A
    kwargs[KmIcdEI]: K_m for I in D + I <-> N + A
    kwargs[KiIcdEN]: K_m for N in D + I <-> N + A
    kwargs[KiIcdEA]: K_m for A in D + I <-> N + A

    :return param_list: list of non-dimensional parameters
    """

    # constants
    T = 298 # room temperature in kelvin
    R =  8.314 # gas constant
    DeltaGDhaT = -35.1 / (R * T)  # using Ph 7.8 since the IcdE reaction is forward processing
    DeltaGDhaB = -18.0 / (R * T)  # using Ph 7.8 since the IcdE reaction is forward processing
    DeltaGIcdE = -11.4 / (R * T)  # using Ph 7.8 since the IcdE reaction is forward processing

    # time scale
    t0 = 3 * kwargs['Rc'] /kwargs['km']


    # non-dimensional parameters

    param_dict = dict()

    param_dict['alpha0'] = t0*kwargs['VfDhaB']/kwargs['KmDhaBG']
    param_dict['alpha1'] = t0*kwargs['VfDhaB']/kwargs['KmDhaTH']
    param_dict['alpha2'] = t0*kwargs['VfDhaT']/kwargs['KmDhaTH']
    param_dict['alpha3'] = t0*kwargs['VfDhaT']/kwargs['KmDhaTN']
    param_dict['alpha4'] = t0*kwargs['VfIcdE']/kwargs['KmDhaTN']
    param_dict['alpha5'] = t0*kwargs['VfDhaT']/kwargs['KiDhaTP']
    param_dict['alpha6'] = t0*kwargs['VfDhaT']/kwargs['KmIcdED']
    param_dict['alpha7'] = t0*kwargs['VfIcdE']/kwargs['KmIcdED']
    param_dict['alpha8'] = t0*kwargs['VfIcdE']/kwargs['KmIcdEI']
    param_dict['alpha9'] = t0*kwargs['VfIcdE']/kwargs['KiIcdEA']

    param_dict['beta0'] = kwargs['KmDhaTH']/kwargs['KiDhaBH']
    param_dict['beta1'] = kwargs['KmIcdED']/kwargs['KiDhaTD']
    param_dict['beta2'] = kwargs['KmDhaTN']/kwargs['KiIcdEN']

    param_dict['gamma0'] = (kwargs['KmDhaTH']/kwargs['KmDhaBG'])*np.exp(DeltaGDhaB)
    param_dict['gamma1'] = (kwargs['KiDhaTP']*kwargs['KmIcdED']/(kwargs['KmDhaTH']*kwargs['KmDhaTN']))*np.exp(DeltaGDhaT)
    param_dict['gamma2'] = (kwargs['KiIcdEA']*kwargs['KmDhaTN']/(kwargs['KmIcdED']*kwargs['KmIcdEI']))*np.exp(DeltaGIcdE)

    param_dict['km'] = kwargs['km']
    param_dict['kc'] = kwargs['kc']
    param_dict['t0'] = t0

    return param_dict

def initialize_integration_params(Vratio = 0.01, Rc = 1.e-5,Diff = 1.e-4,Rb = 5.e-5,
                                  ngrid = 25):
    """

    Initializes parameters to be used numerial scheme

    :param Vratio: Ratio of cell volume to external volume
    :param Rc: Radius of compartment (cm)
    :param Diff: Diffusion coefficient
    :param Rb: Effective Radius of cell (cm)
    :param ngrid: number of spatial grid points
    :return integration_params: dictionary of integration constants
    """


    # Integration Parameters
    integration_params = dict()
    integration_params['Vratio'] = Vratio
    integration_params['Rc'] = Rc
    integration_params['Diff'] = Diff
    integration_params['Rb'] = Rb
    integration_params['m_m'] = (Rc**3)/3
    integration_params['m_c'] = (Rb**3)/3
    integration_params['ngrid'] = int(ngrid)

    return integration_params


def SDeriv(t, x,integration_params,**kwargs):
    """
    Computes the spatial derivative of the system at time point, t
    :param t: time
    :param x: state variables
    :param diffeq_params: differential equation parameters
    :param integration_params: integration parameters
    :return: d: a list of values of the spatial derivative at time point, t
    """


    ###################################################################################
    ################################# Initialization ##################################
    ###################################################################################

    # Integration Parameters
    n_compounds_cell = 5

    # Differential Equations parameters
    param_vals = kwargs
    param_vals['Rc'] = integration_params['Rc']
    param_dict = initialize_dimless_param(**param_vals)

    # coefficients for the diffusion equation

    pdecoef = param_dict['t0']*integration_params['Diff']*((3.**4.)**(1/3))/((integration_params['m_m']**2.)**(1./3.))
    bc1coef = integration_params['Diff']*((3.**2.)**(1/3))/(integration_params['m_m']**(1./3.))
    bc2coef = integration_params['Diff']*(((3.*integration_params['m_c'])**2)**(1./3.))/integration_params['m_m']
    ode2coef = param_dict['t0']*integration_params['Vratio']*(3.*param_dict['kc']/integration_params['Rb'])


    # rescaling
    M_cell = (integration_params['Rb'] / integration_params['Rc']) ** 3  # why?
    M_mcp = 1.  # why?
    DeltaM = np.divide((M_cell - M_mcp), (integration_params['ngrid']-1))

    assert len(x) == 5 * (2 + (integration_params['ngrid'])) + 2
    d = np.zeros((len(x))).tolist()  # convert to list to allow use of symbolic derivatives

    ###################################################################################
    ################################## MCP reactions ##################################
    ###################################################################################

    R_DhaB = (x[2] - param_dict['gamma0'] * x[3]) / (1 + x[2] + x[3] * param_dict['beta0'])
    R_DhaT = (x[3] * x[0] - param_dict['gamma1'] * x[4] * x[1]) / (1 + x[3] * x[0] + param_dict['beta1'] * x[4] * x[1])
    R_IcdE = (x[1] * x[6] - param_dict['gamma2'] * x[5] * x[0]) / (1 + x[6] * x[1] + param_dict['beta2'] * x[5] * x[0])

    d[0] = -param_dict['alpha3'] * R_DhaT + param_dict['alpha4'] * R_IcdE  # microcompartment equation for N
    d[1] = param_dict['alpha6'] * R_DhaT - param_dict['alpha7'] * R_IcdE  # microcompartment equation for D

    d[2] = -param_dict['alpha0'] * R_DhaB + x[2 + n_compounds_cell] - x[2]  # microcompartment equation for G
    d[3] = param_dict['alpha1'] * R_DhaB - param_dict['alpha2'] * R_DhaT + x[3 + n_compounds_cell] - x[3]  # microcompartment equation for H
    d[4] = param_dict['alpha5'] * R_DhaT + x[4 + n_compounds_cell] - x[4]  # microcompartment equation for P
    d[5] = param_dict['alpha8'] * R_IcdE + x[5 + n_compounds_cell] - x[5]  # microcompartment equation for A
    d[6] = - param_dict['alpha9'] * R_IcdE + x[6 + n_compounds_cell] - x[6]  # microcompartment equation for I



    ####################################################################################
    ##################################### boundary of MCP ##############################
    ####################################################################################

    M = M_mcp

    first_coef = pdecoef * (((M + 0.5 * DeltaM) ** 4.) ** (1. / 3.)) / (DeltaM ** 2)
    second_coef = pdecoef * param_dict['km'] * (((M - 0.5 * DeltaM) ** 4.) ** (1. / 3.)) / (bc1coef * DeltaM)

    for i in range(7, 12):
        # BC at MCP for the ith compound in the cell
        d[i] = first_coef * (x[i + n_compounds_cell] - x[i]) - second_coef * (x[i] - x[i - n_compounds_cell])


    ####################################################################################
    ##################################### interior of cell #############################
    ####################################################################################

    for k in range(2, (integration_params['ngrid'] + 1)):
        start_ind = 7 + (k - 1) * n_compounds_cell
        end_ind = 7 + k * n_compounds_cell
        M += DeltaM  # updated M
        # cell equations for ith compound in the cell
        d[start_ind:end_ind] = pdecoef * ((((M + 0.5 * DeltaM) ** 4.) ** (1. / 3.)) * (x[(start_ind + n_compounds_cell):(end_ind+ n_compounds_cell)] - x[start_ind:end_ind])
                                          - (((M - 0.5 * DeltaM) ** 4.) ** (1. / 3.)) * (x[start_ind:end_ind] -x[(start_ind - n_compounds_cell):(end_ind- n_compounds_cell)]))


    ####################################################################################
    ###################### boundary of cell with external volume #######################
    ####################################################################################

    M = M_cell

    first_coef = pdecoef * param_dict['kc'] * (((M + 0.5 * DeltaM) ** 4) ** (1 / 3.))/ (bc2coef * DeltaM)
    second_coef = pdecoef * (((M - 0.5 * DeltaM) ** 4) ** (1 / 3.))/(DeltaM**2)

    for i in reversed(range(-6, -11, -1)):
        # BC at ext volume for the ith compound in the cell
        d[i] = first_coef * (x[i - n_compounds_cell] - x[i]) - second_coef * (x[i] - x[i + n_compounds_cell])

    #####################################################################################
    ######################### external volume equations #################################
    #####################################################################################

    for i in reversed(range(-1, -6, -1)):
        d[i] = ode2coef * (x[i - n_compounds_cell] - x[i])  # external equation for concentration



    return d

def main():
    """
    plots of internal and external concentration

    """
    #################################################
    # Define spatial derivative and jacobian
    #################################################

    # get parameters
    integration_params = initialize_integration_params(ngrid=25)
    params = {'KmDhaTH': 1.,
              'KmDhaTN': 1.,
              'KiDhaTD': 1.,
              'KiDhaTP': 1.,
              'VfDhaT' : 1.,
              'VfDhaB' : 1.,
              'KmDhaBG' : 1.,
              'KiDhaBH' : 1.,
              'VfIcdE' : 1.,
              'KmIcdED' : 1.,
              'KmIcdEI' : 1.,
              'KiIcdEN' : 1.,
              'KiIcdEA' : 1.,
              'km' : 0.01,
              'kc': 1.0}

    initalvals = {'GInit':200,
                  'IInit':100,
                  'NInit':50,
                  'DInit':50}


    # spatial derivative
    SDerivParameterized = lambda t,x: SDeriv(t,x,integration_params,**params)
    x_list_sp = np.array(sp.symbols('x:' + str( 5*(2+(integration_params['ngrid'])) + 2)))

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
    n_compounds_cell = 5
    y0 = np.zeros((integration_params['ngrid'] + 2) * n_compounds_cell + 2)
    y0[-5] = initalvals['GInit']/ dimscalings['G0']  # y0[-5] gives the initial state of the external substrate.
    y0[-1] = initalvals['IInit'] / dimscalings['I0']  # y0[-1] gives the initial state of the external substrate.
    y0[0] = initalvals['NInit'] / dimscalings['N0']  # y0[5] gives the initial state of the external substrate.
    y0[1] = initalvals['DInit'] / dimscalings['D0']  # y0[6] gives the initial state of the external substrate.


    # time samples
    fintime = 1.e12
    tol = 1e-15
    nsamples = int(1e5)
    timeorig = np.linspace(0,fintime,nsamples)

    #terminal event
    event = lambda t,y: np.absolute(SDerivParameterized(t,y)[-1]) - tol
    event.terminal = True


    sol = solve_ivp(SDerivParameterized,[0, fintime], y0, method="BDF",jac=SDerivGradFunSparse, events = event,
                    t_eval=timeorig, atol=1.0e-10,rtol=1.0e-10)

    print(sol.message)

    #################################################
    # Plot solution
    #################################################

    # external solution
    plt.plot(sol.t,sol.y[-5:,:].T)
    plt.legend(['G','H','P','A','I'],loc='upper right')
    plt.title('Plot of external concentrations')
    plt.xlabel('time')
    plt.ylabel('concentration')
    #filename = "VfDhaB_"+str(VfDhaB)+"_KmDhaBG_" + str(KmDhaBG) + "_KiDhaBH_" + str(KiDhaBH) + "_VfIcdE_"  + str(VfIcdE) + "_KmIcdEA_" + str(KmIcdEA) + "_KmIcdEN_" + str(KmIcdEN) + "_KiIcdED_" + str(KiIcdED) + "_KiIcdEI_" + str(KiIcdEI) + "_GInit_" + str(GInit) + "_NInit_" + "_GInit_" + str(GInit) + "_NInit_" + str(NInit) + "_DInit_" + str(DInit) + "_AInit_" + str(AInit)
    #plt.savefig('/Users/aarcher/PycharmProjects/MCP/WholeCell/plots/ScipyCode_ExternalDynamics_ngrid' + str(integration_params['ngrid'])+'.png')
    plt.show()

    #MCP solutions
    plt.plot(sol.t,sol.y[:7,:].T)
    plt.legend(['N','D','G','H','P','A','I'],loc='upper right')
    plt.title('Plot of MCP concentrations')
    plt.xlabel('time')
    plt.ylabel('concentration')
    #filename = "VfDhaB_"+str(VfDhaB)+"_KmDhaBG_" + str(KmDhaBG) + "_KiDhaBH_" + str(KiDhaBH) + "_VfIcdE_"  + str(VfIcdE) + "_KmIcdEA_" + str(KmIcdEA) + "_KmIcdEN_" + str(KmIcdEN) + "_KiIcdED_" + str(KiIcdED) + "_KiIcdEI_" + str(KiIcdEI) + "_GInit_" + str(GInit) + "_NInit_" + "_GInit_" + str(GInit) + "_NInit_" + str(NInit) + "_DInit_" + str(DInit) + "_AInit_" + str(AInit)
    #plt.savefig('/Users/aarcher/PycharmProjects/MCP/WholeCell/plots/ScipyCode_MCPDynamics_ngrid' + str(integration_params['ngrid']) +'.png')
    plt.show()

    plt.plot(sol.t,sol.y[1:7,:].T)
    plt.legend(['D','G','H','P','A','I'],loc='upper right')
    plt.title('Plot of MCP concentrations without N')
    plt.xlabel('time')
    plt.ylabel('concentration')
    #filename = "VfDhaB_"+str(VfDhaB)+"_KmDhaBG_" + str(KmDhaBG) + "_KiDhaBH_" + str(KiDhaBH) + "_VfIcdE_"  + str(VfIcdE) + "_KmIcdEA_" + str(KmIcdEA) + "_KmIcdEN_" + str(KmIcdEN) + "_KiIcdED_" + str(KiIcdED) + "_KiIcdEI_" + str(KiIcdEI) + "_GInit_" + str(GInit) + "_NInit_" + "_GInit_" + str(GInit) + "_NInit_" + str(NInit) + "_DInit_" + str(DInit) + "_AInit_" + str(AInit)
    #plt.savefig('/Users/aarcher/PycharmProjects/MCP/WholeCell/plots/ScipyCode_MCPDynamicsWithoutN_ngrid' + str(integration_params['ngrid']) +'.png')
    plt.show()

    # plot entire grid

    #create grid
    M_cell = (integration_params['Rb'] / integration_params['Rc']) ** 3  # why?
    M_mcp = 1.
    Mgrid = np.linspace(M_mcp, M_cell, integration_params['ngrid'])
    DeltaM = np.divide((M_cell - M_mcp), (integration_params['ngrid']))
    Mgridfull = np.concatenate(([M_mcp- DeltaM],Mgrid, [M_cell+ DeltaM]))

    #iterate over each reactant
    conc_names = ['G','H','P','A','I']

    for j,name in enumerate(conc_names):
        leg_name = []
        for i in range(0,sol.y.shape[1],sol.y.shape[1]//10):
            plt.plot(Mgridfull,sol.y[(j+2)::5,i].T)
            leg_name.append('t = ' + "{:.2e}".format(sol.t[i]))
        plt.legend(leg_name)
        plt.title('Plot of cellular concentrations of ' + name)
        plt.xlabel('m')
        plt.ylabel('concentration')
        plt.show()

    #check mass balance
    volcell = 4*np.pi*(integration_params['Rb']**3)/3
    volmcp = 4 * np.pi * (integration_params['Rc'] ** 3) / 3
    volratio = integration_params['Vratio']

    scalings = list(dimscalings.values())
    ext_masses_org = np.multiply(y0[-5:],scalings[2:]) * (volcell/volratio)
    cell_masses_org = np.multiply(y0[13:18],scalings[2:]) * (volcell - volmcp)
    mcp_masses_org = np.multiply(y0[:7],scalings) * volmcp



    ext_masses_fin = np.multiply(sol.y[-5:, -1],scalings[2:]) * (volcell/volratio)
    cell_masses_fin = np.multiply(sol.y[13:18,-1],scalings[2:]) * (volcell - volmcp)
    mcp_masses_fin = np.multiply(sol.y[:7, -1],scalings) * volmcp

    print(ext_masses_org.sum() + cell_masses_org.sum() + mcp_masses_org.sum())
    print(ext_masses_fin.sum() + cell_masses_fin.sum() + mcp_masses_fin.sum())

if __name__ == '__main__':
    main()
