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
    dimscalings['iDhaB0'] = kwargs['SigmaDhaB']
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

    param_name = ['VfDhaB', 'VfDhaT', 'VfIcdE', 'KmDhaBG', 'KiDhaBH',
                  'KmDhaTH', 'KmDhaTN', 'KiDhaTP','KiDhaTD', 'KmIcdED',
                  'KmIcdEI', 'KiIcdEA', 'KiIcdEN', 'km', 'kc', 'k-1','k1',
                  'DhaB2Exp','iDhaB1Exp','SigmaDhaB', 'SigmaDhaT','SigmaIcdE',
                  'GInit','IInit', 'NInit','DInit','Rm','Diff']


    for key in kwargs.keys():
        assert key in param_name

    # constants
    RT =  2.479 # constant in kJ/mol
    DeltaGDhaT = -15.1 / RT  # using Ph 7.8 since the DhaT reaction is forward processing
    DeltaGDhaB = -18.0 / RT # using Ph 7.8 since the DhaB reaction is forward processing
    DeltaGIcdE = -18.0 / RT  # using Ph 7.8 since the IcdE reaction is forward processing

    # time scale
    t0 =  kwargs['Rm'] /(3*kwargs['km'] * kwargs['Diff'])


    # non-dimensional parameters

    param_dict = dict()

    param_dict['alpha0'] = t0*kwargs['SigmaDhaB']*kwargs['VfDhaB']/kwargs['KmDhaBG']
    param_dict['alpha1'] = t0*kwargs['SigmaDhaB']*kwargs['VfDhaB']/kwargs['KmDhaTH']
    param_dict['alpha2'] = t0*kwargs['SigmaDhaT']*kwargs['VfDhaT']/kwargs['KmDhaTH']
    param_dict['alpha3'] = t0*kwargs['SigmaDhaT']*kwargs['VfDhaT']/kwargs['KmDhaTN']
    param_dict['alpha4'] = t0*kwargs['SigmaIcdE']*kwargs['VfIcdE']/kwargs['KmDhaTN']
    param_dict['alpha5'] = t0*kwargs['SigmaDhaT']*kwargs['VfDhaT']/kwargs['KiDhaTP']
    param_dict['alpha6'] = t0*kwargs['SigmaDhaT']*kwargs['VfDhaT']/kwargs['KmIcdED']
    param_dict['alpha7'] = t0*kwargs['SigmaIcdE']*kwargs['VfIcdE']/kwargs['KmIcdED']
    param_dict['alpha8'] = t0*kwargs['SigmaIcdE']*kwargs['VfIcdE']/kwargs['KmIcdEI']
    param_dict['alpha9'] = t0*kwargs['SigmaIcdE']*kwargs['VfIcdE']/kwargs['KiIcdEA']
    param_dict['alpha10'] = t0*kwargs['k1']
    param_dict['alpha11'] = t0*kwargs['k-1']

    param_dict['beta0'] = kwargs['KmDhaTH']/kwargs['KiDhaBH']
    param_dict['beta1'] = kwargs['KmIcdED']/kwargs['KiDhaTD']
    param_dict['beta2'] = kwargs['KmDhaTN']/kwargs['KiIcdEN']

    param_dict['gamma0'] = (kwargs['KmDhaTH']/kwargs['KmDhaBG'])*np.exp(DeltaGDhaB)
    param_dict['gamma1'] = (kwargs['KiDhaTP']*kwargs['KmIcdED']/(kwargs['KmDhaTH']*kwargs['KmDhaTN']))*np.exp(DeltaGDhaT)
    param_dict['gamma2'] = (kwargs['KiIcdEA']*kwargs['KmDhaTN']/(kwargs['KmIcdED']*kwargs['KmIcdEI']))*np.exp(DeltaGIcdE)

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
    integration_params['nVars'] = 5*(2+int(ngrid)) + 3
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
    n_compounds_cell = 5
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

    assert integration_params['ngrid'] > 2
    assert len(x) == 5 * (2 + (integration_params['ngrid'])) + 3
    d = np.zeros((len(x))).tolist()  # convert to list to allow use of symbolic derivatives

    ###################################################################################
    ################################## MCP reactions ##################################
    ###################################################################################

    R_DhaB = (param_dict['DhaBT'] - x[2])*(x[3] - param_dict['gamma0'] * x[4]) / (1 + x[3] + x[4] * param_dict['beta0'])
    R_DhaT = (x[4] * x[0] - param_dict['gamma1'] * x[5] * x[1]) / (1 + x[4] * x[0] + param_dict['beta1'] * x[5] * x[1])
    R_IcdE = (x[1] * x[7] - param_dict['gamma2'] * x[6] * x[0]) / (1 + x[7] * x[1] + param_dict['beta2'] * x[6] * x[0])

    d[0] = -param_dict['alpha3'] * R_DhaT + param_dict['alpha4'] * R_IcdE  # microcompartment equation for N
    d[1] = param_dict['alpha6'] * R_DhaT - param_dict['alpha7'] * R_IcdE  # microcompartment equation for D
    d[2] = -param_dict['alpha10'] * x[2]*(x[2] - param_dict['DeltaDhaB']) + param_dict['alpha11']*(param_dict['DhaBT'] - x[2])/ (1 + x[3] + x[4] * param_dict['beta0'])

    d[3] = -param_dict['alpha0'] * R_DhaB + x[3 + n_compounds_cell] - x[3]  # microcompartment equation for G
    d[4] = param_dict['alpha1'] * R_DhaB - param_dict['alpha2'] * R_DhaT + x[4 + n_compounds_cell] - x[4]  # microcompartment equation for H
    d[5] = param_dict['alpha5'] * R_DhaT + x[5 + n_compounds_cell] - x[5]  # microcompartment equation for P
    d[6] = param_dict['alpha9'] * R_IcdE + x[6 + n_compounds_cell] - x[6]  # microcompartment equation for A
    d[7] = - param_dict['alpha8'] * R_IcdE + x[7 + n_compounds_cell] - x[7]  # microcompartment equation for I



    ####################################################################################
    ##################################### boundary of MCP ##############################
    ####################################################################################

    M = M_mcp

    first_coef = pdecoef * (((M + 0.5 * DeltaM) ** 4.) ** (1. / 3.)) / (DeltaM ** 2)
    second_coef = pdecoef * param_dict['km'] * (((M - 0.5 * DeltaM) ** 4.) ** (1. / 3.)) / (bc1coef * DeltaM)

    for i in range(8, 13):
        # BC at MCP for the ith compound in the cell
        d[i] = first_coef * (x[i + n_compounds_cell] - x[i]) - second_coef * (x[i] - x[i - n_compounds_cell])

    ####################################################################################
    ##################################### interior of cell #############################
    ####################################################################################

    for k in range(2, (integration_params['ngrid'])):
        start_ind = 8 + (k - 1) * n_compounds_cell
        end_ind = 8 + k * n_compounds_cell
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

    for i in reversed(range(-6, -11, -1)):
        # BC at ext volume for the ith compound in the cell
        d[i] = first_coef * (x[i + n_compounds_cell] - x[i]) - second_coef * (x[i] - x[i - n_compounds_cell])

    #####################################################################################
    ######################### external volume equations #################################
    #####################################################################################

    for i in reversed(range(-1, -6, -1)):
        d[i] = ode2coef * (x[i - n_compounds_cell] - x[i])  # external equation for concentration
    return d








