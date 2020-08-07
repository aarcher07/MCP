import numpy as np
from scipy.integrate import solve_ivp

import matplotlib.pyplot as plt

import sympy as sp
import scipy.sparse as sparse
import unittest as unittest
from scipy.optimize import root
from Whole_Cell_Engineered_System_IcdE import *

def steady_state_formula(x,integration_params,**kwargs):

    param_vals = kwargs
    param_vals['Rm'] = integration_params['Rm']
    param_vals['Diff'] = integration_params['Diff']

    param_dict = initialize_dimless_param(**param_vals)

    R_DhaB = (x[2] - param_dict['gamma0'] * x[3]) / (1 + x[2] + x[3] * param_dict['beta0'])
    R_DhaT = (x[3] * x[0] - param_dict['gamma1'] * x[4] * x[1]) / (1 + x[3] * x[0] + param_dict['beta1'] * x[4] * x[1])
    R_IcdE = (x[1] * x[6] - param_dict['gamma2'] * x[5] * x[0]) / (1 + x[6] * x[1] + param_dict['beta2'] * x[5] * x[0])

    d = np.zeros((len(x))).tolist()
    d[0] = -param_dict['alpha3'] * R_DhaT + param_dict['alpha4'] * R_IcdE  # microcompartment equation for N
    #d[1] = param_dict['alpha6'] * R_DhaT - param_dict['alpha7'] * R_IcdE  # microcompartment equation for D
    d[2] = -param_dict['alpha0'] * R_DhaB  # microcompartment equation for G
    d[3] = param_dict['alpha1'] * R_DhaB - param_dict['alpha2'] * R_DhaT  # microcompartment equation for H
    d[4] = param_dict['alpha5'] * R_DhaT  # microcompartment equation for P
    d[5] = param_dict['alpha9'] * R_IcdE   # microcompartment equation for A
    #d[6] = - param_dict['alpha8'] * R_IcdE  # microcompartment equation for I

    dimscalings = integration_params['dimscalings']
    scalings = np.array(list(dimscalings.values()))
    d[1] = np.dot(x[:2],scalings[:2])- kwargs['NInit'] - kwargs['DInit']


    #check mass balance
    volcell = 4*np.pi*(integration_params['Rc']**3)/3
    volmcp = 4 * np.pi * (integration_params['Rm'] ** 3) / 3
    volratio = integration_params['Vratio']

    d[6] = np.dot(x[2:],scalings[2:])*(volcell/volratio+volmcp+volcell) \
           - (kwargs['GInit'] + kwargs['IInit'])*(volcell/volratio)
    return d



def main(params=None):


    x_list_sp = np.array(sp.symbols('x0:7'))
    if params is None:
        params = {'KmDhaTH': 0.77,
                  'KmDhaTN': 0.03,
                  'KiDhaTD': 0.23,
                  'KiDhaTP': 7.4,
                  'VfDhaT': 86.2,
                  'VfDhaB': 10.,
                  'KmDhaBG': 0.01,
                  'KiDhaBH': 5.,
                  'VfIcdE': 10.,
                  'KmIcdED': 0.1,
                  'KmIcdEI': 0.02,
                  'KiIcdEN': 3.,
                  'KiIcdEA': 10.,
                  'km': 1.,
                  'kc': 1.,
                  'GInit': 10.,
                  'IInit': 10.,
                  'NInit': 20.,
                  'DInit': 20.}

    integration_params = initialize_integration_params(ngrid=25)
    dimscalings = initialize_dim_scaling(**params)
    scalings = np.array(list(dimscalings.values()))
    integration_params['dimscalings'] = dimscalings

    volcell = 4*np.pi*(integration_params['Rc']**3)/3
    volmcp = 4 * np.pi * (integration_params['Rm'] ** 3) / 3
    volratio = integration_params['Vratio']

    # to figure out -- how to get fixed points
    print(np.dot(5,scalings[-1])*(volcell/volratio+volmcp+volcell))
    val = np.dot(10,scalings[-1])*(volcell/volratio+volmcp+volcell) \
          - (params['GInit'] + params['IInit'])*(volcell/volratio)
    miss_conc =-val/(scalings[2]*(volcell/volratio+volmcp+volcell))


    Sv = lambda x: steady_state_formula(x, integration_params, **params)
    jacsym = sp.Matrix(Sv(x_list_sp)).jacobian(x_list_sp)
    jaclam = sp.lambdify(x_list_sp, jacsym, 'numpy')
    jac = lambda x: jaclam(*x)
    x0 = np.array([20,20,miss_conc,0,0,0,10])
    sol = root(Sv, x0, args=(), method='hybr', jac=jac)
    print(sol.x)
    print(sol)
    print(np.linalg.eigvals(jac(sol.x)))


if __name__ == '__main__':
    main()
