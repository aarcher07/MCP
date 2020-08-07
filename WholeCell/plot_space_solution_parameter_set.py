import numpy as np
from scipy.integrate import solve_ivp
import sympy as sp
import scipy.sparse as sparse
import pdb
from mpi4py import MPI
import time
import matplotlib.pyplot as plt
from Whole_Cell_Engineered_System_IcdE.py import *

def plot_space_solution_parameter_set(params=None,ngrid=25, dim_fintime_seconds = 5*60):
    """
    plots of internal and external concentration

    """
    #################################################
    # Define spatial derivative and jacobian
    #################################################

    # get parameters
    integration_params = initialize_integration_params(ngrid=ngrid)
    if params is None:
        params = {'KmDhaTH': 1.,
                  'KmDhaTN': 1.,
                  'KiDhaTD': 1.,
                  'KiDhaTP': 1.,
                  'VfDhaT': 1.,
                  'VfDhaB': 1.,
                  'KmDhaBG': 1.,
                  'KiDhaBH': 1.,
                  'VfIcdE' : 1.,
                  'KmIcdED' : 1.,
                  'KmIcdEI' : 1.,
                  'KiIcdEN' : 1.,
                  'KiIcdEA' : 1.,
                  'km': 1.,
                  'kc': 1.,
                  'k1': 1.,
                  'k-1': 1.,
                  'DhaB2Exp': 1.,
                  'iDhaB1Exp': 1.,
                  'SigmaDhaB': 10**-1,
                  'SigmaDhaT': 10**-1,
                  'SigmaIcdE': 10**-1,
                  'GInit': 1.,
                  'IInit': 1.,
                  'NInit': 1.,
                  'DInit': 1.}

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
    scalings = list(dimscalings.values())
    tol = 1e-12
    nsamples = 10**4


    # time samples
    mintime_seconds = 10**-4
    
    dim_time_seconds = np.logspace(np.log10(mintime_seconds), np.log10(dim_fintime_seconds), nsamples)

    #time sample in mins
    mintime_min = mintime_seconds/(60)
    dim_fintime_min = dim_fintime_seconds/(60)
    dim_time_min = dim_time_seconds/(60)


    # non dimensionalized time
    t0 = 3*integration_params['Rm']/params['km']
    nondim_time = dim_time_seconds/t0
    nondim_fintime = dim_fintime_seconds/t0   

    n_compounds_cell = 5

    y0 = np.zeros((integration_params['ngrid'] + 2) * n_compounds_cell + 3)
    y0[-5] = params['GInit']/ dimscalings['G0']  # y0[-5] gives the initial state of the external substrate.
    y0[-1] = params['IInit'] / dimscalings['I0']  # y0[-1] gives the initial state of the external substrate.
    y0[0] = params['NInit'] / dimscalings['N0']  # y0[5] gives the initial state of the external substrate.
    y0[1] = params['DInit'] / dimscalings['D0']  # y0[6] gives the initial state of the external substrate.
    y0[2] = params['iDhaB1Exp']/ (params['iDhaB1Exp'] + params['DhaB2Exp'])


    t0 = time.time()
    sol = solve_ivp(SDerivParameterized,[0, nondim_fintime + 1], y0, method="BDF",jac=SDerivGradFunSparse, t_eval=nondim_time,
                    atol=1.0e-5,rtol=1.0e-5)

    t1 = time.time()
    print('time: ' + str(t1 - t0))


    #create grid
    M_cell = (integration_params['Rc'] / integration_params['Rm']) ** 3  # why?
    M_mcp = 1.
    Mgrid = np.linspace(M_mcp, M_cell, integration_params['ngrid'])
    DeltaM = np.divide((M_cell - M_mcp), (integration_params['ngrid']-1))
    Mgridfull = np.concatenate(([M_mcp- DeltaM],Mgrid, [M_cell+ DeltaM]))

    print(sol.message)
    numtimes= 10
    time_index = np.linspace(0,nsamples-1,numtimes).astype(int).tolist()
    #time_spacing = ((np.logspace(1,int(np.log10(nsamples)),num_times) - 1).astype(int)).tolist()
    names = ['NADH','NAD+','iDhaB','Glycerol','3-HPA','1,3-PD0','a-KG','Isocitrate']

    numeachcompound = 2 + ngrid
    sol.y[:3, :] = (np.multiply(sol.y[:3, :].T, scalings[:3])).T
    for i in range(numeachcompound):
        j = range(3+i*n_compounds_cell, 3+(i+1)*n_compounds_cell)
        sol.y[j,:] = (np.multiply(sol.y[j,:].T,scalings[3:])).T

    # grid spacing for each compound
    for i in range(n_compounds_cell):
        y = sol.y[range(3+i,nVars,n_compounds_cell), :][:,time_index]
        fig,ax=plt.subplots()
        plt.plot(Mgridfull,y)
        plt.legend([ '%s' % float('%.3g' % dim_time_min[ind]) + " mins" for ind in time_index])
        plt.xlabel('non-dimensionalize and rescaled radius, ' + r'$m^{\ast}$')
        plt.ylabel('mass/g')
        plt.title('Spatial dynamics of ' + names[3+i])  
        plt.draw() # this is required, or the ticklabels may not exist (yet) at the next step
        labels = [w.get_text() for w in ax.get_xticklabels()]
        locs=list(ax.get_xticks())
        labels+=[r'$m^{\ast}_{MCP}$']
        locs+=[Mgridfull[0]]
        labels+=[r'$m^{\ast}_{ext}$']
        locs+=[Mgridfull[-1]]
        ax.set_xticklabels(labels)
        ax.set_xticks(locs)
        ax.grid()
        plt.show()




if __name__ == '__main__':
    params = {'KmDhaTH': 0.77,
              'KmDhaTN': 0.03,
              'KiDhaTD': 0.23,
              'KiDhaTP': 7.4,
              'VfDhaT': 56.4,
              'VfDhaB': 600.,
              'KmDhaBG': 0.8,
              'KiDhaBH': 5.,
              'VfIcdE' : 30.,
              'KmIcdED' : 0.1,
              'KmIcdEI' : 0.02,
              'KiIcdEN' : 3.,
              'KiIcdEA' : 10.,
              'km': 10.**-4,
              'kc': 10.**-4,
              'k1': 10.,
              'k-1': 2.,
              'DhaB2Exp': 100.,
              'iDhaB1Exp': 1.,
              'SigmaDhaB': 10**-1,
              'SigmaDhaT': 10**-1,
              'SigmaIcdE': 10**-1,
              'GInit': 10.,
              'IInit': 10.,
              'NInit': 20.,
              'DInit': 20.}

    plot_space_solution_parameter_set(params=params,ngrid=3, dim_fintime_seconds= 5*10**3)
