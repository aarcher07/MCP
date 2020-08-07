import numpy as np
from scipy.integrate import solve_ivp
import sympy as sp
import scipy.sparse as sparse
import pdb
from mpi4py import MPI
import time
import matplotlib.pyplot as plt
from Whole_Cell_Engineered_System_IcdE.py import *


def plot_specific_parameter_set(params=None,ngrid=25,fintime=10**6):
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
    n_compounds_cell = 5
    y0 = np.zeros((integration_params['ngrid'] + 2) * n_compounds_cell + 3)
    y0[-5] = params['GInit']/ dimscalings['G0']  # y0[-5] gives the initial state of the external substrate.
    y0[-1] = params['IInit'] / dimscalings['I0']  # y0[-1] gives the initial state of the external substrate.
    y0[0] = params['NInit'] / dimscalings['N0']  # y0[5] gives the initial state of the external substrate.
    y0[1] = params['DInit'] / dimscalings['D0']  # y0[6] gives the initial state of the external substrate.
    y0[2] = params['iDhaB1Exp']/ (params['iDhaB1Exp'] + params['DhaB2Exp'])
    #y0[-5] = 100

    # time samples
    mintime = 1
    tol = 1e-12
    nsamples = 500
    timeorig = np.logspace(np.log10(mintime), np.log10(fintime), nsamples)


    t0 = time.time()
    sol = solve_ivp(SDerivParameterized,[0, fintime], y0, method="BDF",jac=SDerivGradFunSparse, t_eval=timeorig,
                    atol=1.0e-5,rtol=1.0e-5)

    t1 = time.time()
    print('time: ' + str(t1 - t0))
    # plot entire grid

    xvalslogtimeticks = list(range(int(np.log10(fintime))+1))
    xtexlogtimeticks = [r'$10^{' + str(i) + '}$' for i in range(int(np.log10(fintime))+1)]


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
    ncompounds = 5

    sol.y[:3, :] = (np.multiply(sol.y[:3, :].T, scalings[:3])).T

    for i in range(numeachcompound):
        j = range(3+i*ncompounds, 3+(i+1)*ncompounds)
        sol.y[j,:] = (np.multiply(sol.y[j,:].T,scalings[3:])).T

    # cellular solutions
    minval = np.inf
    maxval = -np.inf
    for i in range(ncompounds):
        logy = np.log10((4*np.pi*integration_params['m_m']*sol.y[range(8+i,nVars-5,ncompounds), :]*DeltaM).sum(axis=0))
        plt.plot(np.log10(timeorig),logy)
        logminy = int(round(np.min(logy)))
        logmaxy = int(round(np.max(logy)))
        minval = logminy if logminy < minval else minval
        maxval = logmaxy if logmaxy > maxval else maxval

    plt.title('Plot of cellular masses')
    plt.legend(['G', 'H', 'P', 'A', 'I'], loc='upper right')
    plt.xlabel('time')
    plt.ylabel('mass')
    plt.xticks(xvalslogtimeticks, xtexlogtimeticks)
    yvalslogtimeticks = list(range(minval,maxval))
    ytexlogtimeticks = [r'$10^{' + str(i) + '}$' for i in range(minval,maxval)]
    plt.yticks(yvalslogtimeticks, ytexlogtimeticks)
    #plt.savefig('/Users/aarcher/PycharmProjects/MCP/WholeCell/plots/CellularDynamics_kc_' +
    #            str(params['kc']).replace('.',',') + '_km_' + str(params['km']).replace('.',',') + '.png')
    plt.show()

    # external solutions
    minval = np.inf
    maxval = -np.inf
    for i in range(5):
        logy = np.log10((volcell/volratio)*sol.y[-i-1,:].T)
        plt.plot(np.log10(timeorig),logy)
        logminy = int(round(np.min(logy)))
        logmaxy = int(round(np.max(logy)))
        minval = logminy if logminy < minval else minval
        maxval = logmaxy if logmaxy > maxval else maxval
    plt.legend(['I','A','P','H','G'],loc='upper right')
    plt.title('Plot of external masses')
    plt.xlabel('log(time)')
    plt.ylabel('log(mass)')
    plt.xticks(xvalslogtimeticks, xtexlogtimeticks)
    yvalslogtimeticks = list(range(minval,maxval))
    ytexlogtimeticks = [r'$10^{' + str(i) + '}$' for i in range(minval,maxval)]
    plt.yticks(yvalslogtimeticks, ytexlogtimeticks)
    #plt.savefig('/Users/aarcher/PycharmProjects/MCP/WholeCell/plots/ExternalDynamics_kc_' +
    #            str(params['kc']).replace('.', ',') + '_km_' + str(params['km']).replace('.', ',') + '.png')
    plt.show()

    #MCP solutions
    minval = np.inf
    maxval = -np.inf
    for i in range(8):
        logy = np.log10(volmcp*sol.y[i,:].T)
        plt.plot(np.log10(timeorig),logy)
        logminy = int(round(np.min(logy)))
        logmaxy = int(round(np.max(logy)))
        minval = logminy if logminy < minval else minval
        maxval = logmaxy if logmaxy > maxval else maxval

    plt.legend(['N','D','iDhaB1Exp','G','H','P','A','I'],loc='upper right')
    plt.title('Plot of MCP masses')
    plt.xlabel('log(time)')
    plt.ylabel('log(mass)')
    plt.xticks(xvalslogtimeticks, xtexlogtimeticks)
    yvalslogtimeticks = list(range(minval,maxval))
    ytexlogtimeticks = [r'$10^{' + str(i) + '}$' for i in range(minval,maxval)]
    plt.yticks(yvalslogtimeticks, ytexlogtimeticks)
    #plt.savefig('/Users/aarcher/PycharmProjects/MCP/WholeCell/plots/MCPDynamics_kc_' +
    #            str(params['kc']).replace('.',',') + '_km_' + str(params['km']).replace('.',',') + '.png')
    plt.show()

    #check mass balance
    ext_masses_org = y0[-5:]* (volcell/volratio)
    cell_masses_org = y0[12:17] * (volcell - volmcp)
    mcp_masses_org = y0[:7] * volmcp


    ext_masses_fin = sol.y[-5:, -1] * (volcell/volratio)
    cell_masses_fin = sol.y[12:17,-1] * (volcell - volmcp)
    mcp_masses_fin = sol.y[:7, -1] * volmcp
    print(ext_masses_org.sum() + cell_masses_org.sum() + mcp_masses_org.sum())
    print(ext_masses_fin.sum() + cell_masses_fin.sum() + mcp_masses_fin.sum())
    print((sol.y[-5:, -1]).sum()*(volcell/volratio+volmcp+volcell))


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

    plot_specific_parameter_set(params,ngrid=100)


