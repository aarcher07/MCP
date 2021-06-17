'''
Parallelization of the DhaB-DhaT-IcdE Model.
The DhaB-DhaT-IcdE model contains DhaB-DhaT-IcdE reaction
in the MCP; diffusion in the cell; diffusion from the cell 
in the external volume.

Programme written by aarcher07
Editing History:
- 26/10/20
'''
import numpy as np
from scipy.integrate import solve_ivp
import sympy as sp
import scipy.sparse as sparse
import pdb
from mpi4py import MPI
import time
import matplotlib.pyplot as plt
from Whole_Cell_Engineered_System_IcdE.py import *

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()


def SDerivParallelized(*args):
    """
    Computes the spatial derivative of the system at time point, t
    :param args[0]: time
    :param args[1]: state variables
    :param args[2]: integration parameters
    :param args[3]: differential equation parameters
    :return: d: a list of values of the spatial derivative at time point, t
    """

    ###################################################################################
    ################################# Initialization ##################################
    ###################################################################################

    if rank == 0:

        t = args[0]
        x = args[1]
        integration_params = args[2]
        param_vals = args[3]
        notstop = args[4]

        # sanity check
        assert len(x) == 5 * (2 + (integration_params['ngrid'])) + 2

        #divide the grid for each cell
        ngridperprocessor = integration_params['ngrid'] // size
        ngridperprocessorextra = integration_params['ngrid'] % size

        # get differential equation parameters
        param_vals = param_vals.copy()
        param_vals['Rm'] = integration_params['Rm']
        param_vals['Diff'] = integration_params['Diff']


        # create dimensionless variables
        param_dict = initialize_dimless_param(**param_vals)

        # coefficients for the diffusion equation
        pdecoef = param_dict['t0'] * integration_params['Diff'] * ((3. ** 4.) ** (1 / 3)) / (
                    (integration_params['m_m'] ** 2.) ** (1. / 3.))
        bc1coef = ((3. ** 2.) ** (1 / 3)) / (integration_params['m_m'] ** (1. / 3.))
        bc2coef = (((3. * integration_params['m_c']) ** 2) ** (1. / 3.)) / integration_params['m_m']
        ode2coef = param_dict['t0'] * integration_params['Diff'] * integration_params['Vratio'] * 3. * param_dict[
            'kc'] / integration_params['Rc']

        # rescale grid and create grid
        M_cell = (integration_params['Rc'] / integration_params['Rm']) ** 3
        M_mcp = 1.
        DeltaM = np.divide((M_cell - M_mcp), (integration_params['ngrid'] - 1))

    else:

        #args
        t = None
        x = None
        param_dict = None
        notstop = None

        # params
        ngridperprocessor = None
        ngridperprocessorextra = None
        param_dict = None
        pdecoef = None
        bc1coef = None
        bc2coef = None
        ode2coef = None
        M_cell = None
        M_mcp = None
        DeltaM = None

    # broadcast args
    t = comm.bcast(t, root = 0)
    x = comm.bcast(x, root = 0)
    param_dict = comm.bcast(param_dict, root = 0)

    # broadcast hyperparameters
    ngridperprocessor = comm.bcast(ngridperprocessor, root = 0)
    ngridperprocessorextra = comm.bcast(ngridperprocessorextra, root = 0)
    param_dict = comm.bcast(param_dict, root=0)
    pdecoef = comm.bcast(pdecoef, root=0)
    bc1coef = comm.bcast(bc1coef, root=0)
    bc2coef = comm.bcast(bc2coef, root=0)
    ode2coef = comm.bcast(ode2coef, root=0)
    M_cell = comm.bcast(M_cell, root=0)
    M_mcp = comm.bcast(M_mcp, root=0)
    DeltaM = comm.bcast(DeltaM, root=0)
    n_compounds_cell = 5
    notstop = comm.bcast(notstop, root=0)


    if rank == 0:
        if size == 1:
            nsubgrid = ngridperprocessor + ngridperprocessorextra
            ntotalpoints = len(x)
            d = np.zeros(ntotalpoints).tolist()
        else:
            nsubgrid = ngridperprocessor + ngridperprocessorextra
            ntotalpoints = 7 + n_compounds_cell*nsubgrid
            d = np.zeros(ntotalpoints).tolist()

        # MCP Reactions
        R_DhaB = (x[2] - param_dict['gamma0'] * x[3]) / (1 + x[2] + x[3] * param_dict['beta0'])
        R_DhaT = (x[3] * x[0] - param_dict['gamma1'] * x[4] * x[1]) / (1 + x[3] * x[0] + param_dict['beta1'] * x[4] * x[1])
        R_IcdE = (x[1] * x[6] - param_dict['gamma2'] * x[5] * x[0]) / (1 + x[6] * x[1] + param_dict['beta2'] * x[5] * x[0])

        d[0] = -param_dict['alpha3'] * R_DhaT + param_dict['alpha4'] * R_IcdE  # microcompartment equation for N
        d[1] = param_dict['alpha6'] * R_DhaT - param_dict['alpha7'] * R_IcdE  # microcompartment equation for D
        d[2] = -param_dict['alpha0'] * R_DhaB + x[2 + n_compounds_cell] - x[2]  # microcompartment equation for G
        d[3] = param_dict['alpha1'] * R_DhaB - param_dict['alpha2'] * R_DhaT + x[3 + n_compounds_cell] - x[3]  # microcompartment equation for H
        d[4] = param_dict['alpha5'] * R_DhaT + x[4 + n_compounds_cell] - x[4]  # microcompartment equation for P
        d[5] = param_dict['alpha9'] * R_IcdE + x[5 + n_compounds_cell] - x[5]  # microcompartment equation for A
        d[6] = - param_dict['alpha8'] * R_IcdE + x[6 + n_compounds_cell] - x[6]  # microcompartment equation for I


        M = M_mcp
        first_coef = pdecoef * (((M + 0.5 * DeltaM) ** 4.) ** (1. / 3.)) / (DeltaM ** 2)
        second_coef = pdecoef * param_dict['km'] * (((M - 0.5 * DeltaM) ** 4.) ** (1. / 3.)) / (bc1coef * DeltaM)

        # BC at MCP for the ith compound in the cell
        d[7:12] = first_coef * (x[ (7 + n_compounds_cell) : (12 + n_compounds_cell)] - x[7:12])\
                  - second_coef * (x[7:12] - x[(7 - n_compounds_cell) : (12 - n_compounds_cell)])

        ####################################################################################
        ##################################### interior of cell #############################
        ####################################################################################

        for k in range(1, nsubgrid):
            local_start_ind = 7 + k * n_compounds_cell
            local_end_ind = 7 + (k + 1)* n_compounds_cell

            global_start_ind = 7 + k * n_compounds_cell
            global_end_ind = 7 + (k + 1)* n_compounds_cell

            M += DeltaM  # update M
            first_coef_interior = (pdecoef / (DeltaM ** 2)) *  (((M + 0.5 * DeltaM) ** 4.) ** (1. / 3.))
            second_coef_interior = (pdecoef / (DeltaM ** 2)) * (((M - 0.5 * DeltaM) ** 4.) ** (1. / 3.))
            # cell equations for ith compound in the cell
            d[local_start_ind:local_end_ind] = first_coef_interior * (x[(global_start_ind + n_compounds_cell):(global_end_ind + n_compounds_cell)] - x[global_start_ind:global_end_ind]) \
                                   -  second_coef_interior * (x[global_start_ind:global_end_ind] - x[(global_start_ind - n_compounds_cell):(global_end_ind - n_compounds_cell)])
        if size == 1:
            M = M_cell
            first_coef = pdecoef * param_dict['kc'] * (((M + 0.5 * DeltaM) ** 4) ** (1 / 3.)) / (bc2coef * DeltaM)
            second_coef = pdecoef * (((M - 0.5 * DeltaM) ** 4) ** (1 / 3.)) / (DeltaM ** 2)
            d[local_start_ind:local_end_ind] = first_coef * (
                        x[(global_start_ind + n_compounds_cell):] - x[global_start_ind:global_end_ind]) \
                                               - second_coef * (x[global_start_ind:global_end_ind] - x[(
                                                                                                                   global_start_ind - n_compounds_cell):(
                                                                                                                   global_end_ind - n_compounds_cell)])

            #####################################################################################
            ######################### external volume equations #################################
            #####################################################################################

            local_start_ind = -5
            global_start_ind = -5

            d[local_start_ind:] = ode2coef * (x[(global_start_ind - n_compounds_cell):-5] - x[
                                                                            local_start_ind:])  # external equation for concentration

    elif rank == (size-1):

        nsubgrid = ngridperprocessor
        ntotalpoints = 5 + n_compounds_cell*nsubgrid
        d = np.zeros(ntotalpoints).tolist()
        offset = ngridperprocessorextra + rank * ngridperprocessor
        M = M_mcp + (offset - 1)*DeltaM

        ####################################################################################
        ##################################### interior of cell #############################
        ####################################################################################

        for k in range(0, nsubgrid-1):

            local_start_ind = k * n_compounds_cell
            local_end_ind = (k + 1)* n_compounds_cell

            global_start_ind = 7 + offset*n_compounds_cell  + k*n_compounds_cell
            global_end_ind = 7 + offset*n_compounds_cell  + (k+1)*n_compounds_cell
            M += DeltaM  # update M

            #equations
            first_coef_interior = (pdecoef / (DeltaM ** 2)) * (((M + 0.5 * DeltaM) ** 4.) ** (1. / 3.))
            second_coef_interior = (pdecoef / (DeltaM ** 2)) * (((M - 0.5 * DeltaM) ** 4.) ** (1. / 3.))
            # cell equations for ith compound in the cell
            d[local_start_ind:local_end_ind] = first_coef_interior * (x[(global_start_ind + n_compounds_cell):(global_end_ind + n_compounds_cell)] - x[global_start_ind:global_end_ind]) -  second_coef_interior * (x[global_start_ind:global_end_ind] - x[(global_start_ind - n_compounds_cell):(global_end_ind - n_compounds_cell)])

        ####################################################################################
        ###################### boundary of cell with external volume #######################
        ####################################################################################

        local_start_ind = -10
        local_end_ind = -5
        global_start_ind = -10
        global_end_ind = -5

        M = M_cell
        first_coef = pdecoef * param_dict['kc'] * (((M + 0.5 * DeltaM) ** 4) ** (1 / 3.)) / (bc2coef * DeltaM)
        second_coef = pdecoef * (((M - 0.5 * DeltaM) ** 4) ** (1 / 3.)) / (DeltaM ** 2)
        d[local_start_ind:local_end_ind] = first_coef * (x[(global_start_ind+ n_compounds_cell):] - x[global_start_ind:global_end_ind]) \
                                           - second_coef * (x[global_start_ind:global_end_ind] - x[(global_start_ind - n_compounds_cell):(global_end_ind - n_compounds_cell)])

        #####################################################################################
        ######################### external volume equations #################################
        #####################################################################################

        local_start_ind = -5
        global_start_ind = -5

        d[local_start_ind:] = ode2coef * (x[(global_start_ind - n_compounds_cell):-5] - x[local_start_ind:])  # external equation for concentration

    else:
        nsubgrid = ngridperprocessor
        ntotalpoints = n_compounds_cell*nsubgrid
        d = np.zeros(ntotalpoints).tolist()

        offset = ngridperprocessorextra + rank * ngridperprocessor
        M = M_mcp + (offset - 1)*DeltaM

        ####################################################################################
        ##################################### interior of cell #############################
        ####################################################################################

        for k in range(0, nsubgrid):

            local_start_ind = k * n_compounds_cell
            local_end_ind = (k + 1)* n_compounds_cell
            global_start_ind = 7 + offset*n_compounds_cell + k*n_compounds_cell
            global_end_ind = 7 + offset*n_compounds_cell  + (k+1)*n_compounds_cell
            M += DeltaM  # update M

            #equations
            first_coef_interior = (pdecoef / (DeltaM ** 2)) * (((M + 0.5 * DeltaM) ** 4.) ** (1. / 3.))
            second_coef_interior = (pdecoef / (DeltaM ** 2)) * (((M - 0.5 * DeltaM) ** 4.) ** (1. / 3.))
            # cell equations for ith compound in the cell
            d[local_start_ind:local_end_ind] = first_coef_interior * (x[(global_start_ind + n_compounds_cell):(global_end_ind + n_compounds_cell)] - x[global_start_ind:global_end_ind]) \
                                               -  second_coef_interior * (x[global_start_ind:global_end_ind] - x[(global_start_ind - n_compounds_cell):(global_end_ind - n_compounds_cell)])


    d = comm.gather(d, root=0)

    if rank == 0:
        dS = np.concatenate(d).tolist()
        return dS
    else:
        return notstop


def plot_specific_parameter_set_parallel(params=None,ngrid=25,fintime=10**6):
	'''
	Plots the dynamics of the model with an internal grid points, ngrid, with in
	the cell. The dynamics produced using params. The model solved until time = fintime. 

    :param params: differential equation parameters
    :param ngrid: grid size
	:param fintime: final integration time
	'''
    # get parameters
    notstop = 1
    integration_params = initialize_integration_params(ngrid=ngrid)
    if params is None:
        params = {'KmDhaTH': 0.77,
                  'KmDhaTN': 0.03,
                  'KiDhaTD': 0.23,
                  'KiDhaTP': 7.4,
                  'VfDhaT': 86.2,
                  'VfDhaB': 10.,
                  'KmDhaBG': 0.01,
                  'KiDhaBH': 5.,
                  'VfIcdE' : 30.,
                  'KmIcdED' : 0.1,
                  'KmIcdEI' : 0.02,
                  'KiIcdEN' : 3.,
                  'KiIcdEA' : 10.,
                  'km': 0.1,
                  'kc': 1.,
                  'k1': 10.,
                  'k-1': 2.,
                  'DhaB2Exp': 1.,
                  'iDhaB1Exp': 2.,
                  'SigmaDhaB': 10**-1,
                  'SigmaDhaT': 10**-1,
                  'SigmaIcdE': 10**-1,
                  'GInit': 10.,
                  'IInit': 10.,
                  'NInit': 20.,
                  'DInit': 20.}
    # spatial derivative



    # spatial derivative
    SDerivParameterized = lambda t, x: SDerivParallelized(t, x, integration_params, params, notstop)
    nVars = integration_params['nVars']
    x_list_sp = np.array(sp.symbols('x:' + str(nVars)))

    # jacobian
    if rank == 0:
        SDerivSymbolic = SDerivParameterized(0,x_list_sp)
        SDerivGrad = sp.Matrix(SDerivSymbolic).jacobian(x_list_sp)
        SDerivGradFun = sp.lambdify(x_list_sp, SDerivGrad, 'numpy')
        SDerivGradFunSparse = lambda t, x: sparse.csr_matrix(SDerivGradFun(*x))

        #################################################
        # Integrate with BDF
        #################################################

        # initial conditions
        dimscalings = initialize_dim_scaling(**params)
        n_compounds_cell = 5
        y0 = np.zeros(nVars)
        y0[-5] = params['GInit'] / dimscalings['G0']  # y0[-5] gives the initial state of the external substrate.
        y0[-1] = params['IInit'] / dimscalings['I0']  # y0[-1] gives the initial state of the external substrate.
        y0[0] = params['NInit'] / dimscalings['N0']  # y0[5] gives the initial state of the external substrate.
        y0[1] = params['DInit'] / dimscalings['D0']  # y0[6] gives the initial state of the external substrate.
        # y0[-5] = 100

        # time samples
        tol = 1e-12

        t0 = time.time()
        sol = solve_ivp(SDerivParameterized, [0, fintime], y0, method="BDF", jac=SDerivGradFunSparse,
                        atol=1.0e-6, rtol=1.0e-6)
        t1 = time.time()
        print('time: ' + str(t1-t0))
        print(sol.message)
        notstop = 0
        SDerivParallelized(0, y0, integration_params, params, notstop)

        #################################################
        # Plot solution
        #################################################
        scalings = list(dimscalings.values())
        volcell = 4 * np.pi * (integration_params['Rc'] ** 3) / 3
        volmcp = 4 * np.pi * (integration_params['Rm'] ** 3) / 3
        volratio = integration_params['Vratio']
        # create grid
        M_cell = (integration_params['Rc'] / integration_params['Rm']) ** 3  # why?
        M_mcp = 1.
        Mgrid = np.linspace(M_mcp, M_cell, integration_params['ngrid'] )
        DeltaM = np.divide((M_cell - M_mcp), (integration_params['ngrid'] - 1))

        # rescale the solutions
        numeachcompound = 2 + integration_params['ngrid']
        ncompounds = 5

        sol.y[:2, :] = (np.multiply(sol.y[:2, :].T, scalings[:2])).T
        for i in range(numeachcompound):
            j = range(2 + i * ncompounds, 2 + (i + 1) * ncompounds)
            sol.y[j, :] = (np.multiply(sol.y[j, :].T, scalings[2:])).T

        # cellular solutions

        # for i in range(ncompounds):
        #     plt.plot(sol.t, (4 * np.pi * integration_params['m_m'] * sol.y[range(7 + i, nVars - 5, ncompounds),
        #                                                              :] * DeltaM).sum(axis=0))
        # plt.title('Plot of cellular masses')
        # plt.legend(['G', 'H', 'P', 'A', 'I'], loc='upper right')
        # plt.xlabel('time')
        # plt.ylabel('mass')
        # plt.savefig('/Users/aarcher/PycharmProjects/MCP/WholeCell/plots/CorrectParallel_CellDynamics_ngrid' + str(integration_params['ngrid']) +'.png')
        # plt.show()
        #
        # # external solutions
        # for i in range(5):
        #     plt.plot(sol.t, (volcell / volratio) * sol.y[-i - 1, :].T)
        # plt.legend(['I', 'A', 'P', 'H', 'G'], loc='upper right')
        # plt.title('Plot of external masses')
        # plt.xlabel('time')
        # plt.ylabel('mass')
        # # filename = "VfDhaB_"+str(VfDhaB)+"_KmDhaBG_" + str(KmDhaBG) + "_KiDhaBH_" + str(KiDhaBH) + "_VfIcdE_"  + str(VfIcdE) + "_KmIcdEA_" + str(KmIcdEA) + "_KmIcdEN_" + str(KmIcdEN) + "_KiIcdED_" + str(KiIcdED) + "_KiIcdEI_" + str(KiIcdEI) + "_GInit_" + str(GInit) + "_NInit_" + "_GInit_" + str(GInit) + "_NInit_" + str(NInit) + "_DInit_" + str(DInit) + "_AInit_" + str(AInit)
        # # plt.savefig('/Users/aarcher/PycharmProjects/MCP/WholeCell/plots/ScipyCode_ExternalDynamics_ngrid' + str(integration_params['ngrid'])+'.png')
        # plt.savefig('/Users/aarcher/PycharmProjects/MCP/WholeCell/plots/CorrectParallel_ExternalDynamics_ngrid' + str(integration_params['ngrid']) +'.png')
        # plt.show()
        #
        # # MCP solutions
        #
        # for i in range(7):
        #     plt.plot(sol.t, volmcp * sol.y[i, :].T)
        # plt.legend(['N', 'D', 'G', 'H', 'P', 'A', 'I'], loc='upper right')
        # plt.title('Plot of MCP masses')
        # plt.xlabel('time')
        # plt.ylabel('mass')
        # # filename = "VfDhaB_"+str(VfDhaB)+"_KmDhaBG_" + str(KmDhaBG) + "_KiDhaBH_" + str(KiDhaBH) + "_VfIcdE_"  + str(VfIcdE) + "_KmIcdEA_" + str(KmIcdEA) + "_KmIcdEN_" + str(KmIcdEN) + "_KiIcdED_" + str(KiIcdED) + "_KiIcdEI_" + str(KiIcdEI) + "_GInit_" + str(GInit) + "_NInit_" + "_GInit_" + str(GInit) + "_NInit_" + str(NInit) + "_DInit_" + str(DInit) + "_AInit_" + str(AInit)
        # plt.savefig('/Users/aarcher/PycharmProjects/MCP/WholeCell/plots/CorrectParallel_MCPDynamics_ngrid' + str(integration_params['ngrid']) +'.png')
        # plt.show()

        # check mass balance
        ext_masses_org = y0[-5:] * (volcell / volratio)
        cell_masses_org = y0[12:17] * (volcell - volmcp)
        mcp_masses_org = y0[:7] * volmcp

        ext_masses_fin = sol.y[-5:, -1] * (volcell / volratio)
        cell_masses_fin = sol.y[12:17, -1] * (volcell - volmcp)
        mcp_masses_fin = sol.y[:7, -1] * volmcp
        print(ext_masses_org.sum() + cell_masses_org.sum() + mcp_masses_org.sum())
        print(ext_masses_fin.sum() + cell_masses_fin.sum() + mcp_masses_fin.sum())
        print((sol.y[-5:, -1]).sum() * (volcell / volratio + volmcp + volcell))

    else:
        while notstop:
            notstop = SDerivParameterized(0, x_list_sp)