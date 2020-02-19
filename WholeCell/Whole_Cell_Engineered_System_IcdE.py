import numpy as np
from scipy.integrate import solve_ivp

import matplotlib.pyplot as plt

import sympy as sp
import scipy.sparse as sparse

    ################################################
    # Known Constants
    ################################################

    KmDhaTH = 0.77 #Km for H in N + H <--> D + P
    KmDhaTN = 0.03 #Km for N in N + H <--> D + P
    KiDhaTD = 0.23 #Ki for D in N + H <--> D + P
    KiDhaTP = 7.4  #Ki for P in N + H <--> D + P

    # KmDhaTH = 1. #Km for H in N + H <--> D + P
    # KmDhaTN = 1. #Km for N in N + H <--> D + P
    # KiDhaTD = 1. #Ki for D in N + H <--> D + P
    # KiDhaTP = 1.  #Ki for P in N + H <--> D + P

    VfDhaT = 86.2  #V_f for N + H <--> D + P

    T = 298 # room temperature in kelvin
    R =  8.314 # gas constant
    DeltaGDhaT = -35.1/(R*T) # using Ph 7.8 since the IcdE reaction is forward processing
    DeltaGDhaB = -18.0/(R*T) # using Ph 7.8 since the IcdE reaction is forward processing
    DeltaGIcdE = -11.4/(R*T) # using Ph 7.8 since the IcdE reaction is forward processing

    # cell and MCP constants
    perm_cell = 0.01
    perm_mcp = 1.0
    Vratio = 10
    Rc = 1.e-5 #Radius of compartment (cm)
    Diff = 1.e-4 #Diffusion coefficient
    Rb = 5.e-5 #Effective Radius of cell (cm)
    D = Diff/(Rc**2)
    r_mcp = 1. # why?
    r_cell = Rb/Rc # why?

    # mcp parameters
    n_compounds_mcp = 7
    n_compounds_cell = 5


    ngrid = int(25)

    ################################################
    # Variable Constants
    ################################################

    # DhaB reaction constants
    VfDhaB = 1.0
    KmDhaBG = 1.0
    KiDhaBH = 5.0

    # IcdE reaction constants
    VfIcdE = 1.0
    KmIcdED = 1.0
    KmIcdEI = 3.0
    KiIcdEN = 0.5
    KiIcdEA = 1.0

    # initial conditions for GInit
    GInit = 100.0

    # initial conditions for NInit
    NInit = 400.0

    # initial condition for DInit
    DInit = 400.0

    # initial condition for AInit
    IInit = 200.0


    #################################################
    # Non-dimensional variables
    #################################################


    G0 = KmDhaBG
    H0 = KmDhaTH
    N0 = KmDhaTN
    P0 = KiDhaTP
    D0 = KmIcdED
    A0 = KiIcdEA
    I0 = KmIcdEI
    t0 = 3*r_mcp/perm_mcp

    y0 = np.zeros((ngrid+2)*n_compounds_cell+2)
    y0[-5] = GInit/G0 #y0[-5] gives the initial state of the external substrate. The /G0 turns the value into a dimensionless quantity
    y0[-1] = IInit/I0 #y0[-1] gives the initial state of the external substrate. The /A0 turns the value into a dimensionless quantity
    y0[0] = NInit/N0 #y0[5] gives the initial state of the external substrate. The /N0 turns the value into a dimensionless quantity
    y0[1] = DInit/D0 #y0[6] gives the initial state of the external substrate. The /D0 turns the value into a dimensionless quantity

    param_list = np.zeros((22))
    #non-dimensional parameters
    param_list[0] = t0*VfDhaB/G0
    param_list[1] = t0*VfDhaB/H0
    param_list[2] = t0*VfDhaT/H0
    param_list[3] = t0*VfDhaT/N0
    param_list[4] = t0*VfIcdE/N0
    param_list[5] = t0*VfDhaT/P0
    param_list[6] = t0*VfDhaT/D0
    param_list[7] = t0*VfIcdE/D0
    param_list[8] = t0*VfIcdE/I0
    param_list[9] = t0*VfIcdE/A0

    param_list[10] = KmDhaTH/KiDhaBH
    param_list[11] = KmIcdED/KiDhaTD
    param_list[12] = KmDhaTN/KiIcdEN


    param_list[13] = (H0/KmDhaBG)*np.exp(DeltaGDhaB)
    param_list[14] = (P0*D0/(KmDhaTH*KmDhaTN))*np.exp(DeltaGDhaT)
    param_list[15] = (A0*N0/(KmIcdED*KmIcdEI))*np.exp(DeltaGIcdE)

    param_list[16] = perm_mcp
    param_list[17] = perm_cell

    param_list[18] = GInit
    param_list[19] = NInit
    param_list[20] = DInit
    param_list[21] = IInit

#################################################
# Spatial Derivative
#################################################

def SDeriv(t, x, params):  # spatial derivative

    alpha_list = params[:10]
    beta_list = params[10:13]
    gamma_list = params[13:16]
    perm_mcp = params[16]
    perm_cell = params[17]

    ### why this scaling ####
    M_cell = (Rb / Rc) ** 3 / 3.  # why?
    M_mcp = 1. / 3.  # why?
    M = M_mcp
    DeltaM = np.divide((M_cell - M_mcp), (ngrid))

    ### why this scaling ####

    assert len(x) == 5 * (2 + (ngrid)) + 2

    d = np.zeros((len(x))).tolist()  # convert to list to allow use of symbolic derivatives

    ###################################################################################
    ################################## MCP reactions ##################################
    ###################################################################################

    R_DhaB = (x[2] - gamma_list[0] * x[3]) / (1 + x[2] + x[3] * beta_list[0])
    R_DhaT = (x[3] * x[0] - gamma_list[1] * x[4] * x[1]) / (1 + x[3] * x[0] + beta_list[1] * x[4] * x[1])
    R_IcdE = (x[1] * x[6] - gamma_list[2] * x[5] * x[0]) / (1 + x[6] * x[1] + beta_list[2] * x[5] * x[0])

    d[0] = -alpha_list[3] * R_DhaT + alpha_list[4] * R_IcdE  # microcompartment equation for N
    d[1] = alpha_list[6] * R_DhaT - alpha_list[7] * R_IcdE  # microcompartment equation for D

    d[2] = -alpha_list[0] * R_DhaB + x[2 + n_compounds_cell] - x[2]  # microcompartment equation for G
    d[3] = alpha_list[1] * R_DhaB - alpha_list[2] * R_DhaT + x[3 + n_compounds_cell] - x[
        3]  # microcompartment equation for H
    d[4] = alpha_list[5] * R_DhaT + x[4 + n_compounds_cell] - x[4]  # microcompartment equation for P
    d[5] = alpha_list[8] * R_IcdE + x[5 + n_compounds_cell] - x[5]  # microcompartment equation for A
    d[6] = - alpha_list[9] * R_IcdE + x[6 + n_compounds_cell] - x[6]  # microcompartment equation for I

    ####################################################################################
    ##################################### boundary of MCP ##############################
    ####################################################################################

    M = M_mcp
    for i in range(7, 12):
        first_coef = t0 * (3 ** (4. / 3.)) * (D / (DeltaM) ** 2) * ((((M + 0.5 * DeltaM) ** 4.) ** (1. / 3.))
                                                                    + (((M - 0.5 * DeltaM) ** 4.) ** (1. / 3.)))
        second_coef = t0 * (perm_mcp * (((M - 0.5 * DeltaM) ** 4.) ** (1. / 3.)) * 3 ** (2. / 3.)) / (
                    DeltaM * M_mcp ** (2 / 3))

        # BC at MCP for the ith compound in the cell
        d[i] = first_coef * (x[i + n_compounds_cell] - x[i]) - second_coef * (x[i] - x[i - n_compounds_cell])

        ####################################################################################
    ##################################### interior of cell #############################
    ####################################################################################

    for k in range(2, (ngrid + 1)):
        start_ind = 7 + (k - 1) * n_compounds_cell
        end_ind = 7 + k * n_compounds_cell
        M += DeltaM  # updated M
        # cell equations for ith compound in the cell
        for i in range(start_ind, end_ind):
            coef = t0 * ((3 ** (4. / 3.)) * D / (DeltaM) ** 2)
            d[i] = coef * ((((M + 0.5 * DeltaM) ** 4.) ** (1. / 3.)) * (x[i + n_compounds_cell] - x[i])
                           - (((M - 0.5 * DeltaM) ** 4.) ** (1. / 3.)) * (x[i] - x[i - n_compounds_cell]))

    ####################################################################################
    ###################### boundary of cell with external volume #######################
    ####################################################################################

    M = M_cell

    first_coef = t0 * (3 ** (4. / 3.)) * (D / (DeltaM) ** 2) * ((((M + 0.5 * DeltaM) ** 4) ** (1 / 3.))
                                                                + (((M - 0.5 * DeltaM) ** 4) ** (1 / 3.)))
    second_coef = t0 * (perm_cell * (((M + 0.5 * DeltaM) ** 4) ** (1 / 3.)) * 3 ** (2. / 3.)) / (
                DeltaM * M_cell ** (2 / 3))

    for i in reversed(range(-6, -11, -1)):
        # BC at ext volume for the ith compound in the cell
        d[i] = first_coef * (x[i - n_compounds_cell] - x[i]) - second_coef * (x[i] - x[i + n_compounds_cell])

    #####################################################################################
    ######################### external volume equations #################################
    #####################################################################################

    for i in reversed(range(-1, -6, -1)):
        d[i] = t0 * (3 * perm_cell / r_cell) * (x[i - n_compounds_cell] - x[i]) / (
            Vratio)  # external equation for concentration

    return d


#################################################
# Define spatial derivative and jacobian
#################################################

#spatial derivative
SDerivParameterized = lambda t,x: SDeriv(t,x,param_list)
x_list_sp = np.array(sp.symbols('x:' + str( 5*(2+(ngrid)) + 2)))

#jacobian
SDerivSymbolic = SDeriv(0,x_list_sp,param_list)
SDerivGrad = sp.Matrix(SDerivSymbolic).jacobian(x_list_sp)
SDerivGradFun = sp.lambdify(x_list_sp, SDerivGrad, 'numpy')
SDerivGradFunSparse = lambda t,x: sparse.csr_matrix(SDerivGradFun(*x))

#################################################
# Integrate with BDF
#################################################

# time samples
fintime = 1e4
tol = 1e-4
nsamples = int(1e3)
timeorig = np.linspace(0,fintime,nsamples)

# terminal event
event = lambda t,y: np.absolute(SDerivParameterized(t,y)[-1]) - tol
event.terminal = True


sol = solve_ivp(SDerivParameterized,[0, fintime], y0, method="BDF",jac=SDerivGradFunSparse, events = event,
                t_eval=timeorig, atol=1.0e-3,rtol=1.0e-3)

print(sol.message)

#################################################
# Plot solution
#################################################

plt.plot(sol.t,sol.y[-5:,:].T)
plt.legend(['G','H','P','A','I'],loc='upper right')
plt.title('Plot of concentrations')
plt.xlabel('time')
plt.ylabel('concentration')
#filename = "VfDhaB_"+str(VfDhaB)+"_KmDhaBG_" + str(KmDhaBG) + "_KiDhaBH_" + str(KiDhaBH) + "_VfIcdE_"  + str(VfIcdE) + "_KmIcdEA_" + str(KmIcdEA) + "_KmIcdEN_" + str(KmIcdEN) + "_KiIcdED_" + str(KiIcdED) + "_KiIcdEI_" + str(KiIcdEI) + "_GInit_" + str(GInit) + "_NInit_" + "_GInit_" + str(GInit) + "_NInit_" + str(NInit) + "_DInit_" + str(DInit) + "_AInit_" + str(AInit)
plt.savefig('/Users/aarcher/PycharmProjects/MCP/WholeCell/plots/ScipyCode_ExternalDynamics_ngrid' + str(ngrid)+'.png')
plt.show()

plt.plot(sol.t,sol.y[:7,:].T)
plt.legend(['N','D','G','H','P','A','I'],loc='upper right')
plt.title('Plot of concentrations')
plt.xlabel('time')
plt.ylabel('concentration')
#filename = "VfDhaB_"+str(VfDhaB)+"_KmDhaBG_" + str(KmDhaBG) + "_KiDhaBH_" + str(KiDhaBH) + "_VfIcdE_"  + str(VfIcdE) + "_KmIcdEA_" + str(KmIcdEA) + "_KmIcdEN_" + str(KmIcdEN) + "_KiIcdED_" + str(KiIcdED) + "_KiIcdEI_" + str(KiIcdEI) + "_GInit_" + str(GInit) + "_NInit_" + "_GInit_" + str(GInit) + "_NInit_" + str(NInit) + "_DInit_" + str(DInit) + "_AInit_" + str(AInit)
plt.savefig('/Users/aarcher/PycharmProjects/MCP/WholeCell/plots/ScipyCode_MCPDynamics_ngrid' + str(ngrid) +'.png')
plt.show()

plt.plot(sol.t,sol.y[1:6,:].T)
plt.legend(['D','G','H','P','A'],loc='upper right')
plt.title('Plot of concentrations')
plt.xlabel('time')
plt.ylabel('concentration')
#filename = "VfDhaB_"+str(VfDhaB)+"_KmDhaBG_" + str(KmDhaBG) + "_KiDhaBH_" + str(KiDhaBH) + "_VfIcdE_"  + str(VfIcdE) + "_KmIcdEA_" + str(KmIcdEA) + "_KmIcdEN_" + str(KmIcdEN) + "_KiIcdED_" + str(KiIcdED) + "_KiIcdEI_" + str(KiIcdEI) + "_GInit_" + str(GInit) + "_NInit_" + "_GInit_" + str(GInit) + "_NInit_" + str(NInit) + "_DInit_" + str(DInit) + "_AInit_" + str(AInit)
plt.savefig('/Users/aarcher/PycharmProjects/MCP/WholeCell/plots/ScipyCode_MCPDynamicsWithoutN_ngrid' + str(ngrid) +'.png')
plt.show()
