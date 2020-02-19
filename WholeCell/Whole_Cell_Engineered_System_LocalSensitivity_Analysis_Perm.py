import numpy as np
from scipy.integrate import solve_ivp
from scipy.constants import *
import sys
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib
import math
import sympy as sp
import scipy.sparse as sparse
import time
from numpy.linalg import LinAlgError
from Whole_Cell_Engineered_System_IcdE import *

#################################################
# parameter symbols
#################################################

# dS parameters
params_sp = []
params_sp.append(sp.symbols('km'))
params_sp.append(sp.symbols('kc'))
# initial conditions symbols
params_sp.append(sp.symbols('GInit'))
params_sp.append(sp.symbols('IInit'))

#################################################
# variable symbols
#################################################

x_list_sp = np.array(sp.symbols('x:' + str( 5*(2+(ngrid)) + 2)))

nVars = len(x_list_sp)
nParams = len(params_sp)
nSensitivityEqs = nVars*nParams

#senstivity variables
sensitivityVars = np.array(list(sp.symbols('s0:' + str(nSensitivityEqs))))

#################################################
# define sensitivity equations as a function
#################################################
param = []

param.extend(param_list[:16])
param.extend(params_sp[:3])
param.extend(param_list[19:21])
param.append(params_sp[-1])

SDerivSymbolic = sp.Matrix(SDeriv(0,x_list_sp,param))

# derivative of rhs wrt params
SDerivSymbolicJacParams = SDerivSymbolic.jacobian(params_sp)
SDerivSymbolicJacParamsLamb = sp.lambdify((x_list_sp,params_sp),SDerivSymbolicJacParams,'numpy')
SDerivSymbolicJacParamsLambFun = lambda t,x,params: SDerivSymbolicJacParamsLamb(x,params)

# derivative of rhs wrt Conc
SDerivSymbolicJacConc = SDerivSymbolic.jacobian(x_list_sp)
SDerivSymbolicJacConcLamb = sp.lambdify((x_list_sp,params_sp),SDerivSymbolicJacConc,'numpy')
SDerivSymbolicJacConcLambFun = lambda t,x,params: SDerivSymbolicJacConcLamb(x,params)

# sensitivity equations
def dSens(t,xs,params):
    dxs = []

    x = xs[:nVars]
    param_diff = []
    param_diff.extend(param_list[:16])
    param_diff.extend(params[:3])
    param_diff.extend(param_list[19:21])
    param_diff.append(params[-1])

    dxs.extend(SDeriv(0, x, param_diff))
    # sensitivity
    s = xs[nVars:]

    assert len(s) == nSensitivityEqs
    assert len(params) == nParams
    assert len(x) == nVars

    # compute sensitivity equations
    SDerivSymbolicJacParamsMat = SDerivSymbolicJacParamsLambFun(t,x,params)
    SDerivSymbolicJacConcMat = SDerivSymbolicJacConcLambFun(t,x,params)
    for i,_ in enumerate(x):
        for j,_ in enumerate(params):
            dxs.append(np.dot(SDerivSymbolicJacConcMat[i,:],
                              s[range(j,nSensitivityEqs,nParams)])
                       + SDerivSymbolicJacParamsMat[i,j])
    return dxs

#################################################
# setup ivp
#################################################

# get values of parameters
param_list_abbrev = []
param_list_abbrev.extend(param_list[16:19])
param_list_abbrev.append(param_list[-1])


# compute jacobian -- helpful for NDF
allVars = np.concatenate((x_list_sp,sensitivityVars))
dSensSym = sp.Matrix(dSens(0,allVars,param_list_abbrev))
dSensSymJac = dSensSym.jacobian(allVars)
dSensSymJacDenseMatLam = sp.lambdify(allVars,dSensSymJac)
dSensSymJacSparseMatLamFun = lambda t,xs: sparse.csr_matrix(dSensSymJacDenseMatLam(*xs))

# initial conditions
xs0 = np.concatenate([y0, np.zeros(nSensitivityEqs)])
xs0[range(len(y0)+2,len(y0)+nSensitivityEqs,nParams)] = 1 # for initial condition G0
xs0[range(len(y0)+3,len(y0)+nSensitivityEqs,nParams)] = 1 # for initial condition I0

# differential eq
dSensParams = lambda t,xs: dSens(t,xs,param_list_abbrev)


#################################################
# Integrate with BDF
#################################################

# time samples
fintime = 4000
tol = 1e-4
nsamples = 100
timeorig = np.linspace(0,fintime,nsamples)

# terminal event
event = lambda t,xs: np.absolute(dSensParams(t,xs)[nVars-1]) - tol
event.terminal = True

sol = solve_ivp(dSensParams,[0, fintime], xs0, method="BDF", jac = dSensSymJacSparseMatLamFun,
               events = event, t_eval=timeorig, atol=1.0e-6,
               rtol=1.0e-6)

print(sol.message)


#################################################
# Plot solution
#################################################


plt.plot(sol.t,sol.y[(nVars-5):nVars,:].T)
plt.legend(['G','H','P','A','I'],loc='upper right')
#plt.savefig('/Users/aarcher/PycharmProjects/MCP/WholeCell/plots/ScipyCode_MCPDynamics.png')
plt.show()


plt.plot(sol.t,sol.y[:7,:].T)
plt.legend(['N','D','G','H','P','A','I'],loc='upper right')
#filename = "VfDhaB_"+str(VfDhaB)+"_KmDhaBG_" + str(KmDhaBG) + "_KiDhaBH_" + str(KiDhaBH) + "_VfIcdE_"  + str(VfIcdE) + "_KmIcdEA_" + str(KmIcdEA) + "_KmIcdEN_" + str(KmIcdEN) + "_KiIcdED_" + str(KiIcdED) + "_KiIcdEI_" + str(KiIcdEI) + "_GInit_" + str(GInit) + "_NInit_" + "_GInit_" + str(GInit) + "_NInit_" + str(NInit) + "_DInit_" + str(DInit) + "_AInit_" + str(AInit)
#plt.savefig('/Users/aarcher/PycharmProjects/MCP/WholeCell/plots/ScipyCode_MCPDynamics.png')
plt.show()


namesExt = ['G','H','P','A','I']

for i in range(0,len(namesExt)):
    if i == 0:
        plt.plot(sol.t,sol.y[-nParams:,:].T)
    else:
        plt.plot(sol.t, sol.y[-(i+1)*nParams:-(i*nParams), :].T)
    plt.title(r'Sensitivity, $\partial ' + namesExt[i]+'/\partial p_i$, of the external concentration of '
              + namesExt[i] + ' wrt $p_i = k_m, k_c, G_0, I_0$')
    plt.xlabel('time')
    plt.ylabel(r'$\partial ' + namesExt[i]+'/\partial p_i$')
    plt.legend([r'$k_m$',r'$k_c$',r'$G_0$',r'$I_0$'],loc='upper right')
    plt.savefig('/Users/aarcher/PycharmProjects/MCP/WholeCell/plots/Perm_SensitivityExternal_'+ namesExt[i]
                + '_ngrid' + str(ngrid) +'.png')
    plt.show()


namesInt = ['N','D','G','H','P','A','I']
for i in range(0,len(namesInt)):
    plt.plot(sol.t,sol.y[(nVars + i*nParams):(nVars+(i+1)*nParams),:].T)
    plt.title(r'Sensitivity, $\partial ' + namesInt[i]+'/\partial p_i$, of the internal concentration of '
              + namesInt[i] +' wrt $p_i = k_m, k_c, G_0, I_0$')
    plt.xlabel('time')
    plt.ylabel(r'$\partial ' + namesInt[i]+'/\partial p_i$')
    plt.legend([r'$k_m$',r'$k_c$',r'$G_0$',r'$I_0$'],loc='upper right')
    plt.savefig('/Users/aarcher/PycharmProjects/MCP/WholeCell/plots/Perm_SensitivityInternal_'+ namesInt[i]
                + '_ngrid' + str(ngrid) +'.png')
    plt.show()




