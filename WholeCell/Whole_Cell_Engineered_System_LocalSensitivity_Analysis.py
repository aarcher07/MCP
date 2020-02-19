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
params_sp = list(sp.symbols('alpha0:10'))
params_sp.extend(sp.symbols('beta0:3'))
params_sp.extend(sp.symbols('gamma:3'))

params_sp.append(sp.symbols('km'))
params_sp.append(sp.symbols('kc'))
# initial conditions symbols
params_sp.append(sp.symbols('GInit'))
params_sp.append(sp.symbols('NInit'))
params_sp.append(sp.symbols('DInit'))
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

SDerivSymbolic = sp.Matrix(SDeriv(0,x_list_sp,params_sp))

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

    # concentration
    x = xs[:nVars]
    dxs.extend(SDeriv(0, x, params))

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
            dxs.append(np.dot(SDerivSymbolicJacConcMat[i,:],s[range(j,nSensitivityEqs,nParams)]) + SDerivSymbolicJacParamsMat[i,j])

    print('hi')
    return dxs

#################################################
# solve ivp
#################################################
print('hi')
# allVars = np.concatenate((x_list_sp,sensitivityVars))
# dSensSym = sp.Matrix(dSens(0,allVars,param_list))
# dSensSymJac = dSensSym.jacobian(allVars)
# dSensSymJacDenseMatLam = sp.lambdify(allVars,dSensSymJac)
# dSensSymJacSparseMatLamFun = lambda t,xs: sparse.csr_matrix(dSensSymJacDenseMatLam(*xs))

print('hii')
# initial conditions
xs0 = np.concatenate([y0, np.zeros(nSensitivityEqs)])
xs0[len(y0)] = 1
xs0[len(y0)+1] = 1
xs0[-5] = 1
xs0[-1] = 1

# differential eq
dSensParams = lambda t,xs: dSens(0,xs,param_list)

# time samples
fintime = 20
tol = 1e-4
nsamples = int(10)
timeorig = np.linspace(0,fintime,nsamples)

# terminal event
event = lambda t,xs: np.absolute(dSensParams(t,xs)[nVars-1]) - tol
event.terminal = True

sol = solve_ivp(dSensParams,[0, fintime], xs0, method="BDF", events = event,
                t_eval=timeorig, atol=1.0e-2,rtol=1.0e-2)

print(sol.message)

plt.plot(sol.t,sol.y[nVars:(nVars+nParams),:].T)
#plt.legend(['G','H','P','A','I'],loc='upper right')
#filename = "VfDhaB_"+str(VfDhaB)+"_KmDhaBG_" + str(KmDhaBG) + "_KiDhaBH_" + str(KiDhaBH) + "_VfIcdE_"  + str(VfIcdE) + "_KmIcdEA_" + str(KmIcdEA) + "_KmIcdEN_" + str(KmIcdEN) + "_KiIcdED_" + str(KiIcdED) + "_KiIcdEI_" + str(KiIcdEI) + "_GInit_" + str(GInit) + "_NInit_" + "_GInit_" + str(GInit) + "_NInit_" + str(NInit) + "_DInit_" + str(DInit) + "_AInit_" + str(AInit)
#plt.savefig('/Users/aarcher/PycharmProjects/MCP/WholeCell/plots/ScipyCode_ExternalDynamics.png')
plt.show()


