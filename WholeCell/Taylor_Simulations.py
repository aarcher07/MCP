import pandas as pd
from Whole_Cell_Engineered_System_DhaB_DhaT_Taylor import *


###################################
### Initialization
###################################

ngrid = 100
Ncells = 9 * (10**10)  
secstohrs = 60*60
fintime = 200*60*60
mintime = 10**-10

params_df = pd.read_excel("EncapsulationAndExpressionTableResults(Surface Area).xls")
integration_params = initialize_integration_params(ngrid=ngrid, Ncells =Ncells, Rm= params_df.iloc[-1]['Effective Radius of MCP (m)'])

Rm = integration_params["Rm"] # In metres
volmcp = 4*np.pi*(integration_params['Rm']**3)/3 # In metres^3
params = {'KmDhaTH': 0.77, # mM
      'KmDhaTN': 0.03, # mM
      'kcatfDhaT': 59.4, # seconds
      'kcatfDhaB': params_df.iloc[-1]['DhaB, Kcat (/s)'], # Input
      'KmDhaBG': params_df.iloc[-1]['DhaB, Km (mM)'], # Input
      'km': 10**-6, 
      'kc': 10**-6,
      'SigmaDhaB': params_df.iloc[-1]['Maximum Concentration of DhaB1 in MCP (mM)'], # Input
      'SigmaDhaT': params_df.iloc[-1]['Maximum Concentration of DhaT in MCP (mM)'], # Input
      'GInit': 200, #  2 * 10^(-4) mol/cm3 = 200 mM. 
      'NInit': 0.5, # mM
      'DInit': 0.5} # mM


tolG = 0.01*params['GInit']



#################################################
# Setup Intergration Functions  
#################################################

# spatial derivative
SDerivParameterized = lambda t,x: SDeriv(t,x,integration_params,params)
nVars = integration_params['nVars']
x_list_sp = np.array(sp.symbols('x:' + str(nVars)))

#jacobian
SDerivSymbolic = SDerivParameterized(0,x_list_sp)
SDerivGrad = sp.Matrix(SDerivSymbolic).jacobian(x_list_sp)
SDerivGradFun = sp.lambdify(x_list_sp, SDerivGrad, 'numpy')
SDerivGradFunSparse = lambda t,x: sparse.csr_matrix(SDerivGradFun(*x))

# stop condition
def event_Gmin(t,y):
    return y[-3] - tolG
#################################################
# Integrate with BDF
#################################################

# initial conditions
n_compounds_cell = 3
y0 = np.zeros((integration_params['ngrid'] + 2) * n_compounds_cell + 2)
y0[-3] = params['GInit']  # y0[-5] gives the initial state of the external substrate.
y0[0] = params['NInit']  # y0[5] gives the initial state of the external substrate.
y0[1] = params['DInit']  # y0[6] gives the initial state of the external substrate.

# time samples
tol = 1e-10
nsamples = 500
timeorig = np.logspace(np.log10(mintime), np.log10(fintime), nsamples)


time_1 = time.time()
sol = solve_ivp(SDerivParameterized,[0, fintime+1], y0, method="BDF",jac=SDerivGradFunSparse, t_eval=timeorig,
                atol=tol,rtol=tol, events=event_Gmin)

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
print(sol.t_events[0])
#################################################
# Plot solution
#################################################
volcell = 4*np.pi*(integration_params['Rc']**3)/3
volmcp = 4 * np.pi * (integration_params['Rm'] ** 3) / 3
volratio = integration_params['Vratio']


# rescale the solutions
numeachcompound = 2 + integration_params['ngrid']
ncompounds = 3
timeorighours = timeorig/secstohrs
# cellular solutions
minval = np.inf
maxval = -np.inf
for i in range(0,ncompounds):
    ycell = sol.y[8+i, :]
    plt.plot(timeorighours,ycell)


plt.title('Plot of cellular masses')
plt.legend(['G', 'H', 'P'], loc='upper right')
# plt.legend(['H','P'],loc='upper right')
plt.xlabel('time (hr)')
plt.ylabel('concentration (mM)')
plt.show()


meancellularconcH = (4*np.pi*integration_params['m_m']*sol.y[range(6,nVars-3,ncompounds), -1]*DeltaM).sum(axis=0)/(volcell-volmcp)
print(meancellularconcH)
# external solutions
minval = np.inf
maxval = -np.inf
for i in reversed(range(0,3)):
    yext = sol.y[-i-1,:].T
    plt.plot(timeorighours,yext)
plt.ylim([0,3*max(sol.y[-2,:])])
plt.legend(['G','H','P'],loc='upper right')
# plt.legend(['H','P'],loc='upper right')
plt.title('Plot of external masses')
plt.xlabel('time (hr)')
plt.ylabel('concentration (mM)')
plt.show()

#MCP solutions
minval = np.inf
maxval = -np.inf
for i in range(5):
    ymcp = sol.y[i,:].T
    plt.plot(timeorighours,ymcp)


plt.legend(['N','D','G','H','P'],loc='upper right')
# plt.legend(['P'],loc='upper right')

plt.title('Plot of MCP masses')
plt.xlabel('time (hr)')
plt.ylabel('concentration (mM)')
plt.show()

#check mass balance
ext_masses_org = y0[-3:]* (volcell/volratio)
cell_masses_org = y0[8:11] * (volcell - volmcp)
mcp_masses_org = y0[:5] * volmcp


ext_masses_fin = sol.y[-3:, -1] * (volcell/volratio)
cell_masses_fin = sol.y[8:11,-1] * (volcell - volmcp)
mcp_masses_fin = sol.y[:5, -1] * volmcp
print(ext_masses_org.sum() + Ncells*cell_masses_org.sum() + Ncells*mcp_masses_org.sum())
print(ext_masses_fin.sum() + Ncells*cell_masses_fin.sum() + Ncells*mcp_masses_fin.sum())
print((sol.y[-3:, -1]).sum()*(volcell/volratio+Ncells*volmcp+Ncells*volcell))