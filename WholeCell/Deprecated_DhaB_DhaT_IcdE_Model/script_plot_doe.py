from Whole_Cell_Engineered_System_IcdE import *
from mpi4py import MPI 
import numpy as np
import matplotlib.pyplot as plt

Vratio = 10**-1
ngrid = 3
integration_params = initialize_integration_params(Vratio = Vratio,ngrid = ngrid,Rm = 1.e-5,Rc = 5.e-5)
# params = {'KmDhaTH': 1.,
#           'KmDhaTN': 1.,
#           'KiDhaTD': 1.,
#           'KiDhaTP': 1.,
#           'VfDhaT': 1.,
#           'VfDhaB': 1.,
#           'KmDhaBG': 1.,
#           'KiDhaBH': 1.,
#           'VfIcdE' : 1.,
#           'KmIcdED' : 1.,
#           'KmIcdEI' : 1.,
#           'KiIcdEN' : 1.,
#           'KiIcdEA' : 1.,
#           'km': 10.,
#           'kc': 1.,
#           'k1': 1.,
#           'k-1': 1.,
#           'DhaB2Exp': 1.,
#           'iDhaB1Exp': 3.,
#           'Sigma': 1.,
#           'GInit': 10.,
#           'IInit': 20.,
#           'NInit': 20.,
#           'DInit': 20.}


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


# initial conditions
dimscalings = initialize_dim_scaling(**params)
tol = 1e-12
nsamples = 10**4


# time samples
mintime_seconds = 10**-4
dim_fintime_seconds = 5*60
dim_time_seconds = np.logspace(np.log10(mintime_seconds), np.log10(dim_fintime_seconds), nsamples)

# non dimensionalized time
t0 = 3*integration_params['Rm']/params['km']
nondim_time = dim_time_seconds/t0
nondim_fintime = dim_fintime_seconds/t0

#time sample in mins
mintime_min = mintime_seconds/(60)
dim_fintime_min = dim_fintime_seconds/(60)
dim_time_min = dim_time_seconds/(60)

#time sample in hours
mintime_hours = mintime_seconds/(60.**2)
dim_fintime_hours = dim_fintime_seconds/(60.**2)
dim_time_hours = dim_time_seconds/(60.**2)


# plotting variables for rank 0 -- plotting in terms of dimensional time, seconds
xticksrange_seconds= range(int(np.log10(mintime_seconds)), int(np.log10(dim_fintime_seconds))+1)
xvalslogtimeticks_seconds = list(xticksrange_seconds)
xtexlogtimeticks_seconds = [r'$10^{' + str(i) + '}$' for i in xvalslogtimeticks_seconds]

# plotting variables for rank 0 -- plotting in terms of dimensional time, min
xticksrange_min= range(int(np.log10(mintime_min)), int(np.log10(dim_fintime_min))+1)
xvalslogtimeticks_min = list(xticksrange_min)
xtexlogtimeticks_min = [r'$10^{' + str(i) + '}$' for i in xvalslogtimeticks_min]


# plotting variables for rank 0 -- plotting in terms of dimensional time, hours
xticksrange_hours= range(int(np.log10(mintime_hours)), int(np.log10(dim_fintime_hours))+1)
xvalslogtimeticks_hours = list(xticksrange_hours)
xtexlogtimeticks_hours = [r'$10^{' + str(i) + '}$' for i in xvalslogtimeticks_hours]

names = ['NADH','NAD+','iDhaB','Glycerol','3-HPA','1,3-PD0','a-KG','Isocitrate']

# scalings
M_cell = (integration_params['Rc'] / integration_params['Rm']) ** 3  # why?
M_mcp = 1.
scalings = list(dimscalings.values())
volcell = 4 * np.pi * (integration_params['Rc']**3)/3
volmcp = 4*np.pi * (integration_params['Rm'] ** 3) / 3
volratio = integration_params['Vratio']

# spatial derivative
SDerivParameterized = lambda t,x: SDeriv(t,x,integration_params,params)
nVars = 5*(2+int(ngrid)) + 3
x_list_sp = np.array(sp.symbols('x:' + str(nVars)))

#jacobian
SDerivSymbolic = SDerivParameterized(0,x_list_sp)
SDerivGrad = sp.Matrix(SDerivSymbolic).jacobian(x_list_sp)
SDerivGradFun = sp.lambdify(x_list_sp, SDerivGrad, 'numpy')
SDerivGradFunSparse = lambda t,x: sparse.csr_matrix(SDerivGradFun(*x))

# initial values
n_compounds_cell = 5
y0 = np.zeros(integration_params['nVars'])
y0[-5] = params['GInit']/ dimscalings['G0']  # y0[-5] gives the initial state of the external substrate.
y0[-1] = params['IInit'] / dimscalings['I0']  # y0[-1] gives the initial state of the external substrate.
y0[0] = params['NInit'] / dimscalings['N0']  # y0[5] gives the initial state of the external substrate.
y0[1] = params['DInit'] / dimscalings['D0']  # y0[6] gives the initial state of the external substrate.
y0[2] = params['iDhaB1Exp']/ (params['iDhaB1Exp'] + params['DhaB2Exp'])
#################################################
# Integrate with BDF
#################################################

sol = solve_ivp(SDerivParameterized,[0, nondim_fintime + 1], y0, method="BDF",jac=SDerivGradFunSparse, t_eval=nondim_time,
                atol=1.0e-5,rtol=1.0e-5)

ysol = sol.y


nVars = integration_params['nVars']
fig, ax = plt.subplots(1, 3)
fig.tight_layout(w_pad=0.5, h_pad=1.0) 
#create grid
Mgrid = np.linspace(M_mcp, M_cell, integration_params['ngrid']-1)
DeltaM = np.divide((M_cell - M_mcp), (integration_params['ngrid']-1))
Mgridfull = np.concatenate(([M_mcp- DeltaM],Mgrid, [M_cell+ DeltaM]))

cell_minval = np.inf
cell_maxval = -np.inf

ext_minval = np.inf
ext_maxval = -np.inf

MCP_minval = np.inf
MCP_maxval = -np.inf

#################################################
# Plot solution
#################################################

# rescale the solutions
numeachcompound = 2 + ngrid
ncompounds = 5
nVars = 5*(2+int(ngrid)) + 3
# non-dimensionalized solution
ysol[:3, :] = (np.multiply(sol.y[:3, :].T, scalings[:3])).T
for i in range(numeachcompound):
    j = range(3+i*ncompounds, 3+(i+1)*ncompounds)
    ysol[j,:] = (np.multiply(ysol[j,:].T,scalings[3:])).T


# MCP solutions
for i in range(8):
    # logy = np.log10(volmcp*ysol[i,:].T)
    # # logy = np.log10(ysol[i,:].T)
    # ax[0].plot(np.log10(dim_time_hours),logy)
    # logminy = int(round(np.min(logy)))
    # logmaxy = int(round(np.max(logy)))
    # MCP_minval = logminy if logminy < MCP_minval else MCP_minval
    # MCP_maxval = logmaxy if logmaxy > MCP_maxval else MCP_maxval

    y = volmcp*ysol[i,:].T
    ax[0].plot(dim_time_min,y)


# cellular solutions
for i in range(ncompounds):
    # logy = np.log10((4*np.pi*integration_params['m_m']*ysol[range(8+i,nVars-5,ncompounds), :]*DeltaM).sum(axis=0))
    # ax[1].plot(np.log10(dim_time_hours),logy)
    # logminy = int(round(np.min(logy)))
    # logmaxy = int(round(np.max(logy)))
    # cell_minval = logminy if logminy < cell_minval else cell_minval
    # cell_maxval = logmaxy if logmaxy > cell_maxval else cell_maxval

    y = (4*np.pi*integration_params['m_m']*ysol[range(8+i,nVars-5,ncompounds), :]*DeltaM).sum(axis=0)
    ax[1].plot(dim_time_min,ysol[13+i])


# external solutions
for i in reversed(range(5)):
    # logy = np.log10((volcell/volratio)*ysol[-i-1,:].T)
    # # logy = np.log10(ysol[-i-1,:].T)
    # ax[2].plot(np.log10(dim_time_hours),logy)
    # logminy = int(round(np.min(logy)))
    # logmaxy = int(round(np.max(logy)))

    # ext_minval = logminy if logminy < ext_minval else ext_minval
    # ext_maxval = logmaxy if logmaxy > ext_maxval else ext_maxval

    y = volcell/volratio*ysol[-i-1,:].T
    ax[2].plot(dim_time_min,ysol[-i-1,:].T)
print('Steady state concentration of external P is: ' +str(ysol[-3,-1]))
print('Steady state concentration of external H is: ' +str(ysol[-4,-1]))
print('Steady state concentration of cellular H is: ' +str(ysol[-14,-1]))
#set tick
# MCPyvalslogtimeticks = list(range(MCP_minval,MCP_maxval+1))
# MCPytexlogtimeticks = [r'$10^{' + str(i) + '}$' for i in range(MCP_minval,MCP_maxval+1)]
# cellyvalslogtimeticks = list(range(cell_minval,cell_maxval+1))
# cellytexlogtimeticks = [r'$10^{' + str(i) + '}$' for i in range(cell_minval,cell_maxval+1)]
# extlogtimeticks = list(range(ext_minval,ext_maxval+1))
# extytexlogtimeticks = [r'$10^{' + str(i) + '}$' for i in range(ext_minval,ext_maxval+1)]


# edit plots except the first two rows


ax[0].legend(names,fontsize=10)
ax[0].set_xlabel('time/min')
ax[0].set_ylabel('mass/g')	
ax[0].set_title('Plot of MCP masses')
# ax[0].set_xticks(xvalslogtimeticks_hours)
# ax[0].set_xticklabels(xtexlogtimeticks_hours)
# ax[0].set_yticks(MCPyvalslogtimeticks)
# ax[0].set_yticklabels(MCPytexlogtimeticks)

ax[1].legend(names[3:],fontsize=10)
ax[1].set_xlabel('time/min')
ax[1].set_ylabel('mass/g')	
ax[1].set_title('Plot of cellular masses')
# ax[1].set_xticks(xvalslogtimeticks_hours)
# ax[1].set_xticklabels(xtexlogtimeticks_hours)
# ax[1].set_yticks(cellyvalslogtimeticks)
# ax[1].set_yticklabels(cellytexlogtimeticks)

ax[2].legend(names[3:],fontsize=10)
ax[2].set_xlabel('time/min')
ax[2].set_ylabel('mass/g')	
ax[2].set_title('Plot of external masses')
# ax[2].set_xticks(xvalslogtimeticks_hours)
# ax[2].set_xticklabels(xtexlogtimeticks_hours)
# ax[2].set_yticks(extlogtimeticks)
# ax[2].set_yticklabels(extytexlogtimeticks)

plt.show()
