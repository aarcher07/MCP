'''
This script generates plots comparing the solutions as the grid 
is varied. This was done to find an optimal grid solutions. This isn't
a good idea as behaviours can vary drastically as parameters vary.

The code is parallelized using mpi4py.

Programme written by aarcher07
Editing History:
- 26/10/20
'''

from Whole_Cell_Engineered_System_IcdE import *
from mpi4py import MPI 
import numpy as np
import matplotlib.pyplot as plt


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


if rank == 0:
	integration_params = initialize_integration_params()
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
	          'km': 0.01,
	          'kc': 0.01,
	          'GInit': 10.,
	          'IInit': 10.,
	          'NInit': 20.,
	          'DInit': 20.}
	grid_sizes_global = [5,25,50,100]
	#grid_sizes_global = [3,4,5,10]
	grid_array_split = np.array_split(grid_sizes_global,size)

	# initial conditions
	dimscalings = initialize_dim_scaling(**params)

	# time samples
	mintime = 1
	dim_fintime = 10**7
	tol = 1e-12
	nsamples = 500
	dim_time = np.logspace(np.log10(mintime), np.log10(dim_fintime), nsamples)
	nondim_time = dim_time/params['km']
	nondim_fintime = dim_fintime/params['km']

	# plotting variables for rank 0 -- plotting in terms of dimensional time
	xvalslogtimeticks = list(range(int(np.log10(dim_fintime))+1))
	xtexlogtimeticks = [r'$10^{' + str(i) + '}$' for i in range(int(np.log10(dim_fintime))+1)]
	ngrid_legend = ['ngrid = ' + str(ngrid) for ngrid in  grid_sizes_global]
	names = ['N','D','G','H','P','A','I']

	# scalings
	M_cell = (integration_params['Rc'] / integration_params['Rm']) ** 3  # why?
	M_mcp = 1.
	scalings = list(dimscalings.values())
	volcell = 4 * np.pi * (integration_params['Rc']**3)/3
	volmcp = 4*np.pi * (integration_params['Rm'] ** 3) / 3
	volratio = integration_params['Vratio']


else:
	grid_sizes_global = None
	grid_array_split = None
	integration_params = None
	params = None
	dimscalings = None
	nondim_time = None
	nondim_fintime = None
	tol = None

# transmit to other processors
integration_params = comm.bcast(integration_params, root = 0)
params = comm.bcast(params, root = 0)
grid_sizes_local = comm.scatter(grid_array_split,root=0)
dimscalings = comm.bcast(dimscalings, root = 0)
nondim_time = comm.bcast(nondim_time, root = 0)
tol = comm.bcast(tol, root = 0)
nondim_fintime = comm.bcast(nondim_fintime, root = 0)

ysol_global = None
ysol_local = []
# on processors with non-zero grids, plot solutions
if len(grid_sizes_local) != 0:
	for i,ngrid in enumerate(grid_sizes_local):
		integration_params['ngrid'] = ngrid
		integration_params['nVars'] = 5*(2+int(ngrid)) + 2

		# spatial derivative
		SDerivParameterized = lambda t,x: SDeriv(t,x,integration_params,params)
		nVars = 5*(2+int(ngrid)) + 2
		x_list_sp = np.array(sp.symbols('x:' + str(nVars)))

		#jacobian
		SDerivSymbolic = SDerivParameterized(0,x_list_sp)
		SDerivGrad = sp.Matrix(SDerivSymbolic).jacobian(x_list_sp)
		SDerivGradFun = sp.lambdify(x_list_sp, SDerivGrad, 'numpy')
		SDerivGradFunSparse = lambda t,x: sparse.csr_matrix(SDerivGradFun(*x))

		# initial values
		n_compounds_cell = 5
		y0 = np.zeros((integration_params['ngrid'] + 2) * n_compounds_cell + 2)
		y0[-5] = params['GInit']/ dimscalings['G0']  # y0[-5] gives the initial state of the external substrate.
		y0[-1] = params['IInit'] / dimscalings['I0']  # y0[-1] gives the initial state of the external substrate.
		y0[0] = params['NInit'] / dimscalings['N0']  # y0[5] gives the initial state of the external substrate.
		y0[1] = params['DInit'] / dimscalings['D0']  # y0[6] gives the initial state of the external substrate.

		#################################################
		# Integrate with BDF
		#################################################

		sol = solve_ivp(SDerivParameterized,[0, nondim_fintime], y0, method="BDF",jac=SDerivGradFunSparse, t_eval=nondim_time,
		                atol=1.0e-5,rtol=1.0e-5)

		ysol_local.append(sol.y)


ysol_global = comm.gather(ysol_local, root = 0)

if rank == 0:

	ysol_global = sum(ysol_global,[]) # concatenate list of lists
	nVars = integration_params['nVars']
	fig, ax = plt.subplots(7, 3, figsize=(10,20))
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

	for i, (ngrid,ysol) in enumerate(zip(grid_sizes_global,ysol_global)):
		# rescale the solutions
		numeachcompound = 2 + ngrid
		ncompounds = 5
		nVars = 5*(2+int(ngrid)) + 2
		# non-dimensionalized solution
		ysol[:2, :] = (np.multiply(ysol[:2, :].T, scalings[:2])).T
		for i in range(numeachcompound):
		    j = range(2+i*ncompounds, 2+(i+1)*ncompounds)
		    ysol[j,:] = (np.multiply(ysol[j,:].T,scalings[2:])).T


		# MCP solutions
		for i in range(7):
		    logy = np.log10(volmcp*ysol[i,:].T)
		    ax[i,0].plot(np.log10(dim_time),logy)
		    logminy = int(round(np.min(logy)))
		    logmaxy = int(round(np.max(logy)))
		    MCP_minval = logminy if logminy < MCP_minval else MCP_minval
		    MCP_maxval = logmaxy if logmaxy > MCP_maxval else MCP_maxval

		# cellular solutions
		for i in range(ncompounds):
		    logy = np.log10((4*np.pi*integration_params['m_m']*ysol[range(7+i,nVars-5,ncompounds), :]*DeltaM).sum(axis=0))
		    ax[2+i,1].plot(np.log10(dim_time),logy)
		    logminy = int(round(np.min(logy)))
		    logmaxy = int(round(np.max(logy)))
		    cell_minval = logminy if logminy < cell_minval else cell_minval
		    cell_maxval = logmaxy if logmaxy > cell_maxval else cell_maxval


		# external solutions
		for i in reversed(range(5)):
		    logy = np.log10((volcell/volratio)*ysol[-i-1,:].T)
		    ax[2+i,2].plot(np.log10(dim_time),logy)
		    logminy = int(round(np.min(logy)))
		    logmaxy = int(round(np.max(logy)))
		    ext_minval = logminy if logminy < ext_minval else ext_minval
		    ext_maxval = logmaxy if logmaxy > ext_maxval else ext_maxval


	#set ticks
	MCPyvalslogtimeticks = list(range(MCP_minval,MCP_maxval+1))
	MCPytexlogtimeticks = [r'$10^{' + str(i) + '}$' for i in range(MCP_minval,MCP_maxval+1)]
	cellyvalslogtimeticks = list(range(cell_minval,cell_maxval+1))
	cellytexlogtimeticks = [r'$10^{' + str(i) + '}$' for i in range(cell_minval,cell_maxval+1)]
	extlogtimeticks = list(range(ext_minval,ext_maxval+1))
	extytexlogtimeticks = [r'$10^{' + str(i) + '}$' for i in range(ext_minval,ext_maxval+1)]

	# edit plots except the first two rows

	for i in range(ax.shape[0]):
		if i < 2:
			ax[i,0].legend(ngrid_legend,fontsize=5)
			ax[i,0].set_xlabel('log(time)')
			ax[i,0].set_ylabel('log(mass)')	
			ax[i,0].set_xticks(xvalslogtimeticks)
			ax[i,0].set_xticklabels(xtexlogtimeticks)
			ax[i,0].set_title('Plot of MCP masses of ' + names[i])
			ax[i,0].set_yticks(MCPyvalslogtimeticks)
			ax[i,0].set_yticklabels(MCPytexlogtimeticks)
		else:
			for j in range(ax.shape[1]):
				ax[i,j].legend(ngrid_legend,fontsize=6)
				ax[i,j].set_xlabel('log(time)')
				ax[i,j].set_ylabel('log(mass)')	
				ax[i,j].set_xticks(xvalslogtimeticks)
				ax[i,j].set_xticklabels(xtexlogtimeticks)
				if j == 0:
					ax[i,j].set_title('Plot of MCP masses of ' + names[i])
					ax[i,j].set_yticks(MCPyvalslogtimeticks)
					ax[i,j].set_yticklabels(MCPytexlogtimeticks)
				if j == 1:
					ax[i,j].set_title('Plot of cellular masses ' + names[i])
					ax[i,j].set_yticks(cellyvalslogtimeticks)
					ax[i,j].set_yticklabels(cellytexlogtimeticks)
				if j == 2:
					ax[i,j].set_title('Plot of external masses ' + names[i])
					ax[i,j].set_yticks(extlogtimeticks)
					ax[i,j].set_yticklabels(extytexlogtimeticks)

	plt.show()
