import pandas as pd
from Whole_Cell_Engineered_System_DhaB_DhaT_Taylor import *
from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()
secstohrs = 60*60
secstomins = 60
###################################
### initialization
###################################

if rank == 0:
  ngrid= 100
  Ncells = 9 * (10**10)  
  fintime = 200*60*60
  mintime = 10**-10


  params = {'KmDhaTH': 0.77, # mM
        'KmDhaTN': 0.03, # mM
        'kcatfDhaT': 59.4, # (/s)
        'kcatfDhaB': None, # Input (/s)
        'KmDhaBG': None, # Input (mM)
        'km': 10**-4, 
        'kc': 10.**-4,
        'SigmaDhaB': None, # Input (mM)
        'SigmaDhaT': None, # Input (mM)
        'GInit': 200, #  2 * 10^(-4) mol/cm3 = 200 mM. 
        'NInit': 0.5, # mM
        'DInit': 0.5} # mM
  tolG = 0.01*params['GInit']
  filenum = 1
  filename = ["EncapsulationOnlyTableResultsSingleMCP(Surface Area)WithoutNumberMCP.xls",
              "EncapsulationAndExpressionTableResultsSingleMCP(Surface Area)WithoutNumberMCP.xls"]
  # dataframe of unknown values
  params_ET_DhaB_df  = pd.read_excel(filename[filenum])
  avogardos_number = 6.02 * (10 ** 23)
  params_ET_DhaB_df = np.array_split(params_ET_DhaB_df,size)

  integration_params = initialize_integration_params(ngrid=ngrid, Ncells =Ncells)
  integration_params['volmcp'] = None
  integration_params['tolG'] = tolG
  integration_params['fintime'] = fintime
  integration_params['mintime'] = mintime  
else:
  integration_params = None
  params = None
  params_ET_DhaB_df = None 

integration_params = comm.bcast(integration_params, root=0)
params = comm.bcast(params, root=0)
params_ET_DhaB_df_div = comm.scatter(params_ET_DhaB_df, root=0)
num_samples = len(params_ET_DhaB_df_div.index)

# stop condition
def event_Gmin(t,y):
    return y[-3] - integration_params['tolG']


for i in params_ET_DhaB_df_div.index:
  #################################################
  # Setup Params
  #################################################
  Rm= params_ET_DhaB_df_div.loc[i,'Effective Radius of MCP (m)']
  integration_params['Rm'] = Rm # In metres
  integration_params['m_m'] = (Rm**3)/3 #In metres^3
  integration_params['volmcp'] = 4*np.pi*(integration_params['Rm']**3)/3 # In metres^3

  params['kcatfDhaB'] = params_ET_DhaB_df_div.loc[i,'DhaB, Kcat (/s)']
  params['KmDhaBG'] = params_ET_DhaB_df_div.loc[i,'DhaB, Km (mM)']
  params['SigmaDhaB'] = params_ET_DhaB_df_div.loc[i,'Maximum Concentration of DhaB1 in MCP (mM)']
  params['SigmaDhaT'] = params_ET_DhaB_df_div.loc[i,'Maximum Concentration of DhaT in MCP (mM)']
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

  #################################################
  # Integrate with BDF
  #################################################

  # initial conditions
  n_compounds_cell = 3
  y0 = np.zeros((integration_params['ngrid'] + 2) * n_compounds_cell + 2)
  y0[-3] = params['GInit']  # y0[-3] gives the initial state of the external substrate.
  y0[0] = params['NInit']  # y0[0] gives the initial state of the external substrate.
  y0[1] = params['DInit']  # y0[1] gives the initial state of the external substrate.

  # time samples
  mintime = integration_params['mintime']
  tol = 1e-10
  nsamples = 500
  fintime = integration_params["fintime"]
  timeorig = np.logspace(np.log10(mintime), np.log10(fintime), nsamples)

  try:
    sol = solve_ivp(SDerivParameterized,[0, fintime+1], y0, method="BDF",jac=SDerivGradFunSparse, t_eval=timeorig,
                    atol=tol,rtol=tol, events=event_Gmin)
    #################################################
    ## Store Results
    #################################################

    #create grid
    M_cell = (integration_params['Rc'] / integration_params['Rm']) ** 3  # why?
    M_mcp = 1.
    Mgrid = np.linspace(M_mcp, M_cell, integration_params['ngrid'])
    DeltaM = np.divide((M_cell - M_mcp), (integration_params['ngrid']-1))
    volmcp = integration_params['volmcp']
    volcell = 4 * np.pi * (integration_params['Rc'] ** 3) / 3

    if sol.t_events[0].size != 0:
      params_ET_DhaB_df_div.loc[i,'Time to consume 99% of Glycerol (hrs)'] = sol.t_events[0][0]/secstohrs 
      params_ET_DhaB_df_div.loc[i,'Average Cytosolic Concentration of 3-HPO (mM)'] = (4*np.pi*integration_params['m_m']*sol.y[range(6,nVars-3,n_compounds_cell), -1]*DeltaM).sum(axis=0)/(volcell-volmcp)
      params_ET_DhaB_df_div.loc[i,'External Concentration of 1,3-PDO (mM)'] = sol.y[-1,-1]
    else:
      params_ET_DhaB_df_div.loc[i,'Time to consume  99% o Glycerol (hrs)'] = np.nan
      params_ET_DhaB_df_div.loc[i,'Average Cytosolic Concentration of 3-HPO (mM)'] = np.nan
      params_ET_DhaB_df_div.loc[i,'External Concentration of 1,3-PDO (mM)'] = np.nan

  except RuntimeError:
    params_ET_DhaB_df_div.loc[i,'Time to consume Glycerol (s)'] = np.nan
    params_ET_DhaB_df_div.loc[i,'Average Cytosolic Concentration of 3-HPO (mM)'] = np.nan
    params_ET_DhaB_df_div.loc[i,'External Concentration of 1,3-PDO (mM)'] = np.nan


params_ET_DhaB_df = comm.gather(params_ET_DhaB_df_div, root=0)
if rank == 0:
  params_ET_DhaB_df = pd.concat(params_ET_DhaB_df)
  params_ET_DhaB_df.to_excel(filename[filenum])
  # params_ET_DhaB_df.to_excel("test.xls")

