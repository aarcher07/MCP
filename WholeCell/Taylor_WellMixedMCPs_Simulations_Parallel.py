import pandas as pd
from Whole_Cell_Engineered_System_WellMixed_MCPs_DhaB_DhaT_Taylor import *
from mpi4py import MPI
import numpy as np
from sklearn.metrics import auc

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()
secstohrs = 60*60
secstomins = 60
###################################
### initialization
###################################

if rank == 0:
  NcellsPerMCubed = 8 * (10**14) # cells per  metre^3
  fintime = 200*60*60
  mintime = 10**-10

  params = {'KmDhaTH': 0.77, # mM
        'KmDhaTN': 0.03, # mM
        'kcatfDhaT': 59.4, # (/s)
        'kcatfDhaB': None, # Input (/s)
        'KmDhaBG': None, # Input (mM)
        'km': 10**-7, #metres per second
        'kc': 10.**-5, #metres per second
        'SigmaDhaB': None, # Input (mM)
        'SigmaDhaT': None, # Input (mM)
        'GInit': 200, #  2 * 10^(-4) mol/cm3 = 200 mM 
        'NInit': 1, # mM
        'DInit': 1} # mM

  filenum = 0
  cellular_geometry = "rod"
  additional_name = "_constant_NADH_1mM_"
  #additional_name = "_"
  if cellular_geometry == "rod":
    Rc = 0.375e-6
    Lc = 2.47e-6
  elif cellular_geometry == "sphere":
    Rc = 0.68e-6 
    Lc = None

  filename = ["EncapsulationOnlyTableResultsWellMixed.xls",
              "EncapsulationAndExpressionResultsWellMixedMCP.xls"]
  # dataframe of unknown values
  params_ET_DhaB_df  = pd.read_excel(filename[filenum])
  avogardos_number = 6.02 * (10 ** 23)
  params_ET_DhaB_df = np.array_split(params_ET_DhaB_df,size)
  external_volume =  9e-6
  Ncells = external_volume*NcellsPerMCubed
  integration_params = initialize_integration_params(external_volume = external_volume, Rc = Rc, Lc = Lc,
                                                       Ncells =Ncells, cellular_geometry=cellular_geometry)
  integration_params['volmcp'] = 4*np.pi*(integration_params['Rm']**3)/3 # In metres^3
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
    return y[-3] - 0.01*params['GInit']
def event_Pmax(t,y):
    return y[-1] - 0.99*params['GInit']

for i in params_ET_DhaB_df_div.index:
  #################################################
  # Setup Params
  #################################################
  Nmcp = params_ET_DhaB_df_div.loc[i,'Number of MCPs']
  Rm = params_ET_DhaB_df_div.loc[i,'Effective Radius of MCP (m)']
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
  y0 = np.zeros(3 * n_compounds_cell + 2)
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
                    atol=tol,rtol=tol, events=[event_Gmin,event_Pmax])
    #################################################
    ## Store Results
    #################################################

    if sol.t_events[0].size != 0:
      params_ET_DhaB_df_div.loc[i,'Time to consume 99% of Glycerol (hrs)'] = sol.t_events[0][0]/secstohrs 
    else:
      params_ET_DhaB_df_div.loc[i,'Time to consume  99% of Glycerol (hrs)'] = np.nan

    if sol.t_events[1].size != 0:
      params_ET_DhaB_df_div.loc[i,'Time to produce 99% of 1,3-PDO (hrs)'] = sol.t_events[1][0]/secstohrs 
    else:
      params_ET_DhaB_df_div.loc[i,'Time to produce 99% of 1,3-PDO (hrs)'] = np.nan

    params_ET_DhaB_df_div.loc[i,'Exposure to 3-HPA (mM hrs)'] = auc(timeorig/secstohrs,sol.y[6, :])
    params_ET_DhaB_df_div.loc[i,'Maximum concentration of 3-HPA (mM)'] = np.max(sol.y[6, :])
    params_ET_DhaB_df_div.loc[i,'External Concentration of 1,3-PDO at steady state (mM)'] = sol.y[-1,-1]

  except RuntimeError:
    params_ET_DhaB_df_div.loc[i,'Time to consume  99% of Glycerol (hrs)'] = np.nan
    params_ET_DhaB_df_div.loc[i,'Time to consume  99% of 1,3-PDO (hrs)'] = np.nan
    params_ET_DhaB_df_div.loc[i,'Exposure to 3-HPA (mM hrs)'] = np.nan
    params_ET_DhaB_df_div.loc[i,'External Concentration of 1,3-PDO at steady state  (mM)'] = np.nan
    params_ET_DhaB_df_div.loc[i,'Maximum concentration of 3-HPA (mM)'] = np.nan


params_ET_DhaB_df = comm.gather(params_ET_DhaB_df_div, root=0)
if rank == 0:
  params_ET_DhaB_df = pd.concat(params_ET_DhaB_df)
  params_ET_DhaB_df.to_excel(cellular_geometry +additional_name + filename[filenum])

