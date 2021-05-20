import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, '.')
from dhaB_dhaT_model import DhaBDhaTModel
from data_set_constants import *
from model_constants import *
def test_devs():
    external_volume = 0.002

    params = {'cellperGlyMass': 10**(5.73158464),
                'PermCellGlycerol': 10**(-3.55285234),
                'PermCellPDO': 10**(-3.85344833),
                'PermCell3HPA': 10**(-2.34212333),
                'VmaxfDhaB': 10**(3.26266939),
                'KmDhaBG': 10**(0.71152905) ,
                'VmaxfDhaT': 10**(2.85561206),
                'KmDhaTH': 10**(0.69665821),
                'VmaxfGlpK':10**(1.99560497) ,
                'KmGlpKG': 10**(-1.24867452)}

    init_conds={'G_CYTO_INIT': 0.,
                'H_CYTO_INIT': 0.,
                'P_CYTO_INIT': 0,
                'G_EXT_INIT': 50.,
                'H_EXT_INIT': 0.,
                'P_EXT_INIT': 0,
                'CELL_CONC_INIT': 10**8,
                }


    dhaB_dhaT_model = DhaBDhaTModel(external_volume=external_volume)

    mintime = 10**(-15)
    fintime = 12*60*60

    #################################################
    # Integrate with BDF
    #################################################

    # initial conditions
    n_compounds_cell = 3
    y0 = np.zeros(dhaB_dhaT_model.nvars)
    for i,init_names in enumerate(VARIABLE_INIT_NAMES):
        y0[i] = init_conds[init_names]

    tol = 1e-7
    nsamples = 500
    timeorig = np.logspace(np.log10(mintime), np.log10(fintime), nsamples)
    dhaB_dhaT_model._set_symbolic_sderiv_conc_fun()
    ds = lambda t,x: dhaB_dhaT_model._sderiv(t, x, params)
    ds_jac = lambda t,x: dhaB_dhaT_model._sderiv_jac_conc_fun(t,x,params)

    sol = solve_ivp(ds,[0, fintime+1], y0, method = 'BDF', jac = ds_jac, t_eval=timeorig,
                    atol=tol,rtol=tol)

    print(sol.message)

    #################################################
    # Plot solution
    #################################################
    volcell = dhaB_dhaT_model.cell_volume
    colour = ['b','r','y','c','m']

    # rescale the solutions
    timeorighours = sol.t/HRS_TO_SECS

    # external solution
    for i in range(0,3):
        yext = sol.y[3+i,:].T
        plt.plot(timeorighours,yext, colour[i])

    plt.legend(['Glycerol','3-HPA','1,3-PDO'],loc='upper right')
    plt.title('Plot of external concentration')
    plt.xlabel('time (hr)')
    plt.ylabel('concentration (mM)')
    plt.show()

    #cell solutions
    for i in range(3):
        ymcp = sol.y[i,:].T
        plt.plot(timeorighours,ymcp, colour[i])


    plt.legend(['Glycerol','3-HPA','1,3-PDO'],loc='upper right')
    plt.title('Plot of cytosol concentrations')
    plt.xlabel('time (hr)')
    plt.ylabel('concentration (mM)')
    plt.show()

    plt.plot(timeorighours,sol.y[-1,:].T/((10**-0.3531)*DCW_TO_COUNT_CONC), colour[i])
    plt.title('Plot of cell concentration')
    plt.xlabel('time (hr)')
    plt.ylabel('concentration (cell per m^3)')
    plt.show()

    #check mass balance
    ext_masses_org = y0[3:6]* external_volume
    cell_masses_org = y0[:3] * volcell
    ext_masses_fin = sol.y[3:6, -1] * external_volume
    cell_masses_fin = sol.y[:3,-1] * volcell
    print(ext_masses_org.sum() + external_volume*y0[-1]*cell_masses_org.sum())
    print(ext_masses_fin.sum() + external_volume*sol.y[-1,-1]*cell_masses_fin.sum())


def test_QoI():
    external_volume = 0.002

    params = {'scalar': 0.5,
                'cellperGlyMass': 10**(5.73158464),
                'PermCellGlycerol': 10**(-3.55285234),
                'PermCellPDO': 10**(-3.85344833),
                'PermCell3HPA': 10**(-2.34212333),
                'VmaxfDhaB': 10**(3.26266939),
                'KmDhaBG': 10**(0.71152905) ,
                'VmaxfDhaT': 10**(2.85561206),
                'KmDhaTH': 10**(0.69665821),
                'VmaxfGlpK':10**(1.99560497),
                'KmGlpKG': 10**(-1.24867452)}

    init_conds={'G_CYTO_INIT': 0.,
                'H_CYTO_INIT': 0.,
                'P_CYTO_INIT': 0,
                'G_EXT_INIT': 50.,
                'H_EXT_INIT': 0.,
                'P_EXT_INIT': 0,
                'CELL_CONC_INIT': 10**8,
                }


    dhaB_dhaT_model = DhaBDhaTModel(external_volume=external_volume)
    qoi_vals = dhaB_dhaT_model.QoI(params,init_conds)
    print(qoi_vals)

    plt.plot(TIME_EVALS,qoi_vals[:,0])
    plt.title('Plot of external glycerol')
    plt.xlabel('time (hr)')
    plt.ylabel('concentration (mM)')
    plt.show()

    plt.plot(TIME_EVALS,qoi_vals[:,1])
    plt.title('Plot of external 1,3-PDO')
    plt.xlabel('time (hr)')
    plt.ylabel('concentration (mM)')
    plt.show()

    plt.plot(TIME_EVALS,qoi_vals[:,2])
    plt.title('Plot of dry cell weight')
    plt.xlabel('time (hr)')
    plt.ylabel('concentration (g per m^3)')
    plt.show()

if __name__ == '__main__':
    test_devs()
    test_QoI()
