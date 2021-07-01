import numpy as np
from scipy.integrate import BDF,solve_ivp
import matplotlib as mpl
mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'
import matplotlib.pyplot as plt
from base_dhaB_dhaT_model import *
from scipy.interpolate import UnivariateSpline
from csaps import csaps

def test_devs():
    gly_cond = 50
    x=TIME_SAMPLES[gly_cond]
    y = DATA_SAMPLES[gly_cond][:,2]
    spl = csaps(TIME_SAMPLES[gly_cond],DATA_SAMPLES[gly_cond][:,2], smooth = 0.6)
    cell_per_metres_cubed = lambda t: spl(t)*DCW_TO_COUNT_CONC
    plt.plot(x, y, 'ro', ms=5)
    plt.plot(x, spl(x), 'g', lw=3)

    time = []
    xs = TIME_SAMPLES[gly_cond]
    cell_per_metres_cubed_discrete = []
    N = 1000
    t, step_size = np.linspace(0,xs.iloc[-1],num=N,retstep=True)

    Navg = 10
    for i in range(N-1):
        tavg = np.linspace(t[i],t[i+1],num=Navg)
        splmeansvals = np.mean(spl(tavg))
        time.append(t[i])
        cell_per_metres_cubed_discrete.append(splmeansvals)
    time.append(t[-1])
    cell_per_metres_cubed_discrete.append(splmeansvals)
    plt.step(time,cell_per_metres_cubed_discrete,where='post')
    plt.legend(['spline', 'step'], loc='upper right')

    plt.show()

    external_volume = 0.002

    params = {'cellperGlyMass': 10 ** (5.73158464),
              'PermCellGlycerol': 10 ** (-3.55285234),
              'PermCellPDO': 10 ** (-3.85344833),
              'PermCell3HPA': 10 ** (-2.34212333),
              'VmaxfDhaB': 10 ** (3.26266939),
              'KmDhaBG': 10 ** (0.71152905),
              'VmaxfDhaT': 10 ** (3.5561206),
              'KmDhaTH': 10 ** (0.69665821),
              'VmaxfGlpK': 0,
              'KmGlpKG': 10 ** (-1.24867452)}

    init_conds = {'G_CYTO_INIT': 0,
                  'H_CYTO_INIT': 0.,
                  'P_CYTO_INIT': 0,
                  'G_EXT_INIT': gly_cond,
                  'H_EXT_INIT': 0.,
                  'P_EXT_INIT': 0,
                  }
    dhaB_dhaT_model = DhaBDhaTModelAlt(external_volume=external_volume)
    mintime = 10 ** (-15)
    fintime = (2) * 60 * 60

    #################################################
    # Integrate with BDF
    #################################################

    # initial conditions
    n_compounds_cell = 3
    y0 = np.zeros(dhaB_dhaT_model.nvars)
    for i, init_names in enumerate(VARIABLE_INIT_NAMES):
        if i < 6:
            y0[i] = init_conds[init_names]
    tol = 1e-8
    dhaB_dhaT_model._set_symbolic_sderiv_conc_fun()
    time_concat = [0]
    sol_concat = np.array([y0])
    for i in range(len(cell_per_metres_cubed_discrete)-1):
        params["ncells"] = cell_per_metres_cubed_discrete[i]*DCW_TO_COUNT_CONC
        ds = lambda t, x: dhaB_dhaT_model._sderiv(t, x, params)
        ds_jac = lambda t, x: dhaB_dhaT_model._sderiv_jac_conc_fun(t, x, params)
        sol = solve_ivp(ds, [t[i]*HRS_TO_SECS,t[i + 1]*HRS_TO_SECS], y0, method="BDF", jac=ds_jac,
                        t_eval=np.linspace(t[i]*HRS_TO_SECS,t[i + 1]*HRS_TO_SECS,num=10),atol=1e-7, rtol=1e-7)
        time_concat = np.concatenate((time_concat,sol.t))
        sol_concat = np.concatenate((sol_concat,sol.y.T))
        y0 = sol.y[:,-1]
        ratio = cell_per_metres_cubed_discrete[i]/cell_per_metres_cubed_discrete[i+1]
        #ratio = 1
        y0[0:3] = y0[0:3]*ratio


    # ds = lambda t, x: dhaB_dhaT_model._sderiv(t, x, params)
    # ds_jac = lambda t, x: dhaB_dhaT_model._sderiv_jac_conc_fun(t, x, params)
    # sol = solve_ivp(ds, [xs.iloc[0] * HRS_TO_SECS, xs.iloc[-1] * HRS_TO_SECS], y0, method="BDF", jac=ds_jac,
    #                 atol=1e-1, rtol=1e-1)
    # sol_concat = sol.y.T
    # time_concat = sol.t
    #################################################
    # Plot solution
    #################################################
    volcell = dhaB_dhaT_model.cell_volume
    colour = ['b', 'r', 'y', 'c', 'm']

    # rescale the solutions
    timeorighours = sol.t / HRS_TO_SECS

    # external solution
    for i in range(0, 3):
        yext = sol_concat[:,3 + i]
        plt.plot(time_concat, yext, colour[i])

    plt.legend(['Glycerol', '3-HPA', '1,3-PDO'], loc='upper right')
    plt.title('Plot of external concentration')
    plt.xlabel('time (hr)')
    plt.ylabel('concentration (mM)')
    plt.show()

    # cell solutions
    for i in range(3):
        ycell = sol_concat[:,i]
        plt.plot(time_concat, ycell, colour[i])

    plt.legend(['Glycerol', '3-HPA', '1,3-PDO'], loc='upper right')
    plt.title('Plot of cytosol concentrations')
    plt.xlabel('time (hr)')
    plt.ylabel('concentration (mM)')
    plt.show()

    # check mass balance
    y0 = np.zeros(dhaB_dhaT_model.nvars)
    for i, init_names in enumerate(VARIABLE_INIT_NAMES):
        if i < 6:
            y0[i] = init_conds[init_names]
    ext_masses_org = y0[3:6] * external_volume
    cell_masses_org = y0[:3] * volcell
    ext_masses_fin = sol.y[3:6, -1] * external_volume
    cell_masses_fin = sol.y[:3, -1] * volcell
    print(ext_masses_org.sum() + external_volume * y0[-1] * cell_masses_org.sum())
    print(ext_masses_fin.sum() + external_volume * sol.y[-1, -1] * cell_masses_fin.sum())

if __name__ == '__main__':
    test_devs()
    #test_QoI()
