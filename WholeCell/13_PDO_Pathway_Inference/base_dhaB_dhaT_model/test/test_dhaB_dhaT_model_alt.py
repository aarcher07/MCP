import numpy as np
from scipy.integrate import BDF,solve_ivp
import matplotlib as mpl
#mpl.rcParams['text.usetex'] = True
#mpl.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'
import matplotlib.pyplot as plt
from base_dhaB_dhaT_model import *
from scipy.interpolate import UnivariateSpline
from csaps import csaps

def test_devs():
    gly_cond = 50
    spl = csaps(TIME_SAMPLES[gly_cond],DATA_SAMPLES[gly_cond][:,2], smooth = 0.6)
    xs = TIME_SAMPLES[gly_cond]

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

    #################################################
    # Integrate with BDF
    #################################################


    # run with true ratio
    dhaB_dhaT_model._set_symbolic_sderiv_conc_fun()


    all_sol_y = []
    all_time_t = []
    for j in range(2,6):
        # initial conditions
        y0 = np.zeros(dhaB_dhaT_model.nvars)
        for i, init_names in enumerate(VARIABLE_INIT_NAMES):
            if i < 6:
                y0[i] = init_conds[init_names]
        time_concat = [0]
        sol_concat = np.array([y0])

        #discretize time series cell data
        N = 10**j
        t = np.logspace(-12, np.log10(xs.iloc[-1]), num=N)
        Navg = 10

        time = []
        cell_per_metres_cubed_discrete = []

        for i in range(N - 1):
            tavg = np.linspace(t[i], t[i + 1], num=Navg)
            splmeansvals = np.mean(spl(tavg))
            time.append(t[i])
            cell_per_metres_cubed_discrete.append(splmeansvals)
        time.append(t[-1])
        # cell_per_metres_cubed_discrete.append(splmeansvals)
        # plt.step(time, cell_per_metres_cubed_discrete, where='post')
        # plt.legend(['data', 'spline', 'step approximation of split'], loc='upper right')
        # plt.savefig('/piggy/home/aga3723/Desktop/cell_growth_N_' + str(N), bbox_inches='tight')
        # plt.close()

        #solve at each discretized point forward in time
        for i in range(len(cell_per_metres_cubed_discrete)-1):
            params["ncells"] = cell_per_metres_cubed_discrete[i]*DCW_TO_COUNT_CONC
            ds = lambda t, x: dhaB_dhaT_model._sderiv(t, x, params)
            ds_jac = lambda t, x: dhaB_dhaT_model._sderiv_jac_conc_fun(t, x, params)
            sol = solve_ivp(ds, [time[i]*HRS_TO_SECS,time[i + 1]*HRS_TO_SECS], y0, method="BDF", jac=ds_jac,
                            t_eval=np.linspace(time[i]*HRS_TO_SECS,time[i + 1]*HRS_TO_SECS,num=10),atol=1e-7, rtol=1e-7)
            time_concat = np.concatenate((time_concat,sol.t))
            sol_concat = np.concatenate((sol_concat,sol.y.T))
            y0 = sol.y[:,-1]
            ratio = 1
            y0[0:3] = y0[0:3]*ratio
        all_sol_y.append(sol_concat)
        all_time_t.append(time_concat)


    #################################################
    # Plot solution
    #################################################
    volcell = dhaB_dhaT_model.cell_volume
    colour = ['b', 'r', 'y', 'c', 'm', 'g']

    # external solution
    for i in range(0, 3):
        for j in range(len(all_time_t)):
            sol_concat = all_sol_y[j]
            time_concat = all_time_t[j]
            yext = sol_concat[:,3 + i]
            plt.plot(time_concat, yext)

        plt.legend(['N=10^2', 'N=10^3', 'N=10^4', 'N=10^5'], loc='upper right')
        plt.title('Plot of ' + ['Glycerol', '3-HPA', '1,3-PDO'][i] + ' external concentration')
        plt.xlabel('time (hr)')
        plt.ylabel('concentration (mM)')
        plt.savefig('/piggy/home/aga3723/Desktop/external_solution_'+  ['Glycerol', '3-HPA', '1,3-PDO'][i] +'_ratio_1_all_N')
        plt.close()

    # cytosol solution
    for i in range(0, 3):
        for j in range(len(all_time_t)):
            sol_concat = all_sol_y[j]
            time_concat = all_time_t[j]
            yext = sol_concat[:,i]
            plt.plot(time_concat, yext)

        plt.legend(['N=10^2', 'N=10^3', 'N=10^4', 'N=10^5'], loc='upper right')
        plt.title('Plot of ' + ['Glycerol', '3-HPA', '1,3-PDO'][i] + ' cytosol concentration')
        plt.xlabel('time (hr)')
        plt.ylabel('concentration (mM)')
        plt.savefig('/piggy/home/aga3723/Desktop/cytosol_solution_'+  ['Glycerol', '3-HPA', '1,3-PDO'][i] +'_ratio_1_all_N')
        plt.close()

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
