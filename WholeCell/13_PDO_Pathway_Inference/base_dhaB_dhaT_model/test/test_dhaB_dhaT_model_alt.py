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
    for gly_cond in [50, 60, 70, 80]:
        x=TIME_SAMPLES[gly_cond]
        y = DATA_SAMPLES[gly_cond][:,2]
        spl = csaps(TIME_SAMPLES[gly_cond],DATA_SAMPLES[gly_cond][:,2], smooth = 0.6)
        cell_per_metres_cubed = lambda t: spl(t)*DCW_TO_COUNT_CONC
        plt.plot(x, y, 'ro', ms=5)
        plt.plot(x, spl(x), 'g', lw=3)
        plt.show()

        external_volume = 0.002

        params = {'cellperGlyMass': 10 ** (5.73158464),
                  'PermCellGlycerol': 10 ** (-3.55285234),
                  'PermCellPDO': 10 ** (-3.85344833),
                  'PermCell3HPA': 10 ** (-2.34212333),
                  'VmaxfDhaB': 10 ** (3.26266939),
                  'KmDhaBG': 10 ** (0.71152905),
                  'VmaxfDhaT': 10 ** (2.85561206),
                  'KmDhaTH': 10 ** (0.69665821),
                  'VmaxfGlpK': 10 ** (3),
                  'KmGlpKG': 10 ** (-1.24867452)}

        init_conds = {'G_CYTO_INIT': gly_cond,
                      'H_CYTO_INIT': 0.,
                      'P_CYTO_INIT': 0,
                      'G_EXT_INIT': gly_cond,
                      'H_EXT_INIT': 0.,
                      'P_EXT_INIT': 0,
                      }
        dt = 0.01
        dhaB_dhaT_model = DhaBDhaTModelAlt(cell_per_metres_cubed,dt = dt,external_volume=external_volume)

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
        nsamples = 500
        timeorig = np.logspace(np.log10(mintime), np.log10(fintime), nsamples)
        dhaB_dhaT_model._set_symbolic_sderiv_conc_fun()
        ds = lambda t, x: dhaB_dhaT_model._sderiv(t, x, params)
        ds_jac = lambda t, x: dhaB_dhaT_model._sderiv_jac_conc_fun(t, x, params)
        sol = BDF(ds, 0, y0,fintime + 1, jac=ds_jac, t_eval=timeorig,first_step=0.1,
                        atol=1e-4, rtol=1e-4)
        y = []
        t = []
        while(sol.status == "running"):
            print(sol.step_size)
            sol.step()
            print(sol.step_size)
            y.append(sol.y)
            t.append(sol.t)
            break
        y = np.array(y).T
        print(y.shape)
        print(sol.status)

        #################################################
        # Plot solution
        #################################################
        volcell = dhaB_dhaT_model.cell_volume
        colour = ['b', 'r', 'y', 'c', 'm']

        # rescale the solutions
        timeorighours = sol.t / HRS_TO_SECS

        # external solution
        for i in range(0, 3):
            yext = y[3 + i, :].T
            plt.plot(t, yext, colour[i])

        plt.legend(['Glycerol', '3-HPA', '1,3-PDO'], loc='upper right')
        plt.title('Plot of external concentration')
        plt.xlabel('time (hr)')
        plt.ylabel('concentration (mM)')
        plt.show()

        # cell solutions
        for i in range(3):
            ymcp = y[i, :].T
            plt.plot(t, ymcp, colour[i])

        plt.legend(['Glycerol', '3-HPA', '1,3-PDO'], loc='upper right')
        plt.title('Plot of cytosol concentrations')
        plt.xlabel('time (hr)')
        plt.ylabel('concentration (mM)')
        plt.show()

        plt.plot(timeorighours, sol.y[-1, :].T / ((10 ** -0.3531) * DCW_TO_COUNT_CONC), colour[i])
        plt.title('Plot of cell concentration')
        plt.xlabel('time (hr)')
        plt.ylabel('concentration (cell per $m^3$)')
        plt.show()

        # check mass balance
        ext_masses_org = y0[3:6] * external_volume
        cell_masses_org = y0[:3] * volcell
        ext_masses_fin = sol.y[3:6, -1] * external_volume
        cell_masses_fin = sol.y[:3, -1] * volcell
        print(ext_masses_org.sum() + external_volume * y0[-1] * cell_masses_org.sum())
        print(ext_masses_fin.sum() + external_volume * sol.y[-1, -1] * cell_masses_fin.sum())

if __name__ == '__main__':
    test_devs()
    #test_QoI()
