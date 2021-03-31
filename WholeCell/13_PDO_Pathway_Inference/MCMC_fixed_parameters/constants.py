"""
Constants parameters 

Programme written by aarcher07
Editing History:
- 1/3/21
"""

import numpy as np
import pandas as pd
HRS_TO_SECS = 60*60
DCW_TO_COUNT_CONC = 3.2e9/1e-3


PARAMETER_LIST = ['cellperGlyMass',
                  'PermCellGlycerol','PermCellPDO','PermCell3HPA',
                  'VmaxfDhaB', 'KmDhaBG', #'KmDhaBH',
                  'VmaxfDhaT','KmDhaTH',
                  'VmaxfGlpK','KmGlpKG']

VARIABLE_INIT_NAMES = ['G_CYTO_INIT', 'H_CYTO_INIT','P_CYTO_INIT',
                       'G_EXT_INIT', 'H_EXT_INIT','P_EXT_INIT',
                       'CELL_CONC_INIT']


VARS_TO_TEX = {'scalar': r'$\alpha$',
                'cellperGlyMass' : r'$\gamma$',
                'PermCellGlycerol':r'$P_G$',
                'PermCellPDO':r'$P_P$',
                'PermCell3HPA':r'$P_H$',
                'VmaxfDhaB': r'$V_{\text{max},\text{dhaB}}$',
                'KmDhaBG': r'$K_{\text{M},\text{dhaB}}^G$',
                # 'KmDhaBH': r'$K_{\text{M},\text{dhaB}}^H$',
                'VmaxfDhaT': r'$V_{\text{max},\text{dhaT}}$',
                'KmDhaTH': r'$K_{\text{M},\text{dhaT}}^H$',
                'VmaxfGlpK': r'$V_{\text{max},\text{glpK}}$',
                'KmGlpKG': r'$K_{\text{M},\text{glpK}}^G$',
                }

VARS_TO_UNITS = {'scalar': '',
                'cellperGlyMass': '',
                'PermCellGlycerol':'m/s',
                'PermCellPDO':'m/s',
                'PermCell3HPA':'m/s',
                'VmaxfDhaB': 'mM/s',
                'KmDhaBG': 'mM',
                # 'KmDhaBH': 'mM',
                'VmaxfDhaT': 'mM/s',
                'KmDhaTH': 'mM',
                'VmaxfGlpK': 'mM/s',
                'KmGlpKG': 'mM'}

param_sens_log_unif_bounds = {'scalar': [-1, 1],
                        'cellperGlyMass': np.log10([1e4, 1e12]),
                        'PermCellGlycerol': np.log10([1e-8, 1e-2]), 
                        'PermCellPDO': np.log10([1e-6, 1e-2]), 
                        'PermCell3HPA': np.log10([1e-3, 1e-2]),
                        'VmaxfDhaB': np.log10([1e0, 1e3]), 
                        'KmDhaBG': np.log10([1e-1 , 1e1]),
                        # 'KmDhaBH': np.log10([1e-1 , 1e2]),
                        'VmaxfDhaT': np.log10([1e-2,1e2]),
                        'VmaxfGlpK': np.log10([1e-2,1e3]),
                        'KmGlpKG': np.log10([1e-3,1e-1])}

param_sens_log_norm_bounds = {'scalar': [np.log10(0.5), (1/8)**2],
                        'cellperGlyMass': [8, 2**2],
                        'PermCellGlycerol': [-5,(3/2)**2], 
                        'PermCellPDO': [-4,1**2], 
                        'PermCell3HPA': [-2.5,(1/4)**2],
                        'VmaxfDhaB':[1.5,(3/2)**2], 
                        'KmDhaBG': [0, (1/2)**2],
                        # 'KmDhaBH': [1/2., 2**2],
                        'VmaxfDhaT':[0,(3/2)**2],
                        'KmDhaTH': [0,(1/2)**2],
                        'VmaxfGlpK':[0.5,(3/2)**2],
                        'KmGlpKG': [-2,(1/2)**2]}

param_sens_bounds = {'scalar': [1e-1, 1e1],
                     'cellperGlyMass': [1e4, 1e12],
                     'PermCellGlycerol': [1e-8, 1e-2], 
                     'PermCellPDO': [1e-6, 1e-2], 
                     'PermCell3HPA': [1e-3, 1e-2],
                     'VmaxfDhaB': [1e0, 1e3], 
                     'KmDhaBG': [1e-1 , 1e1],
                     'KmDhaBH': [1e-1 , 1e2],
                     'VmaxfDhaT': [1e-2,1e2],
                     'KmDhaTH': [1e-1, 1e1],
                     'VmaxfGlpK':[1e-2,1e3],
                     'KmGlpKG': [1e-3,1e-1]}

INIT_CONDS_GLYPDODCW = {50:  [48.4239274209863, 0.861642364731331,0.060301507537688],
                        60: [57.3451166180758, 1.22448979591837, 0.100696991929568],
                        70: [72.2779071192256, 1.49001347874035, 0.057971014492754],
                        80: [80.9160305343512, 1.52671755725191, 0.07949305141638]}

TIME_EVALS = pd.read_csv("data_time_series_cleaned.csv")["Time"].to_numpy()
TIME_EVALS = np.sort(np.unique(TIME_EVALS))
time_series_df = pd.read_csv("data_time_series_cleaned.csv")[["Glycerol Init", "Time"]]
TIME_EVALS_BOOL = [] 

for gly_cond in INIT_CONDS_GLYPDODCW.keys():
    rows_bool = time_series_df.loc[:,"Glycerol Init"] == gly_cond
    times_samples = time_series_df.loc[rows_bool,"Time"]
    for time in TIME_EVALS:
        TIME_EVALS_BOOL.append(np.any(times_samples == time))


EXTERNAL_VOLUME = 0.002
nparams = len(PARAMETER_LIST) + len(INIT_CONDS_GLYPDODCW[50])