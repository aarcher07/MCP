"""
Constants parameters 

Programme written by aarcher07
Editing History:
- 1/3/21
"""

import numpy as np
import pandas as pd
import scipy.stats as stats

HRS_TO_SECS = 60*60
DCW_TO_COUNT_CONC = 3.2e9/1e-3


MODEL_PARAMETER_LIST = ['cellperGlyMass',
                  'PermCellGlycerol','PermCellPDO','PermCell3HPA',
                  'VmaxfDhaB', 'KmDhaBG', #'KmDhaBH',
                  'VmaxfDhaT','KmDhaTH',
                  'VmaxfGlpK','KmGlpKG']

QoI_PARAMETER_LIST = ['cellperGlyMass', 'scalar',
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

# LOG NORM PARAMETERS 95\% BOUNDS

LOG_PARAMETERS_BOUNDS = {'scalar': [-1., 1.],
                          'cellperGlyMass': np.log10([1e4, 1e12]),
                          'PermCellGlycerol': np.log10([1e-8, 1e-2]),
                          'PermCellPDO': np.log10([1e-6, 1e-2]),
                          'PermCell3HPA': np.log10([1e-4, 1e-2]),
                          'VmaxfDhaB': np.log10([1e-3, 1e6]),
                          'KmDhaBG': np.log10([1e-4 , 1e2]),
                          'VmaxfDhaT': np.log10([1e-3,1e6]),
                          'KmDhaTH': np.log10([1e-4 , 1e2]),
                          'VmaxfGlpK': np.log10([1e-3,1e6]),
                          'KmGlpKG': np.log10([1e-4,1e2])}

PARAMETER_BOUNDS = {key: (10.**np.array(val)).tolist() for key, val in LOG_PARAMETERS_BOUNDS.items()}

INIT_CONDS_GLY_PDO_DCW = {50:  [48.4239274209863, 0.861642364731331,0.060301507537688],
                         60: [57.3451166180758, 1.22448979591837, 0.100696991929568],
                         70: [72.2779071192256, 1.49001347874035, 0.057971014492754],
                         80: [80.9160305343512, 1.52671755725191, 0.07949305141638]}

TIME_EVALS = pd.read_csv("data_time_series_cleaned.csv")["Time"].to_numpy()
TIME_EVALS = np.sort(np.unique(TIME_EVALS))
DATA_COLUMNS = [3,5,6]

EXTERNAL_VOLUME = 0.002

NPARAMS = len(QoI_PARAMETER_LIST) + len(INIT_CONDS_GLY_PDO_DCW[50])
