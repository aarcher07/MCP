"""
Constants parameters 

Programme written by aarcher07
Editing History:
- 1/3/21
"""

import numpy as np
HRS_TO_SECS = 60*60
DCW_TO_COUNT = 8e8

PARAMETER_LIST = ['maxGrowthRate','saturation_const',
                  'PermCellGlycerol','PermCellPDO','PermCell3HPA',
                  'DhaB1Conc', 'DhaTConc', 'GlpKConc',
                  'kcatfDhaB', 'KmDhaBG', 
                  'kcatfDhaT','KmDhaTH',
                  'kcatfGlpK','KmGlpKG',
                  'NADH',
                  'ATP',
                  'G_CYTO_INIT', 'H_CYTO_INIT','P_CYTO_INIT',
                  'G_EXT_INIT', 'H_EXT_INIT','P_EXT_INIT',
                  'CELL_CONC_INIT']

VARIABLE_INIT_NAMES = ['G_CYTO_INIT', 'H_CYTO_INIT','P_CYTO_INIT',
                       'G_EXT_INIT', 'H_EXT_INIT','P_EXT_INIT',
                       'CELL_CONC_INIT']


VARS_TO_TEX = {'maxGrowthRate':r'$r$',
                'saturation_const' : 'K',
                'PermCellGlycerol':r'$P_G$',
                'PermCellPDO':r'$P_P$',
                'PermCell3HPA':r'$P_H$',
                'DhaB1Conc': r'$[\text{dhaB}_1]$',
                'DhaTConc':r'$[\text{dhaT}]$',
                'GlpKConc': r'$[\text{glpK}]$',
                'kcatfDhaB': r'$k_{\text{cat},\text{dhaB}}$',
                'KmDhaBG': r'$K_{\text{M},\text{dhaB}}$',
                'kcatfDhaT': r'$k_{\text{cat},\text{dhaT}}$',
                'KmDhaTH': r'$K_{\text{M},\text{dhaT}}^H$',
                'kcatfGlpK': r'$k_{\text{cat},\text{glpK}}$',
                'KmGlpKG': r'$K_{\text{M},\text{glpK}}^G$',
                'NADH': r'$[\text{NADH}]$',
                'ATP': r'$[\text{ATP}]$'
                }

VARS_TO_UNITS = {'maxGrowthRate':'/Mm s',
                'saturation_const' : 'mM',
                'PermCellGlycerol':'m/s',
                'PermCellPDO':'m/s',
                'PermCell3HPA':'m/s',
                'DhaB1Conc': 'mM',
                'DhaTConc': 'mM',
                'GlpKConc': 'mM',
                'kcatfDhaB': '/s',
                'KmDhaBG': 'mM',
                'kcatfDhaT': '/s',
                'KmDhaTH': 'mM',
                'kcatfGlpK': '/s',
                'KmGlpKG': 'mM',
                'NADH': 'mM',
                'ATP': 'mM'}

param_sens_log_unif_bounds = {'maxGrowthRate': np.log10([10**6, 10**12]), 
                        'saturation_const': np.log10([0.01, 100]),
                        'PermCellGlycerol': np.log10([10**-8, 10**-2]), 
                        'PermCellPDO': np.log10([10**-6, 10**-2]), 
                        'PermCell3HPA': np.log10([10**-3, 10**-2]),
                        'DhaB1Conc': np.log10([10**-2, 10**0]),
                        'DhaTConc': np.log10([10**-2, 10**0]),
                        'GlpKConc': np.log10([10**-1, 10**1]), 
                        'kcatfDhaB': np.log10([10**2, 10**3]), 
                        'KmDhaBG': np.log10([0.1 , 10]),
                        'kcatfDhaT': np.log10([10,10**2]),
                        'KmDhaTH': np.log10([10**-1, 10.]),
                        'kcatfGlpK': np.log10([0.1,100]) ,
                        'KmGlpKG': np.log10([10**-3,10**-1]),
                        'NADH': [0.12,0.60],
                        'ATP': [1.,5.]}
param_sens_log_norm_bounds = {'maxGrowthRate': [-9,3], 
                        'saturation_const': [0,2],
                        'PermCellGlycerol': [-5,2], 
                        'PermCellPDO': [-5,2], 
                        'PermCell3HPA': [-2.5,2],
                        'DhaB1Conc': [1,2],
                        'DhaTConc': [1,2],
                        'GlpKConc': [1,2], 
                        'kcatfDhaB':[3,2], 
                        'KmDhaBG': [-2, 2],
                        'kcatfDhaT': [3,2],
                        'KmDhaTH': [-2,2],
                        'kcatfGlpK':[3,2],
                        'KmGlpKG': [-2,2],
                        'NADH': [np.log10(0.35),1/16.],
                        'ATP': [np.log10(3),1/16.]}

param_sens_bounds = {'maxGrowthRate': [10**6, 10**12], 
                     'saturation_const': [0.01, 100],
                     'PermCellGlycerol': [10**-8, 10**-2], 
                     'PermCellPDO': [10**-6, 10**-2], 
                     'PermCell3HPA': [10**-3, 10**-2],
                     'DhaB1Conc': [10**-2, 10**0],
                     'DhaTConc': [10**-2, 10**0],
                     'GlpKConc': [10**-1, 10**1], 
                     'kcatfDhaB':[10**2, 10**3], 
                     'KmDhaBG': [0.1 , 10],
                     'kcatfDhaT': [10,10**2],
                     'KmDhaTH': [10**-1, 10.],
                     'kcatfGlpK': [0.1,100],
                     'KmGlpKG': [10**-3,10**-1],
                     'NADH': [0.12,0.60],
                     'ATP': [1.,5.]}
