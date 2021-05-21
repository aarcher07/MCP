"""
Model constants

Programme written by aarcher07
Editing History:
- 1/3/21
"""

import numpy as np
import pandas as pd

HRS_TO_SECS = 60*60
DCW_TO_COUNT_CONC = 3.2e9/1e-3

MODEL_PARAMETER_LIST = ['cellperGlyMass',
                  'PermCellGlycerol','PermCellPDO','PermCell3HPA',
                  'VmaxfDhaB', 'KmDhaBG', #'KmDhaBH',
                  'VmaxfDhaT','KmDhaTH',
                  'VmaxfGlpK','KmGlpKG']

QoI_PARAMETER_LIST = ['scalar', 'cellperGlyMass',
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

