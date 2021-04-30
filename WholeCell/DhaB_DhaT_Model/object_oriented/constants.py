"""
Constants parameters 

Programme written by aarcher07
Editing History:
- 1/3/21
"""

import numpy as np

HRS_TO_SECS = 60*60


PARAMETER_LIST = ['kcatfDhaT','KmDhaTH', 'KmDhaTN',
                  'kcatfDhaB', 'KmDhaBG', 
                  'PermMCPPolar', 'PermMCPNonPolar',
                  'PermCellGlycerol','PermCellPDO','PermCell3HPA',
                  'dPacking', 
                  'nmcps',
                  'enz_ratio',
                  'NADH_MCP_INIT','NAD_MCP_INIT']

PERM_PARAMETER_LIST = ['PermMCPPolar', 'PermMCPNonPolar',
                       'PermCellGlycerol','PermCellPDO','PermCell3HPA']

                       
VARIABLE_INIT_NAMES = ['G_MCP_INIT','H_MCP_INIT','P_MCP_INIT',
                       'G_CYTO_INIT', 'H_CYTO_INIT','P_CYTO_INIT',
                       'G_EXT_INIT', 'H_EXT_INIT','P_EXT_INIT']


VARS_TO_TEX = {'kcatfDhaB': r'$k_{\text{cat}}^{f,\text{dhaB}}$',
                'KmDhaBG': r'$K_{\text{M}}^{\text{Glycerol},\text{dhaB}}$',
                'kcatfDhaT': r'$k_{\text{cat}}^{f,\text{dhaT}}$',
                'KmDhaTH': r'$K_{\text{M}}^{\text{3-HPA},\text{dhaT}}$',
                'KmDhaTN': r'$K_{\text{M}}^{\text{NADH},\text{dhaT}}$',
                'NADH_MCP_INIT': r'$[\text{NADH}]$ ',
                'NAD_MCP_INIT': r'$[\text{NAD+}]$ ',
                'PermMCPPolar':r'$P_{\text{MCP},\text{Polar}}$',
                'PermMCPNonPolar':r'$$P_{\text{MCP},\text{Non-Polar}}$',
                'PermCellGlycerol':r'$P_{\text{Cell},G}$',
                'PermCellPDO':r'$P_{\text{Cell},P}$',
                'PermCell3HPA':r'$P_{\text{Cell},H}$',
                'dPacking': 'dPacking', 
                'nmcps': 'Number of MCPs'}

VARS_TO_UNITS = {'kcatfDhaB': '/s',
                'KmDhaBG': 'mM',
                'kcatfDhaT': '/s',
                'KmDhaTH': 'mM',
                'KmDhaTN': 'mM',
                'NADH_MCP_INIT': 'mM',
                'NAD_MCP_INIT': 'mM',
                'PermMCPPolar':'m/s',
                'PermMCPNonPolar':'m/s',
                'PermCellGlycerol':'m/s',
                'PermCellPDO':'m/s',
                'PermCell3HPA':'m/s',
                'dPacking': '', 
                'nmcps': ''}

PARAM_SENS_BOUNDS = {'kcatfDhaB': [400, 860], # /seconds Input
                     'KmDhaBG': [0.6,1.1], # mM Input
                     'kcatfDhaT': [40.,100.], # /seconds
                     'KmDhaTH': [0.1,1.], # mM
                     'KmDhaTN': [0.0116,0.48], # mM
                     'NADH_MCP_INIT': [0.12,0.60],
                     'PermMCPPolar': [10**-3, 10**-2],
                     'PermMCPNonPolar': [5*10**-3, 10**-1],
                     'PermCellGlycerol': [1e-8, 1e-2], 
                     'PermCellPDO': [1e-6, 1e-2], 
                     'PermCell3HPA': [1e-3, 1e-2],
                     'dPacking': [0.3,0.64],
                     'nmcps': [3.,30.]}


PARAM_SENS_MIXED_BOUNDS = {'kcatfDhaB': [400, 860], # /seconds Input
                            'KmDhaBG': [0.6,1.1], # mM Input
                            'kcatfDhaT': [40.,100.], # /seconds
                            'KmDhaTH': [0.1,1.], # mM
                            'KmDhaTN': [0.0116,0.48], # mM
                            'NADH_MCP_INIT': [0.12,0.60],
                            'PermMCPPolar': np.log10([10**-4, 10**-2]),
                            'PermMCPNonPolar': np.log10([10**-2, 10**-1]),
                            'PermCellGlycerol': np.log10([1e-8, 1e-2]), 
                            'PermCellPDO': np.log10([1e-6, 1e-2]), 
                            'PermCell3HPA': np.log10([1e-3, 1e-2]),
                            'dPacking': [0.3,0.64],
                            'nmcps': [3.,30.]}

PARAM_SENS_LOG2_BOUNDS = {'kcatfDhaB': np.log2([400, 860]), # /seconds Input
                            'KmDhaBG': np.log2([0.6,1.1]), # mM Input
                            'kcatfDhaT': np.log2([40.,100.]), # /seconds
                            'KmDhaTH': np.log2([0.1,1.]), # mM
                            'KmDhaTN': np.log2([0.0116,0.48]), # mM
                            'NADH_MCP_INIT': np.log2([0.12,0.60]),
                            'PermMCPPolar': np.log2([10**-4, 10**-2]),
                            'PermMCPNonPolar': np.log2([10**-2, 10**-1]),
                            'PermCellGlycerol': np.log2([1e-8, 1e-2]), 
                            'PermCellPDO': np.log2([1e-6, 1e-2]), 
                            'PermCell3HPA': np.log2([1e-3, 1e-2]),
                            'dPacking': np.log2([0.3,0.64]),
                            'nmcps': np.log2([3.,30.])
                            }

PARAM_SENS_LOG10_BOUNDS = {'kcatfDhaB': np.log10([400, 860]), # /seconds Input
                            'KmDhaBG': np.log10([0.6,1.1]), # mM Input
                            'kcatfDhaT': np.log10([40.,100.]), # /seconds
                            'KmDhaTH': np.log10([0.1,1.]), # mM
                            'KmDhaTN': np.log10([0.0116,0.48]), # mM
                            'NADH_MCP_INIT': np.log10([0.12,0.60]),
                            'PermMCPPolar': np.log10([10**-4, 10**-2]),
                            'PermMCPNonPolar': np.log10([10**-2, 10**-1]),
                            'PermCellGlycerol': np.log10([1e-8, 1e-2]), 
                            'PermCellPDO': np.log10([1e-6, 1e-2]), 
                            'PermCell3HPA': np.log10([1e-3, 1e-2]), 
                            'dPacking': np.log10([0.3,0.64]),
                            'nmcps': np.log10([3.,30.])}

QOI_NAMES = ["maximum concentration of 3-HPA",
             "Glycerol concentration after 5 hours",
             "1,3-PDO concentration after 5 hours"]