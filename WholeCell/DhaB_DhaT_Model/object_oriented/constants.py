"""
Constants parameters 

Programme written by aarcher07
Editing History:
- 1/3/21
"""


HRS_TO_SECS = 60*60


PARAMETER_LIST = ['KmDhaTH', 'KmDhaTN','kcatfDhaT', 
                  'kcatfDhaB', 'KmDhaBG', 
                  'PermMCPPolar', 'NonPolarBias', 'PermCell', 
                  'dPacking', 
                  'nmcps',
                  'enz_ratio',
                  'NADH_MCP_INIT','NAD_MCP_INIT']

VARIABLE_INIT_NAMES = ['G_MCP_INIT','H_MCP_INIT','P_MCP_INIT',
                       'G_CYTO_INIT', 'H_CYTO_INIT','P_CYTO,INIT',
                       'G_EXT_INIT', 'H_EXT_INIT','P_EXT_INIT']


VARS_TO_TEX = {'kcatfDhaB': r'$k_{\text{cat}}^{f,\text{dhaB}}$',
                'KmDhaBG': r'$K_{\text{M}}^{\text{Glycerol},\text{dhaB}}$',
                'kcatfDhaT': r'$k_{\text{cat}}^{f,\text{dhaT}}$',
                'KmDhaTH': r'$K_{\text{M}}^{\text{3-HPA},\text{dhaT}}$',
                'KmDhaTN': r'$K_{\text{M}}^{\text{NADH},\text{dhaT}}$',
                'NADH_MCP_INIT': r'$[\text{NADH}]$ ',
                'NAD_MCP_INIT': r'$[\text{NAD+}]$ ',
                'PermMCPPolar':r'$P_{\text{MCP},\text{Polar}}$',
                'NonPolarBias':r'$\alpha_{\text{MCP},\text{Non-Polar}}$',
                'PermCell': r'$P_{\text{Cell}}$',
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
                'NonPolarBias':'',
                'PermCell': 'm/s',
                'dPacking': '', 
                'nmcps': ''}

param_sens_bounds = {'kcatfDhaB': [400, 860], # /seconds Input
                    'KmDhaBG': [0.6,1.1], # mM Input
                    'kcatfDhaT': [40.,100.], # /seconds
                    'KmDhaTH': [0.1,1.], # mM
                    'KmDhaTN': [0.0116,0.48], # mM
                    'NADH_MCP_INIT': [0.12,0.60],
                    'PermMCPPolar': np.log10([10**-4, 10**-2]),
                    'NonPolarBias': np.log10([10**-2, 10**-1]),
                    'PermCell': np.log10([10**-9,10**-4]),
                    'dPacking': [0.3,0.64],
                    'nmcps': [3.,30.]}