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

QOI_NAMES = ["maximum concentration of 3-HPA",
             "Glycerol concentration after 5 hours",
             "1,3-PDO concentration after 5 hours"]
             

FUNCS_TO_NAMES = {'maximum concentration of 3-HPA': r'$\max_{t}\text{3-HPA}(t;\vec{p})$',
                  'Glycerol concentration after 5 hours': r'$\text{Glycerol}(5\text{ hrs}; \vec{p})$',
                  '1,3-PDO concentration after 5 hours': r'$\text{1,3-PDO}(5\text{ hrs}; \vec{p})$'}
                  
FUNCS_TO_FILENAMES = {'maximum concentration of 3-HPA': 'max3HPA',
                     'Glycerol concentration after 5 hours': 'G5hrs',
                     '1,3-PDO concentration after 5 hours': 'P5hrs'}
