"""
Constants parameters 

Programme written by aarcher07
Editing History:
- 1/3/21
"""

import numpy as np

HRS_TO_SECS = 60*60
OD_TO_COUNT_CONC = 3.2e9

MODEL_PARAMETER_LIST = ['PermMCPPropanediol','PermMCPPropionaldehyde','PermMCPPropanol','PermMCPPropionyl','PermMCPPropionate',
                        'nmcps',
                        'PermCellPropanediol', 'PermCellPropionaldehyde', 'PermCellPropanol', 'PermCellPropionyl','PermCellPropionate',
                        'VmaxCDEf', 'KmCDEPropanediol',
                        'VmaxPf', 'KmPfPropionaldehyde',
                        'VmaxPr', 'KmPrPropionyl',
                        'VmaxQf', 'KmQfPropionaldehyde',
                        'VmaxQr', 'KmQrPropanol',
                        'VmaxLf', 'KmLPropionyl',
                        'ncells']



VARIABLE_INIT_NAMES = ['PROPANEDIOL_MCP_INIT', 'PROPIONALDEHYDE_MCP_INIT', 'PROPANOL_MCP_INIT', 'PROPIONYL_MCP_INIT', 'PROPIONATE_MCP_INIT',
                       'PROPANEDIOL_CYTO_INIT', 'PROPIONALDEHYDE_CYTO_INIT', 'PROPANOL_CYTO_INIT', 'PROPIONYL_CYTO_INIT', 'PROPIONATE_CYTO_INIT',
                       'PROPANEDIOL_EXT_INIT', 'PROPIONALDEHYDE_EXT_INIT', 'PROPANOL_EXT_INIT', 'PROPIONYL_EXT_INIT', 'PROPIONATE_EXT_INIT']

