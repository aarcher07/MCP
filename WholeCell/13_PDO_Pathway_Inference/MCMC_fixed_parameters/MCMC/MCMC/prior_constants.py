"""
Constants parameters 

Programme written by aarcher07
Editing History:
- 1/3/21
"""

import numpy as np
import pandas as pd
import scipy.stats as stats


LOG_UNIF_PRIOR_PARAMETERS = {'scalar': [-1., 1.],
                        'cellperGlyMass': np.log10([1e4, 1e12]),
                        'PermCellGlycerol': np.log10([1e-8, 1e-2]), 
                        'PermCellPDO': np.log10([1e-6, 1e-2]), 
                        'PermCell3HPA': np.log10([1e-3, 1e-2]),
                        'VmaxfDhaB': np.log10([5e0, 5e3]), 
                        'KmDhaBG': np.log10([1e-1 , 1e1]),
                             # 'KmDhaBH': np.log10([1e-1 , 1e2]),
                        'VmaxfDhaT': np.log10([5e-1,5e2]),
                        'KmDhaTH': np.log10([1e-2 , 1e2]),
                        'VmaxfGlpK': np.log10([1e-2,1e3]),
                             'KmGlpKG': np.log10([1e-3,1e-1])}

LOG_NORM_PRIOR_PARAMETERS = {'scalar': [np.log10(0.5), (1 / 8.) ** 2],
                            'cellperGlyMass': [8, 2**2],
                            'PermCellGlycerol': [-5,(3/2.)**2],
                            'PermCellPDO': [-4,1**2],
                            'PermCell3HPA': [-2.5,(1/4.)**2],
                            'VmaxfDhaB':[1.5,(3/2)**2],
                            'KmDhaBG': [0, (1/2)**2],
                                 # 'KmDhaBH': [1/2., 2**2],
                            'VmaxfDhaT':[0,(3/2)**2],
                            'KmDhaTH': [0,(1/2)**2],
                            'VmaxfGlpK':[0.5,(3/2)**2],
                            'KmGlpKG': [-2,(1/2)**2]}
