"""
Constants parameters 

Programme written by aarcher07
Editing History:
- 1/3/21
"""

import numpy as np
import pandas as pd
import scipy.stats as stats


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
