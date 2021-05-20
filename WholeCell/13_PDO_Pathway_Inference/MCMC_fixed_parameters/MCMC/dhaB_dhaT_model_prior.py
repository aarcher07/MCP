'''
The DhaB-DhaT model contains DhaB-DhaT reaction pathway
in the MCP; diffusion in the cell; diffusion from the cell 
in the external volume.

This model is currently in use. The DhaB-DhaT model assumes that there 
are M identical MCPs within the cytosol and N identical cells within the 
external volume. From time scsle analysis, gradients in cell are removed.

Programme written by aarcher07
Editing History:
- 28/10/20
'''
import sys
sys.path.insert(0, '.')
from base_dhaB_dhaT_model.dhaB_dhaT_model import DhaBDhaTModel
from base_dhaB_dhaT_model.misc_functions import *

class DhaBDhaTModelMCMC(DhaBDhaTModel):
    def __init__(self, rc = 0.375e-6, lc = 2.47e-6,
                 external_volume = 0.002, transform = ''):
        """
        Initializes parameters to be used numerial scheme
        :param rc: Radius of cell in metres
        :param lc: length of the cell in metres (needed if assuming cells are rods)
        :param external_volume: external volume containing cells in metres^3
        :param transform: transform of the parameters, log10, log10 and standardized, no transform
        """
        # Integration Parameters
        super().__init__(rc, lc,external_volume)
        self.transform_name = transform
        if transform == 'log_unif':
            self._sderiv = self._sderiv_log_unif
        elif transform == 'log_norm':
            self._sderiv = self._sderiv_log_norm
        elif transform == '':
            pass
        else:
            raise ValueError('Unknown transform')
        self._set_symbolic_sderiv_conc_fun()

    def _sderiv_log_unif(self,t,x,log_params):
        """
        Computes the spatial derivative of the system at time point, t, with the parameters
        [-1,1] transformed by transforming parameters into their original values in LOG_UNIF_PRIOR_PARAMETERS
        :param t: time
        :param x: state variables
        :param params_sens: [-1,1] transformed parameter list
        """
        if log_params is None:
            print("Please set the parameter values")
        params = transform_from_log_unif(log_params)

        return super()._sderiv(t,x,params)

    def _sderiv_log_norm(self,t,x,log_params):
        """
        Computes the spatial derivative of the system at time point, t, with the parameters
        [-1,1] transformed by transforming parameters into their original values in LOG_NORM_PRIOR_PARAMETERS
        :param t: time
        :param x: state variables
        :param params_sens:  transformed normal parameter list
        """
        if log_params is None:
            print("Please set the parameter values")
        params = transform_from_log_norm(log_params)

        return super()._sderiv(t,x,params)

