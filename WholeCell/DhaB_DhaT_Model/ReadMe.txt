This code contains a deprecated DhaB-DhaT reaction model of PdU engineered system. This model is deprecated and no longer maintained.  Only the DhaB and DhaT reaction are currently being modelled within M well-mixed, identical cells within the cyostol. 

This model is currently in use. The DhaB-DhaT model assumes that there  are M identical MCPs within the cytosol and N identical cells within the external volume. From time scsle analysis, gradients in cell are removed.

Model and Model Analysis modules:
DhaB_DhaT_Model.py
	* The DhaB12-DhaT model contains DhaB-DhaT-IcdE reaction
	in the MCP; diffusion in the cell; diffusion from the cell 
	in the external volume.
DhaB_DhaT_Model_LocalSensAnalysis.py
	* Local Sensitivity Analysis of DhaB-DhaT Model with functions.
	This module gives the user control over the parameters for 
	which they would like to do sensitivity analysis.


Scripts: 
DhaB_DhaT_Model_LocalSensAnalysis_Plots.py
	* This scripts generates the local sensitivity analysis plots associated
	with DhaB_DhaT_Model_LocalSensAnalysis.py.

Deprecated_Ratio_Simulations_Parallel.py
	* This scipts generates a dataset of toxicity, production and consumption time
	for DhaB to DhaT enzyme concentration ratios within the MCP.
