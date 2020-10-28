This code contains a deprecated DhaB-DhaT reaction model of PdU engineered system. 

This model is deprecated and no longer maintained.  Only the DhaB and DhaT reaction are currently being modelled within M well-mixed, identical cells within the cyostol. 

This model is no longer in use as:
* It assumes that a community of MCPs act as a single MCP.
* It assumes that a community of cells act as a single cell.

Model and Model Analysis modules:
DhaB_DhaT_Model_SingleMCP.py
	* The DhaB12-DhaT model contains DhaB-DhaT-IcdE reaction
	in the MCP; diffusion in the cell; diffusion from the cell 
	in the external volume.
DhaB_DhaT_Model_SingleMCP_Nondimensionalized.py
	* Non dimensionalized version of the model in DhaB_DhaT_Model_SingleMCP.py.

Scripts: 
Ratio_Simulations_WellMixed.py
	* This generates plot of the cellular, MCP and external concentrations
	using a model of the DhaB-DhaT pathway in the MCP of salmonella. 
Ratio_Simulations_WellMixed_Parallelized.py
	* This scipts generates a dataset of toxicity, production and consumption time
	for DhaB to DhaT enzyme concentration ratios within the MCP.
