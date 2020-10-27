This code contains a deprecated DhaB12-DhaT-IcdE reaction model of PdU engineered system. Only the DhaB and DhaT reaction are currently being modelled. This model is deprecated and now longer maintained. 

The IcdE reaction will be added later to the newer model located in DhaB_DhaT_Model.

This model is no longer in use as:
* DhaB12 reaction contains an activation step. This activationstep has not been well studied and is now excluded from current models.
* Not much is known just yet on how the experimentals on including the third reaction. 
* Of three NADH/NAD+ reactions, IcdE reaction most favours the substrates than the products at steady state. IcdE cycling reaction is not recommended for use. 
* Assumptions used to constrain the DhaB and DhaT reaction so that they are forward reactions.
* kcat are erroneous labelled as V_max.



Model and Model Analysis functions:
Whole_Cell_Engineered_System_IcdE.py
	* The DhaB12-DhaT-IcdE model contains DhaB-DhaT-IcdE reaction
	in the MCP; diffusion in the cell; diffusion from the cell 
	in the external volume.
Whole_Cell_Engineered_System_IcdE_LocalSensitivity_Analysis.py
	* Local Sensitivity Analysis of IcdE Model with functions.
Active_Subspaces.py
	* This code generates the average parameter directions that most affects 
	the model in a bounded region of parameter space.
Active_Subspaces_Parallel.py
	* Parallelizes the Active_Subspaces.py code. This code generates the
	average parameter directions that most affects the model in a bounded region
	of parameter space.
	
Model plotting functions:
steady_state_analysis.py
	* Given DhaB-DhaT-IcdE model, this computes dx/dt = 0 of the
	reaction system in the microcompartment.
plot_steady_state_param.py
	* Generates plots of the steady state of the DhaB12-DhaT-IcdE model
	as at most 2 parameters are varied.
plot_specific_parameter_set.py 
	* Generates plots of the time varying state of concentrations
	in the DhaB12-DhaT-IcdE model. Separate plots are generated
	for external, cellular and MCP concentrations.
plot_space_solution_parameter_set.py
	* Generates plots of the time varying state of concentrations
	in the DhaB12-DhaT-IcdE model. 

Scripts: 
script_plot_doe.py
	* This script generates plots for DoE report submitted in June/July.
script_comparing_grid.py
	* This script generates plots comparing the solutions as the grid 
	is varied.
		
Old functions/scripts:
Whole_Cell_Engineered_System_IcdE_Parallelized.py
	* Parallelization of the DhaB-DhaT-IcdE Model.
Deprecated_Whole_Cell_Engineered_System_LocalSensitivity_Analysis.py 
	* Local Sensitivity Analysis of IcdE Model without functions
