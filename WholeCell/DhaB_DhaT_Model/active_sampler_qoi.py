from reduced_qoi import ReducedQoI
import pickle
from constants import PARAM_SENS_LOG10_BOUNDS, HRS_TO_SECS

directory = '/home/aarcher/Dropbox/PycharmProjects/MCP/WholeCell/DhaB_DhaT_Model/data/1:18/'
filename = 'log10/2021_06_29_10:00/'
data_name_pkl = 'sampling_rsampling_N_100000'
active_samples_pkl = 'active_coords_maximin_sampling_rsampling_N_1000'

with open(directory + filename + data_name_pkl + '.pkl', 'rb') as f:
    data_pkl_as = pickle.load(f)

with open(directory + filename + active_samples_pkl + '.pkl', 'rb') as f:
    active_samples_as = pickle.load(f)

cost_matrices = data_pkl_as["FUNCTION_RESULTS"]["FINAL_COST_MATRIX"]
eig_max, active_coordinates = active_samples_as

n_inactive_samples = 1e1
transform = 'log10'

start_time = (10 ** (-15))
final_time = 72 * HRS_TO_SECS
integration_tol = 1e-4
tolsolve = 1e-5
nintegration_samples = 500
enz_ratio_name = "1:18"
enz_ratio_name_split = enz_ratio_name.split(":")
enz_ratio = float(enz_ratio_name_split[0]) / float(enz_ratio_name_split[1])

params_values_fixed = {'NAD_MCP_INIT': 0.1,
                       'enz_ratio': enz_ratio,
                       'G_MCP_INIT': 0,
                       'H_MCP_INIT': 0,
                       'P_MCP_INIT': 0,
                       'G_CYTO_INIT': 0,
                       'H_CYTO_INIT': 0,
                       'P_CYTO_INIT': 0,
                       'G_EXT_INIT': 200,
                       'H_EXT_INIT': 0,
                       'P_EXT_INIT': 0}

red_qoi = ReducedQoI(cost_matrices, n_inactive_samples,
                     start_time, final_time, integration_tol, nintegration_samples, tolsolve,
                     params_values_fixed, list(PARAM_SENS_LOG10_BOUNDS.keys()),
                     transform="log10")

for QoI_keys in active_coordinates.keys():
    for val in active_coordinates[QoI_keys]:
        active_params = {QoI_keys: val}
        print(active_params)
        print(red_qoi.generate_reduced_QoI_vals(active_params, gen_histogram=True))
