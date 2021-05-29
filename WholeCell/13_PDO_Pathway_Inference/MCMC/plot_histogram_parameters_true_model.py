import os
from base_dhaB_dhaT_model.misc_functions import load_obj
from base_dhaB_dhaT_model.model_constants import QoI_PARAMETER_LIST,VARS_TO_TEX,DCW_TO_COUNT_CONC
from base_dhaB_dhaT_model.data_set_constants import INIT_CONDS_GLY_PDO_DCW,TIME_EVALS,DATA_SAMPLES,TIME_SAMPLES
from MCMC.dhaB_dhaT_model_prior import DhaBDhaTModelMCMC
from MCMC.dhaB_dhaT_model_bayesian_funs import loglik
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}'] #for \text command

directory = "/home/aarcher/Dropbox/PycharmProjects/MCP/WholeCell/13_PDO_Pathway_Inference/MCMC/output/MCMC_results_data_quest/MCMC_results_data/adaptive/sigma_[2,2,0,2]/lambda_0,01_beta_0,01/nsamples_100000/norm"
burn_in_n = int(2e4)
data_dict = {QoI_name:[] for QoI_name in QoI_PARAMETER_LIST}
for filename in os.listdir(directory):
    data = load_obj(directory + "/" +filename[:-4])
    for i in range(data.shape[1]):
        data_dict[QoI_PARAMETER_LIST[i]].extend(data[range(burn_in_n,int(1e5),1000),i])

save_file_name = "adaptive/sigma_[2,2,0,2]/lambda_0,01_beta_0,01/nsamples_100000/norm/"
mean_params = []

for QoI_name in QoI_PARAMETER_LIST:
    sns.histplot(data=data_dict[QoI_name],stat='probability', bins='auto', color='#0504aa',alpha=0.7)
    plt.grid(axis='y', alpha=0.75)
    plt.xlabel(VARS_TO_TEX[QoI_name])
    plt.title('Histogram of MCMC walk of '+ VARS_TO_TEX[QoI_name])
    plt.axvline(x=np.mean(data_dict[QoI_name]), color='red',linewidth=4)
    mean_params.append(np.mean(data_dict[QoI_name]))
    plt.show()

dhaB_dhaT_model = DhaBDhaTModelMCMC(transform="log_norm")
mean_param_dict = {QoI_name:param for QoI_name,param in zip(QoI_PARAMETER_LIST,mean_params)}

for gly_cond in INIT_CONDS_GLY_PDO_DCW.keys():
    init_conds={'G_CYTO_INIT': 0,
                'H_CYTO_INIT': 0,
                'P_CYTO_INIT': 0,
                'G_EXT_INIT': INIT_CONDS_GLY_PDO_DCW[gly_cond][0],
                'H_EXT_INIT': INIT_CONDS_GLY_PDO_DCW[gly_cond][1],
                'P_EXT_INIT': 0,
                'CELL_CONC_INIT': INIT_CONDS_GLY_PDO_DCW[gly_cond][2]*0.5217871564671509*DCW_TO_COUNT_CONC
                 }


    qoi_vals = dhaB_dhaT_model.QoI(mean_param_dict,init_conds)

    plt.plot(TIME_EVALS,qoi_vals[:,0])
    plt.scatter(TIME_SAMPLES[gly_cond],DATA_SAMPLES[gly_cond][:,0])
    plt.title('Plot of external glycerol')
    plt.xlabel('time (hr)')
    plt.ylabel('concentration (mM)')
    plt.show()

    plt.plot(TIME_EVALS,qoi_vals[:,1])
    plt.scatter(TIME_SAMPLES[gly_cond],DATA_SAMPLES[gly_cond][:,1])
    plt.title('Plot of external 1,3-PDO')
    plt.xlabel('time (hr)')
    plt.ylabel('concentration (mM)')
    plt.show()

    plt.plot(TIME_EVALS,qoi_vals[:,2])
    plt.scatter(TIME_SAMPLES[gly_cond],DATA_SAMPLES[gly_cond][:,2])
    plt.title('Plot of dry cell weight')
    plt.xlabel('time (hr)')
    plt.ylabel('concentration (g per $m^3$)')
    plt.show()
