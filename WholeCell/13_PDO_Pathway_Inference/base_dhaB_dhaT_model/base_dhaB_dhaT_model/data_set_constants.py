"""
Data set constants

Programme written by aarcher07
Editing History:
- 1/3/21
"""

import numpy as np
import pandas as pd
import sys
from .model_constants import QoI_PARAMETER_LIST
from os.path import dirname, abspath
ROOT_PATH = dirname(dirname(dirname(abspath(__file__))))

INIT_CONDS_GLY_PDO_DCW = {50:  [48.4239274209863, 0.861642364731331,0.060301507537688],
                         60: [57.3451166180758, 1.22448979591837, 0.100696991929568],
                         70: [72.2779071192256, 1.49001347874035, 0.057971014492754],
                         80: [80.9160305343512, 1.52671755725191, 0.07949305141638]} # QoI at time 0 for each experiment

time_series_df = pd.read_csv(ROOT_PATH + "/data/data_time_series.csv")
TIME_EVALS = time_series_df["Time"].to_numpy()
TIME_EVALS = np.sort(np.unique(TIME_EVALS)) # unique time evaluations
DATA_COLUMNS = [3,5,6] # indices of QoI in the differential equation
EXTERNAL_VOLUME = 0.002 # external volume from experiment
NPARAMS = len(QoI_PARAMETER_LIST) + len(INIT_CONDS_GLY_PDO_DCW[50]) # number of parameters (including initial conditions)

TIME_SAMPLES = {} # dictionary of time samples for each initial glycerol concentration experiment
DATA_SAMPLES = {} # dictionary of data collected for each initial glycerol concentration experiment

for gly_cond in [50,60,70,80]:
	rows_bool = time_series_df.loc[:,"Glycerol Init"] == gly_cond
	TIME_SAMPLES[gly_cond] = time_series_df.loc[rows_bool,"Time"]
	DATA_SAMPLES[gly_cond] = time_series_df[["Glycerol","PDO","DCW"]][rows_bool].to_numpy()

