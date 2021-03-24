import pickle
import os

folder_name = os.path.abspath(os.getcwd())

print(folder_name)
with open(folder_name + '/data/nsamples_10_date_2021_03_16_14:10.pkl', 'rb') as f:
     x = pickle.load(f)
print(x)
