import pandas as pd
import matplotlib.pyplot as plt 
from scipy import interpolate
import numpy as np
from csaps import csaps

N = 10**4
M = 1000
smoothing_vec= np.linspace(0.5,0.99995,M)

# get and store PDO energy points
PDOEnergyPoints = pd.read_csv('PDOEnergyPoints3.csv', names=['z','PDO'])
PDOEnergyPoints.sort_values(by='z',inplace=True)
PDOEnergyPoints.drop_duplicates(subset ="z", inplace = True) 
PDO_z_max = PDOEnergyPoints.loc[:,'z'].max()
PDO_z_min = PDOEnergyPoints.loc[:,'z'].min()
z_PDO = np.linspace(PDO_z_min, PDO_z_max, num = N)
h_PDO = (PDO_z_max - PDO_z_min)/(N-1)

# get and store PPN energy points
PPNEnergyPoints = pd.read_csv('PPNEnergyPoints2.csv', names=['z','PPN'])
PPNEnergyPoints.sort_values(by='z',inplace=True)
PPN_z_max = PPNEnergyPoints.loc[:,'z'].max()
PPN_z_min = PPNEnergyPoints.loc[:,'z'].min()
z_PPN = np.linspace(PPN_z_min, PPN_z_max, num = N)
h_PPN = (PPN_z_max - PPN_z_min)/(N-1)

#storing array
PDO_eng_int = []
PPN_eng_int = []

for s in smoothing_vec:

	PDO_cs = csaps(PDOEnergyPoints.loc[:,'z'], PDOEnergyPoints.loc[:,'PDO'], z_PDO, smooth=s)
	PDO_ind_max = np.argmax(PDO_cs)

	PPN_cs = csaps(PPNEnergyPoints.loc[:,'z'], PPNEnergyPoints.loc[:,'PPN'], z_PPN, smooth=s)
	PPN_ind_max = np.argmax(PPN_cs)

	# calculate devs of the PDO curves
	PDO_cs_max = PDO_cs[PDO_ind_max]
	PDO_cs_sec_dev = (PDO_cs[PDO_ind_max+1]-2*PDO_cs[PDO_ind_max]+PDO_cs[PDO_ind_max-1])/(h_PDO)**2 

	# calculate devs of the PPN curves
	PPN_cs_max = PPN_cs[PPN_ind_max]
	PPN_cs_sec_dev = (PPN_cs[PPN_ind_max+1]-2*PPN_cs[PPN_ind_max]+PPN_cs[PPN_ind_max-1])/(h_PPN)**2

	kBT = 293*0.001985
	PDO_eng_int.append(np.exp(PDO_cs_max/kBT)*np.sqrt(-2*np.pi*kBT/PDO_cs_sec_dev))
	PPN_eng_int.append(np.exp(PPN_cs_max/kBT)*np.sqrt(-2*np.pi*kBT/PPN_cs_sec_dev))

plt.plot(smoothing_vec,PDO_eng_int)
plt.title('PDO')
plt.show()
plt.plot(smoothing_vec,PPN_eng_int)
plt.title('PPN')
plt.show()
print('PDO: max ' + str(max(PDO_eng_int)) + ' and min ' + str(min(PDO_eng_int)))
print('PPN: max ' + str(max(PPN_eng_int)) + ' and min ' + str(min(PPN_eng_int)))