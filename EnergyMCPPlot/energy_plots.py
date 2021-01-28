import pandas as pd
import matplotlib.pyplot as plt 
from scipy import interpolate
import numpy as np
from csaps import csaps

N = 10**4

# get and store PDO energy points
PDOEnergyPoints = pd.read_csv('PDOEnergyPoints3.csv', names=['z','PDO'])
PDOEnergyPoints.sort_values(by='z',inplace=True)
PDOEnergyPoints.drop_duplicates(subset ="z", inplace = True) 

PDO_z_max = PDOEnergyPoints.loc[:,'z'].max()
PDO_z_min = PDOEnergyPoints.loc[:,'z'].min()
z_PDO = np.linspace(PDO_z_min, PDO_z_max, num = N)
PDO_cs = csaps(PDOEnergyPoints.loc[:,'z'], PDOEnergyPoints.loc[:,'PDO'], z_PDO, smooth=0.5)
h_PDO = (PDO_z_max - PDO_z_min)/(N-1)

# calculate devs of the PDO curves
start = 2
z_PDO_dev = z_PDO[start:-start]
PDO_cs_first_dev = [(PDO_cs[i+1]-PDO_cs[i-1])/(2*h_PDO) for i in range(start,N-start)]
PDO_cs_sec_dev = [(PDO_cs[i+1]-2*PDO_cs[i]+PDO_cs[i-1])/(h_PDO)**2 for i in range(start,N-start)]
PDO_cs_third_dev = [(PDO_cs[i+2]-2*PDO_cs[i+1]+2*PDO_cs[i-1]-PDO_cs[i-2])/(2*(h_PDO)**3) for i in range(start,N-start)]
PDO_cs_fourth_dev = [(PDO_cs[i+2]-4*PDO_cs[i+1]+6*PDO_cs[i]-4*PDO_cs[i-1]+PDO_cs[i-2])/(h_PDO)**4 for i in range(start,N-start)]

PDO_ind_min = np.argmax(PDO_cs)

# get and store PPN energy points
PPNEnergyPoints = pd.read_csv('PPNEnergyPoints2.csv', names=['z','PPN'])
PPNEnergyPoints.sort_values(by='z',inplace=True)
PPN_z_max = PPNEnergyPoints.loc[:,'z'].max()
PPN_z_min = PPNEnergyPoints.loc[:,'z'].min()
z_PPN = np.linspace(PPN_z_min, PPN_z_max, num = N)
PPN_cs = csaps(PPNEnergyPoints.loc[:,'z'], PPNEnergyPoints.loc[:,'PPN'], z_PPN, smooth=0.5)
h_PPN = (PPN_z_max - PPN_z_min)/(N-1)

# calculate devs of the PPN curves
z_PPN_dev = z_PPN[start:-start]
PPN_cs_first_dev = [(PPN_cs[i+1]-PPN_cs[i-1])/(2*h_PPN) for i in range(start,N-start)]
PPN_cs_sec_dev = [(PPN_cs[i+1]-2*PPN_cs[i]+PPN_cs[i-1])/(h_PPN)**2 for i in range(start,N-start)]
PPN_cs_third_dev = [(PPN_cs[i+2]-2*PPN_cs[i+1]+2*PPN_cs[i-1]-PPN_cs[i-2])/(2*(h_PPN)**3) for i in range(start,N-start)]
#PPN_cs_fourth_dev = [(PPN_cs[i+2]-4*PPN_cs[i+1]+6*PPN_cs[i]-4*PPN_cs[i-1]+PPN_cs[i-2])/(h_PPN)**4 for i in range(start,N-start)]

PPN_ind_min = np.argmax(PPN_cs)

# PDO and PPN curves
plt.scatter(PDOEnergyPoints.loc[:,'z'],PDOEnergyPoints.loc[:,'PDO'], c = 'r',s=5)
plt.plot(z_PDO,PDO_cs, 'r--')
plt.scatter(PPNEnergyPoints.loc[:,'z'],PPNEnergyPoints.loc[:,'PPN'], c = 'b',s=5)
plt.plot(z_PPN,PPN_cs, 'b--')
plt.title('Plot of smoothing splines trained on the PPN and PDO plot')
plt.xlabel('z (Angstroms)')
plt.ylabel('Energy (kcal/mol)')
plt.show()

# PDO 
plt.plot(z_PDO,PDO_cs, 'r--')
plt.plot(z_PDO_dev,PDO_cs_first_dev, 'b--')
plt.plot(z_PDO_dev,PDO_cs_sec_dev, 'y--')

plt.plot(z_PDO_dev,PDO_cs_third_dev, 'k--')
PDO_cs_third_dev_cs = csaps(z_PDO_dev, PDO_cs_third_dev, z_PDO_dev, smooth=0.9)
plt.plot(z_PDO_dev,PDO_cs_third_dev_cs, 'k')

# PDO_cs_fourth_dev_cs = [(PDO_cs_third_dev_cs[i+1]-PDO_cs_third_dev_cs[i-1])/(2*h_PDO) for i in range(1,len(PDO_cs_third_dev_cs)-1)]
# plt.plot(z_PDO_dev[1:-1],PDO_cs_fourth_dev_cs, 'm--')

plt.scatter(z_PDO[PDO_ind_min],PDO_cs[PDO_ind_min])
plt.scatter(z_PDO_dev[PDO_ind_min-1],PDO_cs_first_dev[PDO_ind_min-1])
plt.scatter(z_PDO_dev[PDO_ind_min-1],PDO_cs_sec_dev[PDO_ind_min-1])
plt.scatter(z_PDO_dev[PDO_ind_min-1],PDO_cs_third_dev_cs[PDO_ind_min-1])
# plt.scatter(z_PDO_dev[PDO_ind_min-1],PDO_cs_fourth_dev_cs[PDO_ind_min-1])

plt.axvline(x=z_PDO_dev[PDO_ind_min-1], ymin=np.min([np.min(PDO_cs),np.min(PDO_cs_first_dev),np.min(PDO_cs_sec_dev),np.min(PDO_cs_third_dev_cs)]), 
			ymax=np.max([np.max(PDO_cs),np.max(PDO_cs_first_dev),np.max(PDO_cs_sec_dev),np.max(PDO_cs_third_dev_cs)]))
plt.legend(['PDO', 'first dev PDO', 'second dev PDO', 'third dev PDO', 'smoothed third dev PDO','fourth dev PDO'])
plt.title('Plot of smoothing spline and derivates trained on the PDO plot')
plt.xlabel('z (Angstroms)')
plt.ylabel('Energy (kcal/mol)')

print('max PDO value is ' + str(PDO_cs[PDO_ind_min]))
print('second derivative of PDO is ' + str(PDO_cs_sec_dev[PDO_ind_min-1]))
print('third derivative of PDO is ' + str(PDO_cs_third_dev_cs[PDO_ind_min-1]))
# print('fourth derivative of PDO is ' + str(PDO_cs_fourth_dev_cs[PDO_ind_min-1]))
kBT = 293*0.001985
print(np.exp(PDO_cs[PDO_ind_min]/kBT)*np.sqrt(-2*np.pi*kBT/PDO_cs_sec_dev[PDO_ind_min-1]))
plt.show()



# PPN
plt.plot(z_PPN,PPN_cs, 'r--')
plt.plot(z_PPN_dev,PPN_cs_first_dev, 'b--')
plt.plot(z_PPN_dev,PPN_cs_sec_dev, 'y--')

plt.plot(z_PPN_dev,PPN_cs_third_dev, 'k--')
PPN_cs_third_dev_cs = csaps(z_PPN_dev, PPN_cs_third_dev, z_PPN_dev, smooth=0.9)
plt.plot(z_PPN_dev,PPN_cs_third_dev_cs, 'k')

PPN_cs_fourth_dev_cs = [(PPN_cs_third_dev_cs[i+1]-PPN_cs_third_dev_cs[i-1])/(2*h_PPN) for i in range(1,len(PPN_cs_third_dev_cs)-1)]
plt.plot(z_PPN_dev[1:-1],PPN_cs_fourth_dev_cs, 'm--')

plt.scatter(z_PPN[PPN_ind_min],PPN_cs[PPN_ind_min])
plt.scatter(z_PPN_dev[PPN_ind_min-1],PPN_cs_first_dev[PPN_ind_min-1])
plt.scatter(z_PPN_dev[PPN_ind_min-1],PPN_cs_sec_dev[PPN_ind_min-1])
plt.scatter(z_PPN_dev[PPN_ind_min-1],PPN_cs_third_dev_cs[PPN_ind_min-1])
plt.scatter(z_PPN_dev[PPN_ind_min-1],PPN_cs_fourth_dev_cs[PPN_ind_min-1])

plt.axvline(x=z_PPN_dev[PPN_ind_min-1], ymin=np.min([np.min(PPN_cs),np.min(PPN_cs_first_dev),np.min(PPN_cs_sec_dev),np.min(PPN_cs_third_dev_cs),np.min(PPN_cs_fourth_dev_cs)]), 
			ymax=np.max([np.max(PPN_cs),np.max(PPN_cs_first_dev),np.max(PPN_cs_sec_dev),np.max(PPN_cs_third_dev_cs),np.max(PPN_cs_fourth_dev_cs)]))


plt.title('Plot of smoothing spline and derivates trained on the PPN plot')
plt.legend(['PPN', 'first dev PPN', 'second dev PPN', 'third dev PPN', 'smoothed third dev PPN','fourth dev PPN'])
plt.xlabel('z (Angstroms)')
plt.ylabel('Energy (kcal/mol)')

print('max PPN value is ' + str(PPN_cs[PPN_ind_min]))
print('second derivative of PPN is ' + str(PPN_cs_sec_dev[PPN_ind_min-1]))
print('third derivative of PPN is ' + str(PPN_cs_third_dev_cs[PPN_ind_min-1]))
print('fourth derivative of PPN is ' + str(PPN_cs_fourth_dev_cs[PPN_ind_min-1]))
print(np.exp(PPN_cs[PPN_ind_min]/kBT)*np.sqrt(-2*np.pi*kBT/PPN_cs_sec_dev[PPN_ind_min-1]))

plt.show()
