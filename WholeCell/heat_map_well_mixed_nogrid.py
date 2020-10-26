import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pylab as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.transforms import offset_copy

fname1 = "EncapsulationAndExpressionResultsWellMixedMCP"
fname2 = "EncapsulationOnlyTableResultsWellMixed" 
params_ET_DhaB_df  = pd.read_excel("rod_constant_NADH_1mM_" + fname1 +".xls")
params_ET_DhaB_df2  = pd.read_excel("rod_constant_NADH_1mM_" + fname2 +".xls")

Ratio = params_ET_DhaB_df["Ratio (DhaB1:DhaT)"].unique()
DhaBperDhaT = [-float(r[:r.find(':')])/float(r[(r.find(':')+1):]) for r in Ratio]
index_sorted_ratio = np.argsort(DhaBperDhaT)
RatioSorted = [Ratio[index] for index in index_sorted_ratio]
KmDhaB = params_ET_DhaB_df["DhaB, Km (mM)"].unique()
Km = KmDhaB[0]
KcatDhaB  = params_ET_DhaB_df["DhaB, Kcat (/s)"].unique()
nMCPs = params_ET_DhaB_df["Number of MCPs"].unique()
nMCP = nMCPs[-1]
rows = ['Number of \n MCPs = {}'.format(row) for row in reversed(nMCPs)]
cols = [r'$K_m = {}$ mM'.format(col) for col in KmDhaB]
font_size = 20

# Glycerol Consumption times
maxglyceroltime = max(params_ET_DhaB_df.loc[:,"Time to consume 99% of Glycerol (hrs)"].max(),params_ET_DhaB_df2.loc[:,"Time to consume 99% of Glycerol (hrs)"].max())
minglyceroltime = min(params_ET_DhaB_df.loc[:,"Time to consume 99% of Glycerol (hrs)"].min(),params_ET_DhaB_df2.loc[:,"Time to consume 99% of Glycerol (hrs)"].min())
fig, ax = plt.subplots(1, 1, figsize=(14,10))
cbar_ax = fig.add_axes([.85, .3, .03, .4])
bools_list = np.logical_and(params_ET_DhaB_df["DhaB, Km (mM)"] == Km,
							 params_ET_DhaB_df["Number of MCPs"] == nMCP)
GlycerolConsumptionTime= params_ET_DhaB_df.loc[bools_list,"Time to consume 99% of Glycerol (hrs)"].to_numpy()
GlycerolConsumptionTime_FixedKm_FixednMCPs = GlycerolConsumptionTime.reshape(-1,len(KcatDhaB))
GlycerolConsumptionTime_FixedKm_FixednMCPs_Sorted = GlycerolConsumptionTime_FixedKm_FixednMCPs[index_sorted_ratio,:]
sns.heatmap(GlycerolConsumptionTime_FixedKm_FixednMCPs_Sorted,ax=ax,square=True, cbar= True,
			vmax = maxglyceroltime, vmin= minglyceroltime, cbar_ax= cbar_ax)
ax.set_ylabel('DhaB to \n DhaT ratio', fontsize=font_size) 
ax.set_yticklabels(RatioSorted,rotation=0, fontsize=font_size)
ax.set_xlabel(r'$K_{cat} $' + '(/s)', fontsize=font_size)
ax.set_xticklabels(KcatDhaB,rotation=0, fontsize=font_size)
colorbar = ax.collections[0].colorbar
colorbar.ax.set_ylabel('time (hr)', fontsize=font_size) 
colorbar.ax.tick_params(labelsize=font_size)
fig.subplots_adjust(left=0.15, top=0.95)
plt.savefig(fname1 + "_G99" + ".png", bbox_inches = 'tight', pad_inches = 0)
plt.show()


# 1,3-PDO Production times
fig, ax = plt.subplots(1,1, figsize=(14,10))
maxPDOtime = max(params_ET_DhaB_df.loc[:,"Time to produce 99% of 1,3-PDO (hrs)"].max(),params_ET_DhaB_df2.loc[:,"Time to produce 99% of 1,3-PDO (hrs)"].max())
minPDOtime = min(params_ET_DhaB_df.loc[:,"Time to produce 99% of 1,3-PDO (hrs)"].min(),params_ET_DhaB_df2.loc[:,"Time to produce 99% of 1,3-PDO (hrs)"].min())
cbar_ax = fig.add_axes([.85, .3, .03, .4])
bools_list = np.logical_and(params_ET_DhaB_df["DhaB, Km (mM)"] == Km,
							 params_ET_DhaB_df["Number of MCPs"] == nMCP)
PDOConsumptionTime = params_ET_DhaB_df.loc[bools_list,"Time to produce 99% of 1,3-PDO (hrs)"].to_numpy()
PDOConsumptionTime_FixedKm_FixednMCPs = PDOConsumptionTime.reshape(-1,len(KcatDhaB))
PDOConsumptionTime_FixedKm_FixednMCPs_Sorted = PDOConsumptionTime_FixedKm_FixednMCPs[index_sorted_ratio,:]
sns.heatmap(PDOConsumptionTime_FixedKm_FixednMCPs_Sorted,ax=ax,square=True, cbar= True,
			vmax = maxPDOtime, vmin= minPDOtime, cbar_ax=cbar_ax)	
ax.set_ylabel('DhaB to \n DhaT ratio', fontsize=font_size) 
ax.set_yticklabels(RatioSorted,rotation=0, fontsize=font_size)
ax.set_xlabel(r'$K_{cat} $' + '(/s)', fontsize=font_size)
ax.set_xticklabels(KcatDhaB,rotation=0, fontsize=font_size)
colorbar = ax.collections[0].colorbar
colorbar.ax.set_ylabel('time (hr)', fontsize=font_size) 
colorbar.ax.tick_params(labelsize=font_size)
fig.subplots_adjust(left=0.15, top=0.95)
plt.savefig(fname1 + "_13PDO99" + ".png", bbox_inches = 'tight', pad_inches = 0)
plt.show()


# Maximum concentration of 3-HPA (mM)
fig, ax = plt.subplots(1,1, figsize=(14,10))
maxHPAconc = max(params_ET_DhaB_df.loc[:,"Maximum concentration of 3-HPA (mM)"].max(),params_ET_DhaB_df2.loc[:,"Maximum concentration of 3-HPA (mM)"].max())
minHPAconc = min(params_ET_DhaB_df.loc[:,"Maximum concentration of 3-HPA (mM)"].min(),params_ET_DhaB_df2.loc[:,"Maximum concentration of 3-HPA (mM)"].min())
cbar_ax = fig.add_axes([.85, .3, .03, .4])
bools_list = np.logical_and(params_ET_DhaB_df["DhaB, Km (mM)"] == Km,
							 params_ET_DhaB_df["Number of MCPs"] == nMCP)
MaxHPAConc = params_ET_DhaB_df.loc[bools_list,"Maximum concentration of 3-HPA (mM)"].to_numpy()
MaxHPAConc_FixedKm_FixednMCPs = MaxHPAConc.reshape(-1,len(KcatDhaB))
MaxHPAConc_FixedKm_FixednMCPs_Sorted = MaxHPAConc_FixedKm_FixednMCPs[index_sorted_ratio,:]
sns.heatmap(MaxHPAConc_FixedKm_FixednMCPs_Sorted,ax=ax,square=True, cbar= True,
			vmax = maxHPAconc, vmin= minHPAconc, cbar_ax=cbar_ax)	
ax.set_ylabel('DhaB to \n DhaT ratio', fontsize=font_size) 
ax.set_yticklabels(RatioSorted,rotation=0, fontsize=font_size)
ax.set_xlabel(r'$K_{cat} $' + '(/s)', fontsize=font_size)
ax.set_xticklabels(KcatDhaB,rotation=0, fontsize=font_size)
colorbar = ax.collections[0].colorbar
colorbar.ax.set_ylabel('concentration (mM)', fontsize=font_size) 
colorbar.ax.tick_params(labelsize=font_size)
fig.subplots_adjust(left=0.15, top=0.95)
plt.savefig(fname1 + "_max3HPA" + ".png", bbox_inches = 'tight', pad_inches = 0)
plt.show()


minMIC = 1.0
minMBC = 7.5
# Maximum concentration of 3-HPA (mM) greater than minimum estimated MIC and less than min estimated MBC
fig, ax = plt.subplots(1,1, figsize=(14.5,10))
cbar_ax = fig.add_axes([.8, .3, .03, .4])
myColors = ((0.8, 0.0, 0.0, 1.0), (0.0, 0.8, 0.0, 1.0), (0.0, 0.0, 0.8, 1.0))
cmap = LinearSegmentedColormap.from_list('Custom', myColors, len(myColors))
bools_list = np.logical_and(params_ET_DhaB_df["DhaB, Km (mM)"] == Km,
							 params_ET_DhaB_df["Number of MCPs"] == nMCP)
MaxHPAConc = params_ET_DhaB_df.loc[bools_list,"Maximum concentration of 3-HPA (mM)"].to_numpy()
MICMBCMaxHPAConc= np.logical_and(minMIC < MaxHPAConc,  MaxHPAConc < minMBC).astype(int)  
MinMBCMaxHPAConc= ( MaxHPAConc > minMBC)
MICMBCMaxHPAConc[MinMBCMaxHPAConc] = 2
MICMBCMaxHPAConc_FixedKm_FixednMCPs = MICMBCMaxHPAConc.reshape(-1,len(KcatDhaB))
MICMBCMaxHPAConc_FixedKm_FixednMCPs_Sorted = MICMBCMaxHPAConc_FixedKm_FixednMCPs[index_sorted_ratio,:]
sns.heatmap(MICMBCMaxHPAConc_FixedKm_FixednMCPs_Sorted,ax=ax,square=True, cmap=cmap,
			cbar= True, vmax = 2, vmin= 0, cbar_ax=cbar_ax)	
ax.set_ylabel('DhaB to \n DhaT ratio', fontsize=font_size) 
ax.set_yticklabels(RatioSorted,rotation=0, fontsize=font_size)
ax.set_xlabel(r'$K_{cat} $' + '(/s)', fontsize=font_size)
ax.set_xticklabels(KcatDhaB,rotation=0, fontsize=font_size)
colorbar = ax.collections[0].colorbar
colorbar.set_ticks([0.333, 1, 1.66])
colorbar.set_ticklabels(['less then\n MIC', 'between MIC \n and MBC', 'greater \n than MBC'])
colorbar.ax.tick_params(labelsize=20) 
plt.savefig(fname1 + "_max3HPA_MIC_MBC"+ ".png", bbox_inches = 'tight', pad_inches = 0)
plt.show()

