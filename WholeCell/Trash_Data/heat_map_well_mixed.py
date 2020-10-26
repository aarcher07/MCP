import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pylab as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.transforms import offset_copy

params_ET_DhaB_df  = pd.read_excel("rod_constant_NADH_1mM_EncapsulationAndExpressionResultsWellMixedMCP.xls")
Ratio = params_ET_DhaB_df["Ratio (DhaB1:DhaT)"].unique()
DhaBperDhaT = [-float(r[:r.find(':')])/float(r[(r.find(':')+1):]) for r in Ratio]
index_sorted_ratio = np.argsort(DhaBperDhaT)
RatioSorted = [Ratio[index] for index in index_sorted_ratio]
KmDhaB = params_ET_DhaB_df["DhaB, Km (mM)"].unique()
KcatDhaB  = params_ET_DhaB_df["DhaB, Kcat (/s)"].unique()
nMCPs = params_ET_DhaB_df["Number of MCPs"].unique()
rows = ['Number of \n MCPs = {}'.format(row) for row in reversed(nMCPs)]
cols = [r'$K_m = {}$ mM'.format(col) for col in KmDhaB]
# Glycerol Consumption times
maxglyceroltime = params_ET_DhaB_df.loc[:,"Time to consume 99% of Glycerol (hrs)"].max()
minglyceroltime = params_ET_DhaB_df.loc[:,"Time to consume 99% of Glycerol (hrs)"].min()
fig, ax = plt.subplots(len(nMCPs), len(KmDhaB), figsize=(13,10))
cbar_ax = fig.add_axes([.91, .3, .03, .4])

for counterMCP,nMCP in enumerate(reversed(nMCPs)):
	for counterKm,Km in enumerate(KmDhaB):
		bools_list = np.logical_and(params_ET_DhaB_df["DhaB, Km (mM)"] == Km,
									 params_ET_DhaB_df["Number of MCPs"] == nMCP)
		GlycerolConsumptionTime= params_ET_DhaB_df.loc[bools_list,"Time to consume 99% of Glycerol (hrs)"].to_numpy()
		GlycerolConsumptionTime_FixedKm_FixednMCPs = GlycerolConsumptionTime.reshape(-1,len(KcatDhaB))
		GlycerolConsumptionTime_FixedKm_FixednMCPs_Sorted = GlycerolConsumptionTime_FixedKm_FixednMCPs[index_sorted_ratio,:]
		sns.heatmap(GlycerolConsumptionTime_FixedKm_FixednMCPs_Sorted,ax=ax[counterMCP,counterKm],square=True, cbar= counterMCP == 0,
					vmax = maxglyceroltime, vmin= minglyceroltime, cbar_ax=None if counterMCP else cbar_ax)
		if counterKm == 0:
			ax[counterMCP,counterKm].set_ylabel('DhaB to \n DhaT ratio', fontsize=10) 
			ax[counterMCP,counterKm].set_yticklabels(RatioSorted,rotation=0) 
		else:
			ax[counterMCP,counterKm].set_yticklabels([])


		if counterMCP == (len(nMCPs)-1):
			ax[counterMCP,counterKm].set_xlabel(r'$K_{cat} $' + '(/s)', fontsize=10)
			ax[counterMCP,counterKm].set_xticklabels(KcatDhaB,rotation=0)
		else:
			ax[counterMCP,counterKm].set_xticklabels([])
pad = 5 # in points

for axes, col in zip(ax[-1], cols):
    axes.annotate(col, xy=(0.5, -0.35), xytext=(0, pad),
                xycoords='axes fraction', textcoords='offset points',
                size='large', ha='center', va='baseline')
for axes, row in zip(ax[:,0], rows):
    axes.annotate(row, xy=(0, 0.5), xytext=(-axes.yaxis.labelpad- pad, 0),
                xycoords=axes.yaxis.label, textcoords='offset points',
                size='large', ha='right', va='center')


colorbar = ax[0,0].collections[0].colorbar
colorbar.ax.set_ylabel('time (hr)') 
fig.subplots_adjust(left=0.15, top=0.95)
plt.show()


# 1,3-PDO Production times
fig, ax = plt.subplots(len(nMCPs), len(KmDhaB), figsize=(13,10))
maxPDOtime = params_ET_DhaB_df.loc[:,"Time to produce 99% of 1,3-PDO (hrs)"].max()
minPDOtime = params_ET_DhaB_df.loc[:,"Time to produce 99% of 1,3-PDO (hrs)"].min()
cbar_ax = fig.add_axes([.91, .3, .03, .4])
for counterMCP,nMCP in enumerate(reversed(nMCPs)):
	for counterKm,Km in enumerate(KmDhaB):
		bools_list = np.logical_and(params_ET_DhaB_df["DhaB, Km (mM)"] == Km,
									 params_ET_DhaB_df["Number of MCPs"] == nMCP)
		PDOConsumptionTime = params_ET_DhaB_df.loc[bools_list,"Time to produce 99% of 1,3-PDO (hrs)"].to_numpy()
		PDOConsumptionTime_FixedKm_FixednMCPs = PDOConsumptionTime.reshape(-1,len(KcatDhaB))
		PDOConsumptionTime_FixedKm_FixednMCPs_Sorted = PDOConsumptionTime_FixedKm_FixednMCPs[index_sorted_ratio,:]
		sns.heatmap(PDOConsumptionTime_FixedKm_FixednMCPs_Sorted,ax=ax[counterMCP,counterKm],square=True, cbar= counterMCP == 0,
					vmax = maxPDOtime, vmin= minPDOtime, cbar_ax=None if counterMCP else cbar_ax)	
		if counterKm == 0:
			ax[counterMCP,counterKm].set_ylabel('DhaB to \n DhaT ratio', fontsize=10) 
			ax[counterMCP,counterKm].set_yticklabels(RatioSorted,rotation=0) 
		else:
			ax[counterMCP,counterKm].set_yticklabels([])


		if counterMCP == (len(nMCPs)-1):
			ax[counterMCP,counterKm].set_xlabel(r'$K_{cat} $' + '(/s)', fontsize=10)
			ax[counterMCP,counterKm].set_xticklabels(KcatDhaB,rotation=0)
		else:
			ax[counterMCP,counterKm].set_xticklabels([])

pad = 5 # in points

for axes, col in zip(ax[-1], cols):
    axes.annotate(col, xy=(0.5, -0.35), xytext=(0, pad),
                xycoords='axes fraction', textcoords='offset points',
                size='large', ha='center', va='baseline')
for axes, row in zip(ax[:,0], rows):
    axes.annotate(row, xy=(0, 0.5), xytext=(-axes.yaxis.labelpad- pad, 0),
                xycoords=axes.yaxis.label, textcoords='offset points',
                size='large', ha='right', va='center')

colorbar = ax[0,0].collections[0].colorbar
colorbar.ax.set_ylabel('time (hr)')
fig.subplots_adjust(left=0.15, top=0.95)
plt.show()


# Maximum concentration of 3-HPA (mM)
fig, ax = plt.subplots(len(nMCPs), len(KmDhaB), figsize=(13,10))
maxHPAconc = params_ET_DhaB_df.loc[:,"Maximum concentration of 3-HPA (mM)"].max()
minHPAconc = params_ET_DhaB_df.loc[:,"Maximum concentration of 3-HPA (mM)"].min()
cbar_ax = fig.add_axes([.91, .3, .03, .4])
for counterMCP,nMCP in enumerate(reversed(nMCPs)):
	for counterKm,Km in enumerate(KmDhaB):
		bools_list = np.logical_and(params_ET_DhaB_df["DhaB, Km (mM)"] == Km,
									 params_ET_DhaB_df["Number of MCPs"] == nMCP)
		MaxHPAConc = params_ET_DhaB_df.loc[bools_list,"Maximum concentration of 3-HPA (mM)"].to_numpy()
		MaxHPAConc_FixedKm_FixednMCPs = MaxHPAConc.reshape(-1,len(KcatDhaB))
		MaxHPAConc_FixedKm_FixednMCPs_Sorted = MaxHPAConc_FixedKm_FixednMCPs[index_sorted_ratio,:]
		sns.heatmap(MaxHPAConc_FixedKm_FixednMCPs_Sorted,ax=ax[counterMCP,counterKm],square=True, cbar= counterMCP == 0,
					vmax = maxHPAconc, vmin= minHPAconc, cbar_ax=None if counterMCP else cbar_ax)	
		if counterKm == 0:
			ax[counterMCP,counterKm].set_ylabel('DhaB to \n DhaT ratio', fontsize=10) 
			ax[counterMCP,counterKm].set_yticklabels(RatioSorted,rotation=0) 
		else:
			ax[counterMCP,counterKm].set_yticklabels([])


		if counterMCP == (len(nMCPs)-1):
			ax[counterMCP,counterKm].set_xlabel(r'$K_{cat} $' + '(/s)', fontsize=10)
			ax[counterMCP,counterKm].set_xticklabels(KcatDhaB,rotation=0)
		else:
			ax[counterMCP,counterKm].set_xticklabels([])

pad = 5 # in points

for axes, col in zip(ax[-1], cols):
    axes.annotate(col, xy=(0.5, -0.35), xytext=(0, pad),
                xycoords='axes fraction', textcoords='offset points',
                size='large', ha='center', va='baseline')
for axes, row in zip(ax[:,0], rows):
    axes.annotate(row, xy=(0, 0.5), xytext=(-axes.yaxis.labelpad- pad, 0),
                xycoords=axes.yaxis.label, textcoords='offset points',
                size='large', ha='right', va='center')

colorbar = ax[0,0].collections[0].colorbar
colorbar.ax.set_ylabel('concen-\n tration (mM)') 
fig.subplots_adjust(left=0.15, top=0.95)
plt.show()


minMIC = 1.0
minMBC = 7.5

# Maximum concentration of 3-HPA (mM) greater than minimum estimated MIC and less than min estimated MBC
fig, ax = plt.subplots(len(nMCPs), len(KmDhaB), figsize=(13,10))
cbar_ax = fig.add_axes([.91, .3, .03, .4])
myColors = ((0.8, 0.0, 0.0, 1.0), (0.0, 0.8, 0.0, 1.0), (0.0, 0.0, 0.8, 1.0))
cmap = LinearSegmentedColormap.from_list('Custom', myColors, len(myColors))

for counterMCP,nMCP in enumerate(reversed(nMCPs)):
	for counterKm,Km in enumerate(KmDhaB):
		bools_list = np.logical_and(params_ET_DhaB_df["DhaB, Km (mM)"] == Km,
									 params_ET_DhaB_df["Number of MCPs"] == nMCP)
		MaxHPAConc = params_ET_DhaB_df.loc[bools_list,"Maximum concentration of 3-HPA (mM)"].to_numpy()
		MICMBCMaxHPAConc= np.logical_and(minMIC < MaxHPAConc,  MaxHPAConc < minMBC).astype(int)  
		MinMBCMaxHPAConc= ( MaxHPAConc > minMBC)
		MICMBCMaxHPAConc[MinMBCMaxHPAConc] = 2
		MICMBCMaxHPAConc_FixedKm_FixednMCPs = MICMBCMaxHPAConc.reshape(-1,len(KcatDhaB))
		MICMBCMaxHPAConc_FixedKm_FixednMCPs_Sorted = MICMBCMaxHPAConc_FixedKm_FixednMCPs[index_sorted_ratio,:]
		sns.heatmap(MICMBCMaxHPAConc_FixedKm_FixednMCPs_Sorted,ax=ax[counterMCP,counterKm],square=True, cmap=cmap,
					cbar= (counterMCP+counterKm) == 0, vmax = 2, vmin= 0, cbar_ax=None if (counterMCP+counterKm) else cbar_ax)	
		if counterKm == 0:
			ax[counterMCP,counterKm].set_ylabel('DhaB to \n DhaT ratio', fontsize=10) 
			ax[counterMCP,counterKm].set_yticklabels(RatioSorted,rotation=0) 
		else:
			ax[counterMCP,counterKm].set_yticklabels([])


		if counterMCP == (len(nMCPs)-1):
			ax[counterMCP,counterKm].set_xlabel(r'$K_{cat} $' + '(/s)', fontsize=10)
			ax[counterMCP,counterKm].set_xticklabels(KcatDhaB,rotation=0)
		else:
			ax[counterMCP,counterKm].set_xticklabels([])


colorbar = ax[0,0].collections[0].colorbar
colorbar.set_ticks([0.333, 1, 1.66])
colorbar.set_ticklabels(['less then\n MIC', 'between MIC \n and MBC', 'greater \n than MBC'])
colorbar.ax.tick_params(labelsize=7) 

pad = 5 # in points

for axes, col in zip(ax[-1], cols):
    axes.annotate(col, xy=(0.5, -0.35), xytext=(0, pad),
                xycoords='axes fraction', textcoords='offset points',
                size='large', ha='center', va='baseline')
for axes, row in zip(ax[:,0], rows):
    axes.annotate(row, xy=(0, 0.5), xytext=(-axes.yaxis.labelpad- pad, 0),
                xycoords=axes.yaxis.label, textcoords='offset points',
                size='large', ha='right', va='center')

fig.subplots_adjust(left=0.15, top=0.95)
plt.show()

