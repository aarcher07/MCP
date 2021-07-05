from wild_type_model import WildType
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from csaps import csaps
from constants import HRS_TO_SECS

GC_ODs_N = pd.read_csv("data/GC_ODs_N.csv")
Time = GC_ODs_N.loc[:,'Time'].astype(np.float64)

# log transform and fit
WT_a_log10 = np.log10(GC_ODs_N.loc[:, 'WT_a'])
spllog10 = csaps(Time, WT_a_log10, smooth=0.01)

# plot log10 data and spline
t = np.linspace(0, Time.iloc[-1], num=int(1e2))
plt.scatter(Time, WT_a_log10)
plt.plot(t, spllog10(t))
plt.legend(['data', 'spline'], loc='upper right')
plt.title('log(OD) fit to cubic spline')
plt.show()

# plot untransformed data spline
spl = lambda t: 10**spllog10(t)
plt.scatter(Time, np.power(10,WT_a_log10))
plt.plot(t, spl(t))
plt.title('log(OD) fit to cubic spline transformed')
plt.legend(['data', 'spline'], loc='upper right')
plt.show()
# create model

# MCP geometry
radius_mcp = 7e-8
mcp_surface_area = 4*np.pi*(radius_mcp**2)
mcp_volume = (4/3)*np.pi*(radius_mcp**3)

# cell geometry
cell_radius = 0.375e-6
cell_length = 2.47e-6
cell_surface_area = 2*np.pi*cell_radius*cell_length
cell_volume = 4*np.pi/3*(cell_radius)**3 + np.pi*(cell_length - 2*cell_radius)*(cell_radius**2)

# external volume geometry
external_volume = 0.002
wild_type_model = WildType(spl, Time.iloc[-1], mcp_surface_area, mcp_volume,
                           cell_surface_area, cell_volume, external_volume)
PermMCPPolar =10 ** -2
PermMCPNonPolar = 5 * 10 ** -3

# initialize parameters
params = {'PermMCPPropanediol': PermMCPPolar,
            'PermMCPPropionaldehyde': PermMCPNonPolar,
            'PermMCPPropanol': PermMCPPolar,
            'PermMCPPropionyl': PermMCPNonPolar,
            'PermMCPPropionate': PermMCPPolar,
            'nmcps': 10,
            'PermCellPropanediol': 10**-4,
            'PermCellPropionaldehyde': 10**-2,
            'PermCellPropanol': 10**-4,
            'PermCellPropionyl': 10**-5,
            'PermCellPropionate': 10**-7,
            'VmaxCDEf': (3e2)*(1e2),
            'KmCDEPropanediol': 0.5,
            'VmaxPf': (3e2)*(1e2),
            'KmPfPropionaldehyde': 0.5,
            'VmaxPr': (3e2)*(1e2),
            'KmPrPropionyl':  0.5,
            'VmaxQf': (3e2)*(1e2),
            'KmQfPropionaldehyde':  0.5,
            'VmaxQr': (3e2)*(1e2),
            'KmQrPropanol':  0.5,
            'VmaxLf': 0,
            'KmLPropionyl': 0.5}

# initialize initial conditions
init_conds = {'PROPANEDIOL_MCP_INIT': 0,
              'PROPIONALDEHYDE_MCP_INIT': 0,
              'PROPANOL_MCP_INIT': 0,
              'PROPIONYL_MCP_INIT': 0,
              'PROPIONATE_MCP_INIT': 0,
              'PROPANEDIOL_CYTO_INIT': 0,
              'PROPIONALDEHYDE_CYTO_INIT': 0,
              'PROPANOL_CYTO_INIT': 0,
              'PROPIONYL_CYTO_INIT': 0,
              'PROPIONATE_CYTO_INIT': 0,
              'PROPANEDIOL_EXT_INIT': 50,
              'PROPIONALDEHYDE_EXT_INIT': 0,
              'PROPANOL_EXT_INIT': 0,
              'PROPIONYL_EXT_INIT': 0,
              'PROPIONATE_EXT_INIT': 0}

# run model for parameter set
time_concat, sol_concat = wild_type_model.generate_time_series(init_conds, params)

# plot MCP solutions
yext = sol_concat[:, :5]
plt.plot(time_concat/HRS_TO_SECS, yext)
plt.legend(['Propanediol', 'Propionaldehyde', 'Propanol', 'Propionyl', 'Propionate'], loc='upper right')
plt.title('Plot of MCP concentrations')
plt.xlabel('time (hr)')
plt.ylabel('concentration (mM)')
plt.show()

# plot cellular solution
yext = sol_concat[:, 5:10]
plt.plot(time_concat/HRS_TO_SECS, yext)
plt.legend(['Propanediol', 'Propionaldehyde', 'Propanol', 'Propionyl', 'Propionate'], loc='upper right')
plt.title('Plot of cytosol concentrations')
plt.xlabel('time (hr)')
plt.ylabel('concentration (mM)')
plt.show()

# plot external solution
yext = sol_concat[:, 10:15]
plt.plot(time_concat/HRS_TO_SECS, yext)
plt.legend(['Propanediol', 'Propionaldehyde', 'Propanol', 'Propionyl', 'Propionate'], loc='upper right')
plt.title('Plot of external concentrations')
plt.xlabel('time (hr)')
plt.ylabel('concentration (mM)')
plt.show()
