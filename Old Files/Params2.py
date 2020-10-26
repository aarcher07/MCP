'''Parameters of system - Inputs'''
Rc = 1.0e-5 #cm
Kcde = 0.5e3 #microM
Kpq = 15.0e3 #microM
kcatCDE = 300.0  #reactions/s
kcatPQ = 55.0 #reactions/s
Ncde = 1500.0 #Pdu CDE active sites
Npq = 2500.0 # Pdu PQ active sites
DP = 1.0e-5 #cm^2/s
DA = 1.0e-5 #cm^2/s
DC = 1.0e-5 #cm^2/s
kcP = 1.e-4 #cm/s
kcA = kcP #cm/s
kcC = kcP #cm/s
Pcyt = 30e3 #microM cytosolic concentration
Acyt = 0 #microM cytosolic concentration
Ccyt = 0 #product cytocolic concentration
tstop = 30000 #s final time point for integration 
numgrid = 100.0 #number of grid points the radius is split into
MCPmil = 1.e12

Pmultiplier = 1.
Omultiplier = 1.

'''Parameter combinations - do not touch'''
import numpy as np
VMCP = np.divide(4.,3.)*np.pi*np.power(Rc,3)
Vratio = 20000#np.divide(1-VMCP*MCPmil,VMCP*MCPmil) #ratio of MCP volume to free bath volume per MCP 
Na = 6.022e23 #avogadro's number - constant
MCPMolar = MCPmil*1000/Na
Vcde = Pmultiplier*np.divide(kcatCDE*Ncde*1.0e9,(np.divide(4,3.0)*np.pi*(Rc**3)*Na))
Vpq = Omultiplier*np.divide(kcatPQ*Npq*1.0e9,(np.divide(4,3.0)*np.pi*(Rc**3)*Na))
kappa = Kcde/Kpq
gamma = 2*Vpq/Vcde
xiP = (Kpq*DP)/(Vcde*(Rc**2))
xiA = (Kpq*DA)/(Vcde*(Rc**2))
xiC = (Kpq*DC)/(Vcde*(Rc**2))
taucde = Kcde/Vcde
chiP = (Kpq*kcP)/(Vcde*Rc)
chiA = (Kpq*kcA)/(Vcde*Rc)
chiC = (Kpq*kcC)/(Vcde*Rc)
pcyt = Pcyt/Kcde
acyt = Acyt/Kpq
ccyt = Ccyt/Kpq
dm = 1/float(numgrid)

tfinal = int(np.divide(np.multiply(tstop, kappa),taucde))


nM = 3*np.int(np.reciprocal(dm))+3
initials = np.zeros(nM)

p = np.array([xiP, xiA, xiC, kappa, gamma, chiP, chiA, chiC, pcyt, acyt, Vratio, dm])
