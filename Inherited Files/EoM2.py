import numpy as np

def deriv2(allvar,time, params):
    nM =int(np.floor(np.divide(1,params[-1])))
    d = np.zeros(3*nM+3)

    variables = allvar[:-3]
    ext = allvar[-3:]
    allder = np.zeros(3*nM+6)
    
    for i in range(1,nM):
        d[3*i] = np.divide(9*(((i+0.5)**(4/3.0))*(variables[3*(i+1)] - variables[3*i]) - ((i-0.5)**(4/3.0))*(variables[3*i] - variables[3*(i-1)]))*params[0], (params[-1])**(2/3.0)) - np.divide(variables[3*i],params[3]*(1+variables[3*i]))

        d[3*i + 1] = np.divide(9*(((i+0.5)**(4/3.0))*(variables[3*i+4]-variables[3*i+1]) -((i-0.5)**(4/3.0))*(variables[3*i+1] - variables[3*i-2]))*params[1],(params[-1])**(2/3.0)) + np.divide(variables[3*i],(1+variables[3*i])) - np.divide(params[4]*variables[3*i+1],1+variables[3*i+1])

        d[3*i + 2] = np.divide(9*(((i+0.5)**(4/3.0))*(variables[3*i+5]-variables[3*i+2]) -((i-0.5)**(4/3.0))*(variables[3*i+2] - variables[3*i-1]))*params[2],(params[-1])**(2/3.0)) + np.divide(params[4]*variables[3*i+1],2*(1+variables[3*i+1]))

    d[0] = np.divide(18*(0.5**(4/3.0))*(variables[3] - variables[0])*params[0],(params[-1])**(2/3.0)) - np.divide(variables[0],params[3]*(1+variables[0]))
    d[1] = np.divide(18*(0.5**(4/3.0))*(variables[4] - variables[1])*params[1], (params[-1])**(2/3.0)) + np.divide(variables[0],1+variables[0]) - np.divide(params[4]*variables[1],1+variables[1])
    d[2] = np.divide(18*(0.5**(4/3.0))*(variables[5] - variables[2])*params[2], (params[-1])**(2/3.0)) + np.divide(params[4]*variables[1],2*(1+variables[1]))

    d[-3] = np.divide(3*((nM+0.5)**(2/3.0))*(ext[0] - variables[-3])*params[5],params[-1]**(1/3.0)) - np.divide(9*((nM-0.5)**(4/3.0))*(variables[-3] - variables[-6])*params[0],params[-1]**(2/3.0))
    d[-2] = np.divide(3*((nM+0.5)**(2/3.0))*(ext[1] - variables[-2])*params[6],params[-1]**(1/3.0)) - np.divide(9*((nM-0.5)**(4/3.0))*(variables[-2] - variables[-5])*params[1],params[-1]**(2/3.0))
    d[-1] = np.divide(3*((nM+0.5)**(2/3.0))*(ext[2] - variables[-1])*params[7],params[-1]**(1/3.0)) - np.divide(9*((nM-0.5)**(4/3.0))*(variables[-1] - variables[-4])*params[2],params[-1]**(2/3.0))

    dext = np.zeros(3)
    
    dext[0] = -np.divide(3*((nM+0.5)**(2/3.0))*(ext[0] - variables[-3])*params[5],params[-1]**(1/3.0))*np.divide(params[-1],params[-2])
    dext[1] = -np.divide(3*((nM+0.5)**(2/3.0))*(ext[1] - variables[-2])*params[6],params[-1]**(1/3.0))*np.divide(params[-1],params[-2])
    dext[2] = -np.divide(3*((nM+0.5)**(2/3.0))*(ext[2] - variables[-1])*params[7],params[-1]**(1/3.0))*np.divide(params[-1],params[-2])
    
    allder[:-3] = d
    allder[-3:] = dext
    return allder
