import numpy as np
from numpy.linalg import LinAlgError
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}'] #for \text command

eigenvalues = np.array([2.37039392e+02, 7.43688593e+00, 3.21889199e+00, 1.86595330e+00,
					    7.15352194e-01, 4.63813647e-01, 2.07885092e-01, 7.47566364e-02,
					    3.24046926e-02, 2.06949062e-04])


x_axis_labels = [ r"$\lambda_{" + str(i+1) + "}'$" for i in range(len(eigenvalues))]
plt.yticks(fontsize= 20)
plt.ylabel(r'$\log_{10}(\lambda_i)$',fontsize= 20)
plt.bar(range(len(eigenvalues)),np.log10(eigenvalues))
plt.xticks(list(range(len(eigenvalues))),x_axis_labels,fontsize= 20)

plt.title(r'$\log_{10}$' + ' plot of the estimated eigenvalues of \n' +r'$E[\nabla_{\vec{\widetilde{p}}}\max_{t}$' + '3-HPA' + r'$(t;\vec{p})(\nabla_{\vec{\widetilde{p}}}\max_{t}$' + '3-HPA' + r'$(t;\vec{p}))^{\top}]$' + ' given an enzyme ratio, 1:18',fontsize= 20)
plt.show()


VARS_TO_TEX = {'kcatfDhaB': r'$k_{\text{cat}}^{f,\text{dhaB}}$',
                'KmDhaBG': r'$K_{\text{M}}^{\text{Glycerol},\text{dhaB}}$',
                'kcatfDhaT': r'$k_{\text{cat}}^{f,\text{dhaT}}$',
                'KmDhaTH': r'$K_{\text{M}}^{\text{3-HPA},\text{dhaT}}$',
                'KmDhaTN': r'$K_{\text{M}}^{\text{NADH},\text{dhaT}}$',
                'NADH_MCP_INIT': '[NADH]',
                'NAD_MCP_INIT': '[NAD+]',
                'km':r'$k_{m}$',
                'kc': r'$k_{c}$',
                'dPacking': 'dPacking', 
                'nmcps': 'Number of MCPs'}
                
params_used = ['kcatfDhaB', # /seconds Input
	            'KmDhaBG', # mM Input
	            'kcatfDhaT', # /seconds
	            'KmDhaTH', # mM
	            'KmDhaTN', # mM
	            'NADH_MCP_INIT',
	            'km', 
	            'kc',
	            'dPacking',
	            'nmcps']
var_tex_names = [VARS_TO_TEX[params] for params in params_used]
eigenvec_names = [r"$\vec{v}'_{" + str(i) + "}$" for i in range(len(eigenvalues))]
rescaled = eigenvectors/np.max(np.abs(eigenvectors))
thresholded = np.multiply(rescaled,(np.abs(rescaled) > 1e-1).astype(float))
plt.imshow(thresholded,cmap='Greys')
plt.xticks(list(range(len(eigenvalues))),eigenvec_names,fontsize= 20)
plt.yticks(list(range(len(eigenvalues))),var_tex_names,fontsize= 20)
plt.title('Heat map of estimated eigenvectors of \n' +r'$E[\nabla_{\vec{\widetilde{p}}}\max_{t}$' + '3-HPA' + r'$(t;\vec{p})(\nabla_{\vec{\widetilde{p}}}\max_{t}$' + '3-HPA' + r'$(t;\vec{p}))^{\top}]$,' + ' given an enzyme ratio, 1:18',fontsize= 20)
cbar = plt.colorbar()
cbar.ax.tick_params(labelsize=20)

plt.show()