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

eigenvectors = np.array([[ 3.59966465e-01, -1.71073368e-01, -1.62975831e-01,
        -5.51121207e-02, -9.76472198e-03, -8.61181302e-02,
         1.95594007e-02,  1.29345145e-01, -8.86107015e-01,
        -4.18272534e-02],
       [-6.94777223e-03, -5.64755982e-03, -4.71861086e-03,
        -1.68862503e-03, -5.88381112e-03, -7.19265844e-03,
         6.20698682e-03,  5.50630794e-03, -4.62013921e-02,
         9.98801756e-01],
       [-9.14679684e-01,  8.51544969e-02, -1.66523899e-01,
        -4.66053577e-02,  4.43819707e-04, -2.37341593e-02,
         7.30357061e-02,  6.32532510e-02, -3.40233241e-01,
        -2.34556103e-02],
       [ 6.27125454e-02, -5.57675310e-03, -3.69582428e-01,
        -1.50101216e-02,  2.99491084e-01, -6.96579684e-01,
         4.82234919e-01,  1.26651736e-01,  1.88970986e-01,
         4.27470123e-04],
       [ 6.59363964e-02,  3.13470337e-02, -5.14454059e-01,
        -8.59163521e-02, -8.17281735e-01,  8.80651762e-02,
         1.71617011e-01,  2.82029428e-02,  1.29114549e-01,
        -1.36964020e-03],
       [-6.75032840e-02, -8.08131530e-02,  7.16393368e-01,
         1.22568663e-01, -4.47956554e-01, -3.46187454e-01,
         3.29859269e-01,  1.56641537e-01, -8.20444117e-02,
        -9.17524928e-03],
       [ 1.38839518e-01,  9.37133625e-01,  8.75452015e-02,
        -1.71958430e-01,  2.60655098e-02,  4.85146529e-02,
         9.43122356e-02,  2.06883771e-01, -1.02671058e-01,
         4.14566227e-04],
       [ 2.08677531e-02,  1.43255556e-01, -1.35559839e-01,
         9.56495011e-01,  2.76314144e-02,  1.47578035e-01,
         1.25254503e-01,  6.65510630e-02, -5.58781893e-02,
        -5.72647577e-04],
       [-1.11454149e-02, -2.17156118e-01, -1.24195761e-02,
        -7.51103504e-02,  9.97565267e-02,  2.83435942e-01,
         3.62282636e-02,  9.12653807e-01,  1.49594745e-01,
         2.80095257e-03],
       [ 3.26142305e-02, -1.00039250e-01,  7.08372703e-02,
        -1.48538782e-01,  1.73684040e-01,  5.24094703e-01,
         7.73006248e-01, -2.40206142e-01, -4.20349888e-02,
        -8.81918288e-04]])
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