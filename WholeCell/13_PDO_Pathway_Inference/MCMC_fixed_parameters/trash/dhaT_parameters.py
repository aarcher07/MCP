import sympy as sp
import numpy as np
from scipy.optimize import least_squares
from numpy.random import uniform
import matplotlib.pyplot as plt
import seaborn as sns
cmap = sns.cubehelix_palette(as_cmap=True)

fun_AB = lambda kcat,K_IA,K_MA, K_MB: [-(1/59.5)*sp.exp(kcat) + 1 + sp.exp(K_MB)/5,
									   -(0.03/59.5)*sp.exp(kcat) + sp.exp(K_MA) + (sp.exp(K_IA)*sp.exp(K_MB))/5,
									   -(1/59.4)*sp.exp(kcat) + 1 + (10.*sp.exp(K_MA)),
									   	-(0.77/59.5)*sp.exp(kcat) + sp.exp(K_MB) + 10.*sp.exp(K_IA)*sp.exp(K_MB)]
fun_AB_array = lambda params: np.array(fun_AB(*params),dtype=float)
syms =sp.symbols(['kcat','K_IA','K_MA', 'K_MB'])

fun_AB_sp = fun_AB(*syms)
fun_AB_sp_jac = sp.Matrix(fun_AB_sp).jacobian(syms)
fun_AB_sp_jac_lam = sp.lambdify(syms,fun_AB_sp_jac,'numpy')
fun_AB_jac = lambda params: fun_AB_sp_jac_lam(*params)
res_list = []
cost_vals = []
size=100
while(len(res_list) < size):
	try:
		res = least_squares(fun_AB_array, [uniform(0,10),uniform(-15,0),uniform(-15,0),uniform(-15,0)], jac=fun_AB_jac)
	except ValueError:
		continue
	cost_vals.append(res.cost)
	res_list.append(res.x)


for i in range(4):
	plt.scatter(range(size),np.array(res_list)[:,i],c=cost_vals, cmap=cmap)
	plt.colorbar()
	plt.show()

# fun_QP = lambda kcat,K_IQ,K_MQ, K_MP: [-(1/9.7)*sp.exp(kcat) + 1 + sp.exp(K_MP)/50,
# 									   -(0.23/9.7)*sp.exp(kcat) + sp.exp(K_MQ) + (sp.exp(K_IQ)*sp.exp(K_MP))/50,
# 									   -(1/1.)*sp.exp(kcat) + 1 + (0.5*sp.exp(K_MQ)),
# 									   	-(7.4/1.0)*sp.exp(kcat) + sp.exp(K_MP) + 0.5*sp.exp(K_IQ)*sp.exp(K_MQ)]
# fun_QP_array = lambda params: np.array(fun_AB(*params),dtype=float)
# syms =sp.symbols(['kcat','K_IQ','K_MQ', 'K_MP'])

# fun_QP_sp = fun_QP(*syms)
# fun_QP_sp_jac = sp.Matrix(fun_QP_sp).jacobian(syms)
# fun_QP_sp_jac_lam = sp.lambdify(syms,fun_QP_sp_jac,'numpy')
# fun_QP_jac = lambda params: fun_QP_sp_jac_lam(*params)
# res_list = []
# cost_vals = []
# size=100
# while(len(res_list) < size):
# 	try:
# 		res = least_squares(fun_QP_array, [uniform(0,10),-10**(uniform(-3,1)),-10**(uniform(-3,1)),-10**(uniform(-3,1))], jac=fun_QP_jac)
# 	except ValueError:
# 		continue
# 	cost_vals.append(res.cost)
# 	res_list.append(res.x)


# for i in range(4):
# 	plt.scatter(range(size),np.array(res_list)[:,i],c=cost_vals, cmap=cmap)
# 	plt.show()