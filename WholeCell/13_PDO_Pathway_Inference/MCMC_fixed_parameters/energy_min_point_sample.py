import numpy as np
import math
import matplotlib.pyplot as plt 
from constants import *
import pickle
import time
import scipy.stats as stats

def point_exchange(points, crit, pointdraw,iters = 1000, ndraw =10):
  currcrit = crit(points)
  for k in range(iters):
    rowd = np.random.choice(range(points.shape[0]), size=1)
    for l in range(ndraw):
      oldrow = points[rowd,:]
      points[rowd,:] = pointdraw()
      propcrit = crit(points)
      if(propcrit < currcrit):
        currcrit = propcrit
      else:
        points[rowd,:] = oldrow
  return points

def energyscore(tc,ds = "log_unif"):
  #need MCMC results
  if ds == "log_unif":
    ts = np.random.uniform(size=(10**4,len(PARAMETER_LIST)))
  elif ds == "log_norm":
    mu = np.zeros(len(param_sens_log_norm_bounds))
    sigma = np.zeros(len(param_sens_log_norm_bounds))
    for i,val in enumerate(param_sens_log_norm_bounds.values()):
      mu[i]=val[0]
      sigma[i]=val[1]

    Sigma = np.diag(np.array(sigma))
    ts = stats.multivariate_normal.rvs(mean=mu, cov=Sigma, size=10**4)

  #defining the energy score
  EDtctc =  np.zeros((tc.shape[0],tc.shape[0]))
  EDtcts =  np.zeros((tc.shape[0],ts.shape[0]))
  for dimlcv in range(tc.shape[1]):
    EDtctc = EDtctc+ (np.subtract.outer(tc[:,dimlcv],tc[:,dimlcv]))**2
    EDtcts = EDtcts+ (np.subtract.outer(tc[:,dimlcv],ts[:,dimlcv]))**2

  return -0.5*np.mean(np.sqrt(EDtctc)) + np.mean(np.sqrt(EDtcts))

def main():
  t = np.random.uniform(size=(30,2))
  plt.scatter(t[:,0],t[:,1])
  plt.show()
  pointdraw = lambda: np.random.uniform(size=(1,2))

  tdstar = point_exchange(t,energyscore, pointdraw, iters = 10)
  plt.scatter(tdstar[:,0],tdstar[:,1])
  plt.show()

  tdstar = point_exchange(tdstar,energyscore, pointdraw, iters = 90)
  plt.scatter(tdstar[:,0],tdstar[:,1])
  plt.show()

  tdstar = point_exchange(tdstar,energyscore, pointdraw, iters = 900)
  plt.scatter(tdstar[:,0],tdstar[:,1]) 
  plt.show()


def sample_project(ds="log_unif",length=100):

  if ds == "log_unif":
    t = np.random.uniform(size=(int(length),len(PARAMETER_LIST)))
    pointdraw = lambda: np.random.uniform(size=(1,len(PARAMETER_LIST)))
  elif ds == "log_norm":
    mu = np.zeros(len(param_sens_log_norm_bounds))
    sigma = np.zeros(len(param_sens_log_norm_bounds))
    i=0
    for val in param_sens_log_norm_bounds.values():
      mu[i]=val[0]
      sigma[i]=val[1]
      i+=1
    Sigma = np.diag(np.array(sigma))
    t = stats.multivariate_normal.rvs(mean=mu, cov=Sigma, size=length)
    pointdraw = lambda: stats.multivariate_normal.rvs(mean=mu, cov=Sigma, size=1)

  # plt.scatter(t[:,0],t[:,1])
  # plt.show()
  tdstar = t
  # tdstar = point_exchange(t,energyscore, pointdraw, iters = 10)

  # plt.scatter(tdstar[:,0],tdstar[:,1])
  # plt.show()

  # tdstar = point_exchange(tdstar,energyscore, pointdraw, iters = 90)

  # plt.scatter(tdstar[:,0],tdstar[:,1])
  # plt.show()

  # tdstar = point_exchange(tdstar,energyscore, pointdraw, iters = 900)
  # plt.scatter(tdstar[:,0],tdstar[:,1])
  # plt.show()

  date_string = time.strftime("%Y_%m_%d_%H:%M")
  file_name_pickle = 'emulator_data/' + ds[4:] + "_sample_paramspace_len_"+ str(int(length)) +'_date_'+date_string + '.pkl'
  with open(file_name_pickle, 'wb') as f:
      pickle.dump(tdstar, f)

if __name__ == '__main__':
    for ds in ["log_norm"]:
      for length in [50,100]:
        print(length)
        sample_project(ds,length)