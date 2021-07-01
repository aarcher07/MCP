"""
This code was originally published by the following individuals for use with
Scilab:
    Copyright (C) 2012 - 2013 - Michael Baudin
    Copyright (C) 2012 - Maria Christopoulou
    Copyright (C) 2010 - 2011 - INRIA - Michael Baudin
    Copyright (C) 2009 - Yann Collette
    Copyright (C) 2009 - CEA - Jean-Marc Martinez

    website: forge.scilab.org/index.php/p/scidoe/sourcetree/master/macros
Much thanks goes to these individuals. It has been converted to Python by
Abraham Lee.

Edits made by Andre Archer. The edits were taken from maximinLHS Robert Carnell in R

"""

import numpy as np
import matplotlib.pyplot as plt
from misc import load_obj
from constants import QOI_NAMES
from mpl_toolkits.mplot3d import Axes3D

__all__ = ['lhs']


def lhs(n, nsamples, **kwargs):
    """
    Generate a latin-hypercube design

    @param n : int
        The number of factors to generate nsamples for
    @param nsamples:
        the number of samples
    @param criterion : str
        Allowable values are "center" or "c", "maximin" or "m",
        "centermaximin" or "cm", and "correlation" or "corr". If no value
        given, the design is simply randomized.
    @param niters: int
        number of iterations until FLAG check
        (Default: 10^2)
    @param bounds: n x 2 numpy array
        end points for each factor
        (Default: [0,1]^n)
    @param weight_matrix : n x k numpy array
        Weighs the factors appropriately in the maximin optimization
        (Default: identity matrix)
    @param t0: float
        initial temperature
        (Default: 0.9).
    @param FAC: float
        multiplicative factor to reduce temperature
        (Default: 5).
    @param plot: bool
        boolean to generate plots
        (Default: False).
    @param maxtotaliters : int
        Maximum number of iterations
        (Default: 10^4).
    @return H : 2d-array
        An n-by-nsamples design matrix that has been normalized so factor values
        are uniformly spaced between zero and one.
    """
    H = None

    if 'criterion' in kwargs.keys():
        criterion = kwargs['criterion']
    else:
        criterion = None

    if 'niters' in kwargs.keys():
        niters = kwargs['niters']
    else:
        niters = 1e2

    if 'bounds' in kwargs.keys():
        bounds = kwargs['bounds']
    else:
        bounds = np.zeros((n,2))
        bounds[:,1]= 1.

    if 'weight_matrix' in kwargs.keys():
        weight_matrix = kwargs['weight_matrix']
    else:
        weight_matrix = np.eye(n)

    if criterion.lower() in ('maximin', 'm','centermaximin', 'cm'):
        if 't0' in kwargs.keys():
            t0 = kwargs['t0']
        else:
            t0 = 0.9

        if 'FAC' in kwargs.keys():
            FAC = kwargs['FAC']
        else:
            FAC = 0.95

        if 'plot' in kwargs.keys():
            plot = kwargs['plot']
        else:
            plot = False

        if 'maxtotaliters' in kwargs.keys():
            maxtotaliters = kwargs['maxtotaliters']
        else:
            maxtotaliters = 1e4

    if criterion is not None:
        assert criterion.lower() in ('center', 'c', 'maximin', 'm',
                                     'centermaximin', 'cm'), 'Invalid value for "criterion": {}'.format(criterion)

        if criterion.lower() in ('center', 'c'):
            H = _lhscentered(n, nsamples, bounds)
        elif criterion.lower() in ('maximin', 'm'):
            H = _lhsmaximin(n, nsamples, bounds, niters, weight_matrix, 'maximin',t0, FAC, maxtotaliters,plot)
        elif criterion.lower() in ('centermaximin', 'cm'):
            H = _lhsmaximin(n, nsamples, bounds, niters, weight_matrix, 'centermaximin',t0, FAC, maxtotaliters,plot)
    else:
        H = _lhsclassic(n, nsamples, bounds)

    if plot:
        if H.shape[1] == 2:
            plt.scatter(H[:, 0], H[:, 1])
            plt.title("Plot of the two dimensional input space")
            plt.xlabel("x1")
            plt.ylabel("x2")
            plt.show()
        elif H.shape[1] == 3:
            fig = plt.figure()
            ax = Axes3D(fig)
            ax.scatter(H[:, 0], H[:, 1], H[:, 2])
            plt.title("Plot of the three dimensional input space")
            ax.set_xlabel("x1")
            ax.set_ylabel("x2")
            ax.set_zlabel("x3")
            plt.show()

        y = np.matmul(weight_matrix.T, H.T).T
        if y.shape[1] == 1:
            plt.scatter(y[:, 0], [0] * y.shape[0])
            plt.title("Plot of the coordinate points in the one dimensional active subspace")
            plt.xlabel("y1")
            plt.show()
        elif y.shape[1] == 2:
            plt.scatter(y[:, 0], y[:, 1])
            plt.title("Plot of the coordinate points in the two dimensional active subspace")
            plt.xlabel("y1")
            plt.ylabel("y2")
            plt.show()
        elif y.shape[1] == 3:
            fig = plt.figure()
            ax = Axes3D(fig)
            ax.scatter(y[:, 0], y[:, 1], y[:, 2])
            plt.title("Plot of the coordinate points in the three dimensional active subspace")
            ax.set_xlabel("y1")
            ax.set_ylabel("y2")
            ax.set_zlabel("y3")
            plt.show()
    return H


################################################################################

def _lhsclassic(n, nsamples, bounds):
    """
    Generates Latin hypercube nsamples of R^n points
    @param n: int
        dimensions of points
    @param nsamples: int
        number of points
    @param bounds: n x 2 numpy array
        bounds of region
    @return H: nsamples x n matrix of points
    """
    # Fill points uniformly in each interval
    u = np.random.rand(nsamples, n)
    rdpoints = np.zeros_like(u)
    for j in range(n):
        # Generate the intervals
        cut = np.linspace(bounds[j,0], bounds[j,1], nsamples + 1)
        a = cut[:nsamples]
        b = cut[1:nsamples + 1]
        rdpoints[:, j] = u[:, j] * (b - a) + a

    # Make the random pairings
    H = np.zeros_like(rdpoints)
    for j in range(n):
        order = np.random.permutation(range(nsamples))
        H[:, j] = rdpoints[order, j]
    return H


################################################################################

def _lhscentered(n, nsamples, bounds):
    """
    Generates Latin hypercube nsamples of R^n points without random noise
    @param n: int
        dimensions of points
    @param nsamples: int
        number of points
    @param bounds: n x 2 numpy array
        bounds of region
    @return H: nsamples x n matrix of points
    """
    # Fill points uniformly in each interval
    u = np.random.rand(nsamples, n)
    # Make the random pairings
    H = np.zeros_like(u)
    for j in range(n):
        # Generate the intervals
        cut = np.linspace(bounds[j,0], bounds[j,1], nsamples + 1)
        a = cut[:nsamples]
        b = cut[1:nsamples + 1]
        _center = (a + b) / 2
        H[:, j] = np.random.permutation(_center)
    return H


################################################################################

def _lhsmaximin(n, nsamples, bounds, niters, weight_matrix, lhstype='maximin',
                t0=0.05,FAC=0.95,maxiters = 1e5, plot=False):
    """
    Generates Latin hypercube nsamples of R^n points without random noise
    @param n: int
        dimensions of the points
    @param nsamples: int
        number of points
    @param bounds: n x 2 numpy array
        end points for each factor
    @param niters: int
        number of iterations until FLAG check
    @param weight_matrix:
        Weighs the factors appropriately in the maximin optimization
    @param lhstype: str
        'maximin' or 'centermaximin'
    @param t0: float
        initial temperature
        (Default: 0.9).
    @param FAC: float
        multiplicative factor to reduce temperature
        (Default: 5).
    @param plot: bool
        boolean to generate plots
        (Default: 5).
    @param maxtotaliters: int
        Maximum number of iterations
        (Default: 5).
    @return:
    """
    #initial sample
    if lhstype == 'maximin':
        Hcandidate = _lhsclassic(n, nsamples, bounds)
    else:
        Hcandidate = _lhscentered(n, nsamples, bounds)

    min_dist = np.min(_pdist(Hcandidate,weight_matrix))
    curr_dist = min_dist

    # Maximize the minimum distance between points using point exchange
    Hbest = Hcandidate
    min_dist_array = []
    trans_dist_array = []

    t = t0
    FLAG = 0
    iter = 0
    totaliters = 0
    while( iter < niters):
        # random sample
        j = np.random.randint(0,n)
        i,m = np.random.choice(range(nsamples),size=2,replace=False)

        # swap
        temp_m_j = Hcandidate[m, j]
        temp_i_j = Hcandidate[i, j]
        Hcandidate[i,j] = temp_m_j
        Hcandidate[m,j] = temp_i_j

        #calculate new min distance
        temp_min_dist = np.min(_pdist(Hcandidate,weight_matrix))
        # calculate acceptance probability
        u = min(np.exp(-(curr_dist-temp_min_dist)/t),1)

        # acceptance or rejection update
        if u > np.random.uniform(size=1)[0]:
            FLAG = 1
            t = t * FAC
            curr_dist = temp_min_dist
        else:
            Hcandidate[i, j] = temp_i_j
            Hcandidate[m, j] = temp_m_j

        if min_dist < temp_min_dist:
            Hbest = Hcandidate
            min_dist = temp_min_dist
            iter = 1
        else:
            iter += 1

        # check FLAG and number of iterations
        # temp update -- possibly successful but stagnated walk
        if iter == niters and FLAG:
            t = t*FAC
            iter = 1
            FLAG = 0

        trans_dist_array.append(curr_dist)
        min_dist_array.append(min_dist)
        totaliters +=1
        if maxiters < totaliters:
            break


    if plot:
        plt.plot(min_dist_array)
        plt.title("Maximum minimum distance during random walk")
        plt.ylabel("Minimum distance between points")
        plt.xlabel("Step index")
        plt.show()
    return Hbest


################################################################################

def _pdist(x,weight_matrix):
    """
    Calculate the pair-wise point distances of a matrix
    @param x: 2d-array
        An m-by-n array of scalars, where there are m points in n dimensions.
    @param weight_matrix : 2d-array
        An n-by-k matrix
    @return d: array
        A 1-by-b array of scalars, where b = m*(m - 1)/2. This array contains
        all the pair-wise point distances, arranged in the order (1, 0),
        (2, 0), ..., (m-1, 0), (2, 1), ..., (m-1, 1), ..., (m-1, m-2).
    """

    x = np.atleast_2d(x)
    assert len(x.shape) == 2, 'Input array must be 2d-dimensional'

    y = np.matmul(weight_matrix.T,x.T).T
    m, n = y.shape

    if m < 2:
        return []

    d = []
    for i in range(m - 1):
        for j in range(i + 1, m):
            d.append((sum((y[j, :] - y[i, :]) ** 2)) ** 0.5)

    return np.array(d)

if __name__ == '__main__':
    lhs(2,100,criterion='maximin', niters=int(1e2), FAC=0.9, plot=True,maxtotaliters=1e4)
    lhs(3,100,criterion='maximin', iterations=int(1e2), FAC=0.9, plot=True)


    lhs(2,20,criterion='maximin', iterations=int(5e2),weight_matrix=np.array([[0.7,0.3]]).T, plot=True)

    lhs(3,100,criterion='maximin', iterations=int(5e2),
        weight_matrix=np.array([[0.7,0.3,0.1],[0.1,0.0,-0.7]]).T, plot=True)

    # try with the active subspace matrix
    pickle_data = load_obj("data/1:3/log10/2021_05_07_10:55/sampling_rsampling_N_1000")

    # glycerol after 5 hrs
    cost_mat = pickle_data['FUNCTION_RESULTS']['FINAL_COST_MATRIX'][QOI_NAMES[1]]
    eigs, eigvals = np.linalg.eigh(cost_mat)
    eigenvalues_QoI = np.flip(eigs)
    eigenvectors_QoI = np.flip(eigvals, axis=1)
    print(100*np.cumsum(eigenvalues_QoI)/np.sum(eigenvalues_QoI))
    W1 = eigenvectors_QoI[:, :2]

    lhs(W1.shape[0], 100, bounds = np.array([[-1,1] for i in range(W1.shape[0])]), criterion='maximin',
        niters=int(1e2),weight_matrix=W1, plot=True)

    # maximum 3-HPA
    cost_mat = pickle_data['FUNCTION_RESULTS']['FINAL_COST_MATRIX'][QOI_NAMES[0]]
    eigs, eigvals = np.linalg.eigh(cost_mat)
    eigenvalues_QoI = np.flip(eigs)
    eigenvectors_QoI = np.flip(eigvals, axis=1)
    print(100*np.cumsum(eigenvalues_QoI)/np.sum(eigenvalues_QoI))
    W1 = eigenvectors_QoI[:, :3]

    lhs(W1.shape[0],100,bounds = np.array([[-1,1] for i in range(W1.shape[0])]),criterion='maximin',
        niters=int(1e2),weight_matrix=W1, plot=True)