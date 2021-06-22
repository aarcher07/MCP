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


def lhs(n, bounds= None, samples=None, criterion=None, weight_matrix = None, iterations=None):
    """
    Generate a latin-hypercube design

    Parameters
    ----------
    n : int
        The number of factors to generate samples for

    Optional
    --------
    samples : int
        The number of samples to generate for each factor (Default: n)
    criterion : str
        Allowable values are "center" or "c", "maximin" or "m",
        "centermaximin" or "cm", and "correlation" or "corr". If no value
        given, the design is simply randomized.
    weight_matrix : n x n numpy array
        Weighs the factors appropriately in the maximin optimization
    iterations : int
        The number of iterations in the maximin and correlations algorithms
        (Default: 5).

    Returns
    -------
    H : 2d-array
        An n-by-samples design matrix that has been normalized so factor values
        are uniformly spaced between zero and one.

    Example
    -------
    A 3-factor design (defaults to 3 samples)::

        >>> lhs(3)
        array([[ 0.40069325,  0.08118402,  0.69763298],
               [ 0.19524568,  0.41383587,  0.29947106],
               [ 0.85341601,  0.75460699,  0.360024  ]])

    A 4-factor design with 6 samples::

        >>> lhs(4, samples=6)
        array([[ 0.27226812,  0.02811327,  0.62792445,  0.91988196],
               [ 0.76945538,  0.43501682,  0.01107457,  0.09583358],
               [ 0.45702981,  0.76073773,  0.90245401,  0.18773015],
               [ 0.99342115,  0.85814198,  0.16996665,  0.65069309],
               [ 0.63092013,  0.22148567,  0.33616859,  0.36332478],
               [ 0.05276917,  0.5819198 ,  0.67194243,  0.78703262]])

    A 2-factor design with 5 centered samples::

        >>> lhs(2, samples=5, criterion='center')
        array([[ 0.3,  0.5],
               [ 0.7,  0.9],
               [ 0.1,  0.3],
               [ 0.9,  0.1],
               [ 0.5,  0.7]])

    A 3-factor design with 4 samples where the minimum distance between
    all samples has been maximized::

        >>> lhs(3, samples=4, criterion='maximin')
        array([[ 0.02642564,  0.55576963,  0.50261649],
               [ 0.51606589,  0.88933259,  0.34040838],
               [ 0.98431735,  0.0380364 ,  0.01621717],
               [ 0.40414671,  0.33339132,  0.84845707]])

    A 4-factor design with 5 samples where the samples are as uncorrelated
    as possible (within 10 iterations)::

        >>> lhs(4, samples=5, criterion='correlate', iterations=10)

    """
    H = None

    if samples is None:
        samples = n

    if iterations is None:
        iterations = 5
    if bounds is None:
        bounds = np.zeros((n,2))
        bounds[:,1]= 1.

    if criterion is not None:
        assert criterion.lower() in ('center', 'c', 'maximin', 'm',
                                     'centermaximin', 'cm'), 'Invalid value for "criterion": {}'.format(criterion)

        if criterion.lower() in ('center', 'c'):
            H = _lhscentered(n, bounds, samples)
        elif criterion.lower() in ('maximin', 'm'):
            H = _lhsmaximin(n, bounds, samples, iterations, weight_matrix, 'maximin')
        elif criterion.lower() in ('centermaximin', 'cm'):
            H = _lhsmaximin(n, bounds, samples, iterations, weight_matrix, 'centermaximin')


    else:
        H = _lhsclassic(n, bounds, samples)

    return H


################################################################################

def _lhsclassic(n, bounds, samples):
    # Fill points uniformly in each interval
    u = np.random.rand(samples, n)
    rdpoints = np.zeros_like(u)
    for j in range(n):
        # Generate the intervals
        cut = np.linspace(bounds[j,0], bounds[j,1], samples + 1)
        a = cut[:samples]
        b = cut[1:samples + 1]
        rdpoints[:, j] = u[:, j] * (b - a) + a

    # Make the random pairings
    H = np.zeros_like(rdpoints)
    for j in range(n):
        order = np.random.permutation(range(samples))
        H[:, j] = rdpoints[order, j]

    return H


################################################################################

def _lhscentered(n, bounds, samples):
    # Fill points uniformly in each interval
    u = np.random.rand(samples, n)
    # Make the random pairings
    H = np.zeros_like(u)
    for j in range(n):
        # Generate the intervals
        cut = np.linspace(bounds[j,0], bounds[j,1], samples + 1)
        a = cut[:samples]
        b = cut[1:samples + 1]
        _center = (a + b) / 2
        H[:, j] = np.random.permutation(_center)
    return H


################################################################################

def _lhsmaximin(n, bounds, samples, iterations, weight_matrix, lhstype,t0=0.05,FAC=0.95):
    #initial sample
    if lhstype == 'maximin':
        Hcandidate = _lhsclassic(n, bounds, samples)
    else:
        Hcandidate = _lhscentered(n, bounds, samples)

    if weight_matrix is None:
        weight_matrix = np.eye(n)
    min_dist = np.min(_pdist(Hcandidate,weight_matrix))

    # Maximize the minimum distance between points using point exchange
    min_dist_array = []
    t = t0
    for iter in range(iterations):
        # random sample
        j = np.random.randint(0,n)
        i,m = np.random.choice(range(samples),size=2,replace=False)

        # swap
        temp_m_j = Hcandidate[m, j]
        temp_i_j = Hcandidate[i, j]
        Hcandidate[i,j] = temp_m_j
        Hcandidate[m,j] = temp_i_j

        #calculate new min distance
        temp_min_dist = np.min(_pdist(Hcandidate,weight_matrix))
        # calculate acceptance probability
        u = min(np.exp(-(min_dist-temp_min_dist)/t),1)

        print(temp_min_dist)
        print(min_dist)
        print(u)

        # acceptance or rejection
        # temp update
        if u > np.random.uniform(size=1)[0]:
            min_dist = temp_min_dist
            t = t*FAC
        else:
            Hcandidate[i, j] = temp_i_j
            Hcandidate[m, j] = temp_m_j
        min_dist_array.append(min_dist)

        print(iter)
    if Hcandidate.shape[1] >= 2:
        plt.scatter(Hcandidate[:,0],Hcandidate[:,1])
        plt.show()

    y = np.matmul(weight_matrix.T, Hcandidate.T).T
    print(y.shape)
    if y.shape[1] == 1:
        plt.scatter(y[:, 0], [0]*y.shape[0])
        plt.show()
    elif y.shape[1] == 2:
        plt.scatter(y[:,0],y[:,1])
        plt.show()
    elif y.shape[1] == 3:
        fig = plt.figure()
        ax = Axes3D(fig)
        ax.scatter(y[:,0],y[:,1],y[:,2])
        plt.show()

    plt.plot(min_dist_array)
    plt.show()
    return Hcandidate


################################################################################

def _pdist(x,weight_matrix):
    """
    Calculate the pair-wise point distances of a matrix

    Parameters
    ----------
    x : 2d-array
        An m-by-n array of scalars, where there are m points in n dimensions.
    weight_matrix : 2d-array
        An n-by-k matrix
    Returns
    -------
    d : array
        A 1-by-b array of scalars, where b = m*(m - 1)/2. This array contains
        all the pair-wise point distances, arranged in the order (1, 0),
        (2, 0), ..., (m-1, 0), (2, 1), ..., (m-1, 1), ..., (m-1, m-2).

    Examples
    --------
    ::

        >>> x = np.array([[0.1629447, 0.8616334],
        ...               [0.5811584, 0.3826752],
        ...               [0.2270954, 0.4442068],
        ...               [0.7670017, 0.7264718],
        ...               [0.8253975, 0.1937736]])
        >>> _pdist(x)
        array([ 0.6358488,  0.4223272,  0.6189940,  0.9406808,  0.3593699,
                0.3908118,  0.3087661,  0.6092392,  0.6486001,  0.5358894])

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
    print(lhs(2,samples=20,criterion='maximin', iterations=int(1e3),
              weight_matrix=np.array([[0.7,0.3]]).T).shape)

    print(lhs(3,samples=100,criterion='maximin', iterations=int(1e3),
              weight_matrix=np.array([[0.7,0.3,0.1],[0.1,0.0,-0.7]]).T).shape)

    # try with the active subspace matrix
    pickle_data = load_obj("data/1:3/log10/2021_05_07_10:55/sampling_rsampling_N_1000")
    cost_mat = pickle_data['FUNCTION_RESULTS']['FINAL_COST_MATRIX'][QOI_NAMES[0]]
    eigs, eigvals = np.linalg.eigh(cost_mat)
    eigenvalues_QoI = np.flip(eigs)
    eigenvectors_QoI = np.flip(eigvals, axis=1)
    print(100*np.cumsum(eigenvalues_QoI)/np.sum(eigenvalues_QoI))
    W1 = eigenvectors_QoI[:, :3]
    print(lhs(W1.shape[0],bounds = np.array([[-1,1] for i in range(W1.shape[0])]), samples=100,criterion='maximin',
              iterations=int(1e4),weight_matrix=W1).shape)