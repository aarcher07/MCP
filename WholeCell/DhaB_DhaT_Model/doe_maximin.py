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
from misc import load_obj,save_obj
from constants import QOI_NAMES
from mpl_toolkits.mplot3d import Axes3D

__all__ = ['maximin']

def maximin(n, nsamples, **kwargs):
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


    H = _maximin(n, nsamples, bounds, niters, weight_matrix, t0, FAC, maxtotaliters, plot)

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

def _unifsamp(n, bounds, samples):
    # Fill points uniformly in each interval
    u = np.random.rand(samples, n)
    for i in range(samples):
        # Generate the intervals
        u[i,:] = u[i,:]*(bounds[:,1]- bounds[:,0]) + bounds[:,0]

    return u

################################################################################

def _maximin(n, nsamples, bounds, niters, weight_matrix,t0,FAC,maxtotaliters,plot):
    #initial sample
    Hcandidate = _unifsamp(n, bounds, nsamples)
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
        pointsamp = _unifsamp(n, bounds, 1)
        i= np.random.choice(range(nsamples),size=1)

        # swap
        point_i = Hcandidate[i,:]
        Hcandidate[i,:] = pointsamp[0,:]

        #calculate new min distance
        temp_min_dist = np.min(_pdist(Hcandidate,weight_matrix))
        # calculate acceptance probability
        u = min(np.exp(-(curr_dist-temp_min_dist)/t),1)
        # acceptance or rejection update
        if u > np.random.uniform(size=1)[0]:
            FLAG = 1
            trans_dist_array.append(temp_min_dist)
            t = t * FAC
            curr_dist = temp_min_dist
        else:
            Hcandidate[i, :] = point_i

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

        min_dist_array.append(min_dist)
        totaliters += 1

        if maxtotaliters < totaliters:
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
    # maximin(2,100, niters=int(1e2), FAC=0.9, plot=True,maxtotaliters=1e4)
    # maximin(3,100, iterations=int(1e2), FAC=0.9, plot=True)
    #
    #
    # maximin(2,20, iterations=int(5e2),weight_matrix=np.array([[0.7,0.3]]).T, plot=True)
    #
    # maximin(3,100, iterations=int(5e2),weight_matrix=np.array([[0.7,0.3,0.1],[0.1,0.0,-0.7]]).T,
    #     plot=True)

    # try with the active subspace matrix
    folder_name = "data/1:18/log10/2021_06_29_10:00"
    pickle_data = load_obj(folder_name + "/sampling_rsampling_N_100000")

    active_coordinates = {
        QOI_NAMES[0]:[],
        QOI_NAMES[1]:[],
        QOI_NAMES[2]:[]
    }
    eig_max = {
        QOI_NAMES[0]:None,
        QOI_NAMES[1]:None,
        QOI_NAMES[2]:None
    }
    for i in range(3):
        cost_mat = pickle_data['FUNCTION_RESULTS']['FINAL_COST_MATRIX'][QOI_NAMES[i]]
        eigs, eigvals = np.linalg.eigh(cost_mat)
        eigenvalues_QoI = np.flip(eigs)
        eigenvectors_QoI = np.flip(eigvals, axis=1)
        eig_max_ind = np.argmax(np.cumsum(eigenvalues_QoI)/np.sum(eigenvalues_QoI) > 0.9) +1
        print(eig_max_ind)
        eig_max[QOI_NAMES[i]] = eig_max_ind
        W1 = eigenvectors_QoI[:, :eig_max_ind]
        if eig_max_ind == 1:
            nsamples = 30
        elif eig_max_ind == 2:
            nsamples = 100
        elif eig_max_ind == 3:
            nsamples = 200
        else:
            nsamples = 400
        y = maximin(W1.shape[0], nsamples, bounds = np.array([[-1,1] for i in range(W1.shape[0])]),niters=int(1e2),
            weight_matrix=W1, plot=True,maxtotaliters=1e6)
        active_coordinates[QOI_NAMES[i]] = y
    print(active_coordinates)
    save_obj([eig_max,active_coordinates], folder_name + "/active_coords_maximin_sampling_rsampling_N_1000")