import time
from skopt.space import Space
from skopt.sampler import Lhs
from sklearn.model_selection import train_test_split
from mpi4py import MPI
from ActiveLearning import DhaBDhaTModelActiveLearning
import numpy as np
from base_dhaB_dhaT_model.data_set_constants import INIT_CONDS_GLY_PDO_DCW
from base_dhaB_dhaT_model.model_constants import QoI_PARAMETER_LIST
from base_dhaB_dhaT_model.misc_functions import save_obj
import sys
from os.path import dirname, abspath
ROOT_PATH = dirname(abspath(__file__))

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

def generate_training_data(transform, n_samples, tol=1e-7):
    """
    Generate the generate training and test set using latin hypercube for active learning
	@param transform: parameter distribution -- log uniform and log normal
	@param n_samples: number of parameter samples for generate training set
	@param tol: integration tolerance

	@return input_train: parameter samples of length 4*(0.7*n_samples) with initial conditions
	@return input_test: parameter samples of length 4*(0.3*n_samples) with initial conditions
	@return ftrain: QoI evaluations of Glycerol, 1,3-PDO, DCW for samples in input_train
	@return ftest: QoI evaluations of Glycerol, 1,3-PDO, DCW for samples in input_test
	"""
    dhaB_dhaT_model = DhaBDhaTModelActiveLearning(transform=transform)

    if rank == 0:
        # generate initial parameters and evaluation data
        space = Space([(0., 1.) for _ in range(len(QoI_PARAMETER_LIST))])
        lhs = Lhs(lhs_type="classic", criterion='maximin')
        input_params = lhs.generate(space.dimensions, n_samples)
        test_size = 0.3
        input_train_prop, input_test_prop = train_test_split(input_params, test_size=test_size, random_state=1)

        # split data
        n_samples_training_rank0 = np.shape(input_train_prop)[0] // size + np.shape(input_train_prop)[0] % size
        count_training_scatter = [n_samples_training_rank0]
        n_samples_test_rank0 = np.shape(input_test_prop)[0] // size + np.shape(input_test_prop)[0] % size
        count_test_scatter = [n_samples_test_rank0]
        if size > 2:
            n_samples_training_nonrank0 = np.shape(input_train_prop)[0] // size
            count_training_scatter.extend((size - 2) * [n_samples_training_nonrank0])
            count_training_scatter = np.cumsum(count_training_scatter)

            n_samples_test_nonrank0 = np.shape(input_test_prop)[0] // size
            count_test_scatter.extend((size - 2) * [n_samples_test_nonrank0])
            count_test_scatter = np.cumsum(count_test_scatter)
        input_train_prop_split = np.split(input_train_prop, count_training_scatter)
        input_test_prop_split = np.split(input_test_prop, count_test_scatter)
    else:
        input_train_prop_split = None
        input_test_prop_split = None

    # scatter training data
    input_train_prop_rank = comm.scatter(input_train_prop_split, root=0)
    input_test_prop_rank = comm.scatter(input_test_prop_split, root=0)
    input_train_rank, input_test_rank = [], []
    ftrain_rank, ftest_rank = [], []

    # generate data array
    for explan_set_prop, (explan_set, response_data) in zip([input_train_prop_rank, input_test_prop_rank],
                                                            zip([input_train_rank, input_test_rank],
                                                                [ftrain_rank, ftest_rank])):
        for unif_param in explan_set_prop:
            try:
                response_data.append(dhaB_dhaT_model.QoI_all_exp(unif_param, tol=tol))
                explan_set.append(unif_param)
            except TypeError:
                continue

    # format output data
    if len(ftrain_rank) > 0:
        ftrain_rank = np.concatenate(ftrain_rank)
        ftest_rank = np.concatenate(ftest_rank)

    # gather data
    input_train = comm.gather(input_train_rank, root=0)
    input_test = comm.gather(input_test_rank, root=0)
    ftrain = comm.gather(ftrain_rank, root=0)
    ftest = comm.gather(ftest_rank, root=0)

    if rank == 0:
        # concatenate
        input_train = np.concatenate([x for x in input_train if x != []])
        input_test = np.concatenate([x for x in input_test if x != []])
        ftrain = np.concatenate([x for x in ftrain if isinstance(x, np.ndarray)])
        ftest = np.concatenate([x for x in ftest if isinstance(x, np.ndarray)])

        # add input data and initial conditions
        input_train = [np.concatenate((param, cond)) for param in input_train for cond in
                       INIT_CONDS_GLY_PDO_DCW.values()]
        input_test = [np.concatenate((param, cond)) for param in input_test for cond in INIT_CONDS_GLY_PDO_DCW.values()]
        input_train = np.array(input_train)
        input_test = np.array(input_test)
        return input_train, input_test, ftrain, ftest

def main(argv, arc):
    # generate data
    ds = argv[1]
    n_samples = int(float(argv[2]))
    tol = float(argv[3])
    data = generate_training_data(ds, n_samples, tol=1e-7)
    if rank == 0:
        # generate folder name
        folder_name =ROOT_PATH + "/output/active_learning_data"
        date_string = time.strftime("%Y_%m_%d_%H:%M")
        file_name = "transform_" + ds + "_nsamples_" + str(n_samples) + "_tol_" + "{:.0e}".format(
            tol) + "_date_" + date_string

        # save object
        save_obj(data, folder_name + "/" + file_name)


if __name__ == '__main__':
    main(sys.argv, len(sys.argv))


