import time
from skopt.space import Space
from skopt.sampler import Lhs
from mpi4py import MPI
from ActiveLearning import DhaBDhaTModelActiveLearning
from ActiveLearning.build_separable_GP import *
from base_dhaB_dhaT_model.data_set_constants import NPARAMS, INIT_CONDS_GLY_PDO_DCW
from base_dhaB_dhaT_model.model_constants import QoI_PARAMETER_LIST
from base_dhaB_dhaT_model.misc_functions import load_obj,save_obj
import sys
from os.path import dirname, abspath
ROOT_PATH = dirname(abspath(__file__))

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

def active_learning_train_GP(explan_train,respon_train,explan_test,respon_test,
                             max_training_length,ninitial = 100,tol=1e-7):

    ######################################################################################################
    ######################################### DO INITIAL FIT #############################################
    ######################################################################################################
    dhaB_dhaT_model = DhaBDhaTModelActiveLearning(transform="log_unif")
    if rank == 0:
        fitted_info = fitGP(explan_train, respon_train, init_logeta=0*np.ones(NPARAMS+2),
                            lowerb=-np.ones(NPARAMS+2), upperb=np.ones(NPARAMS+2))
        #get GP data
        predinfo = predictGP(fitted_info, explan_test, explan_train, respon_train)
        predmean = predinfo['pred_mean']

        #compute RMSE
        rmse = np.mean((predmean-respon_test.flatten('F'))**2)
        rmse_array = []
        rmse_array.append(rmse)

        #latin hypercube sampling
        space = Space([(0., 1.) for _ in range(len(QoI_PARAMETER_LIST))])
        lhs = Lhs(lhs_type="classic", criterion='maximin')
    else:
        fitted_info = None

    fitted_info = comm.bcast(fitted_info,root=0)

    while(explan_train.shape[0] < 4*max_training_length):

        ######################################################################################################
        ######################################## PROBLEM SETUP ###############################################
        ######################################################################################################

        etahat = fitted_info['etahat']
        etahat1 = etahat[:NPARAMS]
        sigmahat = fitted_info['sigmahat']

        #create correlation matrix
        corr_tr = corr(explan_train, explan_train, etahat1)

        #generate search parameter set
        if rank == 0:
            params_init = lhs.generate(space.dimensions, ninitial)
            #scatter sizes
            ninitial_rank0 =ninitial // size + ninitial % size
            count_scatter = [ninitial_rank0]
            if size > 2:
                ninitial_nonrank0 = ninitial // size
                count_scatter.extend((size - 2) * [ninitial_nonrank0])
                count_scatter = np.cumsum(count_scatter)
            params_init_split = np.split(params_init, count_scatter)
        else:
            params_init_split = None

        #scatter data and parameters
        params_init = comm.scatter(params_init_split, root=0)

        # neg entropy
        def neg_log_det_fun(param_new):
            p_new = np.array([np.concatenate((param_new,cond)) for cond in INIT_CONDS_GLY_PDO_DCW.values()])
            corr_tr_new = corr(explan_train, p_new, etahat1)
            corr_new_new = corr(p_new, p_new, etahat1)
            corr_prods = np.matmul(corr_tr_new.T,np.linalg.solve(corr_tr,corr_tr_new))
            cov_mat = sigmahat*(corr_new_new - corr_prods)
            return -np.log(np.linalg.det(cov_mat))

        ######################################################################################################
        #################################### CHOOSE DESIGN POINT #############################################
        ######################################################################################################
        rmse_prev = np.inf
        unif_param_prev = None
        y_prev = None
        fitted_info_prev = None

        for i in range(params_init.shape[0]):
            # find proposed point
            unif_param_prop = minimize(neg_log_det_fun, params_init[i], method="L-BFGS-B",
                                  bounds= [[0.,1.] for _ in range(len(QoI_PARAMETER_LIST))],
                                  options={'maxiter': 10**3}).x
            # store proposed point
            unif_param_prop_full = np.array([np.concatenate((unif_param_prop,cond)) for cond in INIT_CONDS_GLY_PDO_DCW.values()])
            explan_train_prop = np.concatenate((explan_train,unif_param_prop_full))


            # generate data
            try:
                y_prop = dhaB_dhaT_model.QoI_all_exp(unif_param_prop,tol = tol)
            except TypeError:
                continue

            # store response data
            respon_train_prop = np.concatenate((respon_train, y_prop))
            fitted_info_prop = fitGP(explan_train_prop, respon_train_prop, init_logeta=np.log(etahat),
                                    lowerb=-np.ones(NPARAMS+2), upperb=np.ones(NPARAMS+2))

            # compute rmse test of new potential fit
            predinfo_prop = predictGP(fitted_info_prop, explan_test, explan_train_prop, respon_train_prop)
            predmean_prop = predinfo_prop['pred_mean']
            rmse_prop = np.mean((predmean_prop-respon_test.flatten('F'))**2)

            # compare with previous fit
            if rmse_prop < rmse_prev:
                rmse_prev = rmse_prop
                unif_param_prev = unif_param_prop
                y_prev = y_prop
                fitted_info_prev = fitted_info_prop

        #gather and bcast data with smallest rmse
        rmse_prev_array = comm.allgather(rmse_prev)
        rmse_min_ind = np.argmin(rmse_prev_array)
        unif_param_prev = comm.bcast(unif_param_prev, root=rmse_min_ind)
        y_prev = comm.bcast(y_prev, root=rmse_min_ind)

        # generate full training set
        unif_param_prop_full = np.array([np.concatenate((unif_param_prev,cond)) for cond in INIT_CONDS_GLY_PDO_DCW.values()])
        explan_train = np.concatenate((explan_train,unif_param_prop_full))
        respon_train = np.concatenate((respon_train, y_prev))
        fitted_info = comm.bcast(fitted_info_prev, root=rmse_min_ind)

        #store rmse
        if rank == 0:
            rmse_array.append(rmse_prev_array[rmse_min_ind])

    if rank == 0:
        dict_data = {'explan_training_set': explan_train,
                     'respon_train': respon_train,
                     'fitted_info': fitted_info,
                     'rmse_array': rmse_array}
        return dict_data

def main(argv, arc):
    #load file name
    data_file_name = argv[1]
    max_training_length = int(float(argv[2]))

    if len(argv) > 4:
        ninitial = int(argv[3])
    else:
        ninitial = 100

    explan_train,explan_test,respon_train,respon_test = load_obj(ROOT_PATH+"/output/active_learning_data/"
                                                                 + data_file_name)

    # train GP
    time1 = time.time()
    learnt_GP = active_learning_train_GP(explan_train,respon_train,explan_test,respon_test,
                                         max_training_length,ninitial)
    time2 = time.time()
    if rank == 0:
        print(time2-time1)
        date_string = time.strftime("%Y_%m_%d_%H:%M")
        file_name= 'separableGPtraining' + '_nsamples_'+ str(max_training_length) + '_date_'+date_string
        plt.scatter(range(len(learnt_GP['rmse_array'])),learnt_GP['rmse_array'])
        plt.savefig(ROOT_PATH+'/output/active_learning_results/rsme_' + file_name + '.jpg',bbox_inches='tight')
        save_obj(learnt_GP,ROOT_PATH+'/output/active_learning_results/'+ file_name)

if __name__ == '__main__':
    main(sys.argv, len(sys.argv))
