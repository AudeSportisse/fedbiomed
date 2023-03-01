#!/usr/bin/env python
# coding: utf-8

# # Fed-BioMed Researcher to train miwae with adni dataset

# ## Start the network
# Before running this notebook, start the network with `./scripts/fedbiomed_run network`
#
## Start the network and setting the node up
# Before running this notebook, you shoud start the network from fedbiomed-network, as detailed in https://gitlab.inria.fr/fedbiomed/fedbiomed-network
# Therefore, it is necessary to previously configure a node:
# 1. `./scripts/fedbiomed_run node add`
#   * Select option 1 to add a csv file to the node
#   * Choose the name, tags and description of the dataset
#     * use `#test_data`` for the tags
#   * Pick the .csv file from your PC (here: pseudo_adni_mod.csv)
#   * Data must have been added
# 2. Check that your data has been added by executing `./scripts/fedbiomed_run node list`
# 3. Run the node using `./scripts/fedbiomed_run node start`. Wait until you get `Starting task manager`. it means you are online.


# ## Create an experiment to train a model on the data found



# Declare a torch training plan MyTrainingPlan class to send for training on the node
import numpy as np
import pandas as pd
import argparse
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, mean_squared_error

from func_miwae_traumabase import databases, databases_pred, generate_save_plots_prediction, recover_data_prediction, save_results_prediction
from class_miwae_traumabase import FedMeanStdTrainingPlan, SGDRegressorTraumabaseTrainingPlan, FedLogisticRegTraumabase

from fedbiomed.researcher.experiment import Experiment

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Pipeline fed standardization and hemo shock prediction, Traumabase')
    parser.add_argument('--method', metavar='-m', type=str, default='FedAvg', choices = ['FedAvg', 'FedProx', 'FedProx_loc', 'Scaffold', 'Local', 'Centralized'],
                        help='Methods for the running experiment')
    parser.add_argument('--regressor', metavar='-r', type=str, default='linear', choices = ['linear', 'logistic'],
                        help='Methods for the running experiment')
    parser.add_argument('--task', metavar='-ts', type=str, default='prediction', choices = ['imputation', 'prediction'],
                        help='Task to be performed with the pipeline')
    parser.add_argument('--Test_id', metavar='-tid', type=int, default=4,
                        help='Id of the Test dataset (between 1 and 4)')
    parser.add_argument('--tags', metavar='-t', type=str, default='traumabase_pred', help='Dataset tags')
    parser.add_argument('--Rounds', metavar='-r', type=int, default=100,
                        help='Number of rounds for imputation')
    parser.add_argument('--Epochs', metavar='-e', type=int, default=5,
                        help='Number of epochs for imputation')
    parser.add_argument('--data_folder', metavar='-d', type=str, default='data/',
                        help='Datasets folder')
    parser.add_argument('--root_data_folder', metavar='-rdf', type=str, default=None, choices=['fedbiomed','home'],
                        help='Root directory for data')
    parser.add_argument('--result_folder', metavar='-rf', type=str, default='results', 
                        help='Folder cotaining the results csv')
    parser.add_argument('--batch_size', metavar='-bs', type=int, default=48,
                        help='Batch size')
    parser.add_argument('--learning_rate', metavar='-lr', type=float, default=5e-4,
                        help='Learning rate')
    parser.add_argument('--standardize', metavar='-std', default=True, action=argparse.BooleanOptionalAction,
                        help='Standardize data for regression')
    parser.add_argument('--do_fed_std', metavar='-fstd', default=True, action=argparse.BooleanOptionalAction,
                        help='Recover federated mean and std')
    parser.add_argument('--do_figures', metavar='-fig', default=True, action=argparse.BooleanOptionalAction,
                        help='Generate and save figures during local training')

    args = parser.parse_args()

    method = args.method
    task = args.task
    idx_Test_data = int(args.Test_id)
    tags = args.tags
    data_folder = args.data_folder
    root_dir = args.root_data_folder
    regressor = args.regressor

    target_col = ['choc_hemo']

    ###########################################################
    # Recover data size                                       #
    ###########################################################

    from fedbiomed.researcher.requests import Requests
    req = Requests()
    xx = req.list()
    dataset_size = [xx[i][0]['shape'][1] for i in xx]
    min_n_samples = min([xx[i][0]['shape'][0] for i in xx])
    assert min(dataset_size)==max(dataset_size)
    data_size = dataset_size[0]

    num_covariates = 7#13

    if num_covariates == 7:
        regressors_col = ['fracas_du_bassin', 'catecholamines', 'intubation_orotracheale_smur',
                        'sexe','expansion_volemique', 'penetrant', 'age', 'pression_arterielle_systolique_PAS_minimum', 
                        'pression_arterielle_diastolique_PAD_minimum', 'frequence_cardiaque_FC_maximum', 'hemocue_initial']
    elif num_covariates == 13:
        regressors_col = ['fracas_du_bassin_-1.0', 'fracas_du_bassin_0.0', 'fracas_du_bassin_1.0', 
                        'catecholamines_-1.0', 'catecholamines_0.0', 'catecholamines_1.0', 
                        'intubation_orotracheale_smur_-1.0', 'intubation_orotracheale_smur_0.0', 'intubation_orotracheale_smur_1.0', 
                        'sexe', 'expansion_volemique', 'penetrant', 'age', 'pression_arterielle_systolique_PAS_minimum', 
                        'pression_arterielle_diastolique_PAD_minimum', 'frequence_cardiaque_FC_maximum', 'hemocue_initial']

    #Number of partecipating clients
    N_cl = len(dataset_size)

    ###########################################################
    # Recover full dataset and test dataset for testing phase #
    ###########################################################

    idx_clients=[*range(1,N_cl+2)]
    idx_clients.remove(idx_Test_data)

    Clients_data, data_test = databases_pred(data_folder=data_folder,idx_clients=idx_clients,
                                            root_dir=root_dir,idx_Test_data=idx_Test_data, imputed = True)

    ###########################################################
    # Recover global mean and std in a federated manner       #
    ###########################################################

    fed_mean, fed_std = None, None

    if args.standardize:
        if method not in ['FedProx_loc','Local','Centralized']:
            if args.do_fed_std ==  True:

                from fedbiomed.researcher.aggregators.fedstandard import FedStandard

                # NOTE: we need to perform only 1 round of 1 epoch to recover global mean and std
                model_args = {'n_features': data_size, 'n_cov': num_covariates}

                training_args = {
                    'batch_size': 48, 
                    'optimizer_args': {
                        'lr': 0
                    }, 
                    'log_interval' : 1,
                    'num_updates': 1, 
                    'dry_run': False,  
                }

                fed_mean_std = Experiment(tags=tags,
                                model_args=model_args,
                                training_plan_class=FedMeanStdTrainingPlan,
                                training_args=training_args,
                                round_limit=1,
                                aggregator=FedStandard(),
                                node_selection_strategy=None)

                fed_mean_std.run()
                fed_mean = fed_mean_std.aggregated_params()[0]['params']['fed_mean']
                fed_std = fed_mean_std.aggregated_params()[0]['params']['fed_std']

            else:
                npzfile = np.load(args.result_folder+'/clients_imputed/'+method+'_mean_std.npz')
                fed_mean, fed_std = npzfile['mean'], npzfile['std']

    ###########################################################
    #Define the hyperparameters for SGDregressor              #
    ###########################################################

    tol = 1e-5
    eta0 = 0.05
    n_epochs=args.Epochs
    batch_size = args.batch_size
    num_updates = int(np.ceil(min_n_samples/batch_size)*n_epochs)
    rounds = args.Rounds

    ###########################################################
    #Define the federated SGDregressor model                  #
    ###########################################################

    if method not in ['Local','Centralized']:
        model_args = {'n_features': len(regressors_col), 'n_cov': num_covariates-1, 'use_gpu': True, 
                    'regressors_col':regressors_col, 'target_col': target_col, 'tol': tol,'eta0': eta0}
        if args.standardize:
            print('standardization added')
            standardization = {} if method == 'FedProx_loc' else {'fed_mean':fed_mean.tolist(),'fed_std':fed_std.tolist()}
            model_args.update(standardization=standardization)

        training_args = {'batch_size': batch_size, 'num_updates': num_updates, 'dry_run': False}

        if regressor == 'logistic':
            model_args.update(n_classes = 2)
            training_plan = FedLogisticRegTraumabase
        elif regressor == 'linear':
            training_plan = SGDRegressorTraumabaseTrainingPlan

        ###########################################################
        #Declare the experiment                                   #
        ###########################################################

        from fedbiomed.researcher.aggregators.fedavg import FedAverage
        from fedbiomed.researcher.aggregators.scaffold import Scaffold

        aggregator = Scaffold() if method == 'Scaffold' else FedAverage()

        if 'fedprox_mu' in training_args:
            del training_args['fedprox_mu']

        exp = Experiment(tags=tags,
                        model_args=model_args,
                        training_plan_class=training_plan,
                        training_args=training_args,
                        round_limit=rounds,
                        aggregator=aggregator,
                        node_selection_strategy=None)

        exp.run_once()

        if 'FedProx' in method:
            # Starting from the second round, FedProx is used with mu=0.1
            # We first update the training args
            training_args.update(fedprox_mu = 0.1)

            # Then update training args in the experiment
            exp.set_training_args(training_args)

        exp.run()

    ###########################################################
    #Local model                                              #
    ###########################################################    
    elif method == 'Local':

        #TO BE FILLED

        if args.do_figures==True:
            Loss_cls = [[] for _ in range(N_cl)]
            Accuracy_cls = [[] for _ in range(N_cl)]
            MSE_cls = [[] for _ in range(N_cl)]

        # Recall all hyperparameters
        n_epochs_local = n_epochs*rounds
        n_epochs_centralized = n_epochs*rounds*N_cl

        # .................

        Coeff_loc = []
        Intercept_loc = []

        #for cls in range(N_cl):
            
            # .................

        #    for ep in range(1,n_epochs_local):
                
                # .................

            # Append updated coeff, intercept

    elif method == 'Centralized':
        # Centralized training
        #if args.do_figures==True:
        #    Loss_tot = []
        #    Accuracy_tot = []
        #    MSE_tot = []

        Data_tot = pd.concat(Clients_data, ignore_index=True)

        # Training loop

        #for ep in range(1,n_epochs_centralized):

            # .................

    ###########################################################
    #Testing phase (imputation)                               #
    ###########################################################
    result_folder = args.result_folder

    if task == 'prediction':
        X_test = data_test[regressors_col].values
        y_test = data_test[target_col].values.astype(int)
        
        if args.standardize:
            xfull_global_std, xfull_local_std = recover_data_prediction(X_test, num_covariates-1, fed_mean, fed_std)
            X_test=xfull_global_std

        # we create here several instances of SGDRegressor using same sklearn arguments
        # we have used for Federated Learning training
        model_pred = exp.training_plan().model()
        regressor_args = {key: model_args[key] for key in model_args.keys() if key in model_pred.get_params().keys()}

        testing_error = []
        Validation = []

        w0=5/6
        w1=1/6
        for i in range(rounds):
            model_pred.coef_ = exp.aggregated_params()[i]['params']['coef_'].copy()
            model_pred.intercept_ = exp.aggregated_params()[i]['params']['intercept_'].copy()
            y_pred = model_pred.predict(X_test).astype(int)
            mse = np.mean((y_pred - y_test)**2)
            testing_error.append(mse)
            tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
            validation_err = (w0*fn+w1*fp)/(fn+fp)
            Validation.append(validation_err)

        model_pred.coef_ = exp.aggregated_params()[rounds - 1]['params']['coef_'].copy()
        model_pred.intercept_ = exp.aggregated_params()[rounds - 1]['params']['intercept_'].copy() 

        y_pred = model_pred.predict(X_test).astype(int)
        #if regressor == 'logistic':
        #    y_predict_prob = model_pred.predict_proba(X_test)
        #    y_predict_prob_class_1 = y_predict_prob[:,1]
        #    y_pred = [1 if prob > 0.45 else 0 for prob in y_predict_prob_class_1]

        conf_matr = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        accuracy = accuracy_score(y_test, y_pred)
        F1 = f1_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred,zero_division=0)
        mse = mean_squared_error(y_test, y_pred)

        validation_err = (w0*fn+w1*fp)/(fn+fp)    

        save_results_prediction(result_folder, method, regressor, n_epochs, rounds, F1, precision, mse, accuracy, conf_matr, validation_err)

        if args.do_figures==True:
            generate_save_plots_prediction(result_folder,testing_error,Validation,conf_matr,method,regressor)

        print(model_pred.coef_)
        print(model_pred.feature_names_in_)
