import numpy as np
import torch
import torch.nn as nn
import argparse
import sys

from func_adni_benchmark import save_results, databases
from func_gain_adni import gain_loss, gain_impute, Discr_Gener_optD_optG, gain_testing_func  
from class_gain_adni import FedMeanStdTrainingPlan, GAINTrainingPlan

from fedbiomed.researcher.experiment import Experiment



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Pipeline fed standardization and imputation ADNI')
    parser.add_argument('--scenario', metavar='-s', type=str, default='notiid', choices = ['site_1', 'site_2', 'notiid'],help='Scenario for data splitting')
    parser.add_argument('--method', metavar='-m', type=str, default='FedGain', choices = ['FedGain', 'Local'],help='Methods for the running experiment')
    parser.add_argument('--Test_id', metavar='-tid', type=int, default=4,
                        help='Id of the Test dataset (between 1 and 4)')
    parser.add_argument('--tags', metavar='-t', type=str, default='traumabase', help='Dataset tags', choices  ['traumabase','traumabase_imp','traumabase_pred'])
    parser.add_argument('--task', metavar='-ts', type=str, default='imputation', choices = ['imputation', 'prediction'],
                        help='Task to be performed with the pipeline')
    parser.add_argument('--data_folder', metavar='-d', type=str, default='datasets/',
                        help='Datasets folder')
    parser.add_argument('--root_data_folder', metavar='-rdf', type=str, default=None, choices=['default'],help='Root directory for data')
    parser.add_argument('--result_folder', metavar='-rf', type=str, default='results',help='Folder cotaining the results csv')
    parser.add_argument('--seed', metavar='-seed', type=int, default=0,help='seed to use')
    parser.add_argument('--hidden', metavar='-h', type=int, default=128,
                        help='Number of hidden units')
    parser.add_argument('--latent', metavar='-d', type=int, default=2,
                        help='Latent dimension')
    parser.add_argument('--K', metavar='-k', type=int, default=50,
                        help='Number of IS during training')
    parser.add_argument('--batch_size', metavar='-bs', type=int, default=48,
                        help='Batch size')
    parser.add_argument('--learning_rate', metavar='-lr', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--Rounds', metavar='-r', type=int, default=100,
                        help='Number of rounds')
    parser.add_argument('--Epochs', metavar='-e', type=int, default=10,
                        help='Number of epochs')
    
    
    args = parser.parse_args()
    
    method = args.method

    Split_type = args.scenario
    idx_Test_data = int(args.Test_id)
    tags = args.tags
    data_folder = args.data_folder
    root_dir = args.root_data_folder
    
        
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
    
    ############################################################################
    num_covariates = 12 if task=='imputation' else 13 #13 14
    ############################################################################

    #Number of partecipating clients
    N_cl = len(dataset_size)
        
    ###########################################################
    # Recover full dataset and test dataset for testing phase #
    ###########################################################

    idx_clients=[*range(1,N_cl+2)]
    idx_clients.remove(idx_Test_data)

    Clients_data, Clients_missing, data_test, data_test_missing, Perc_missing, Perc_missing_test = databases(data_folder,Split_type,idx_clients,idx_Test_data,N_cl,root_dir)
    
    
    ###########################################################
    # Recover global mean and std in a federated manner       #
    ###########################################################

    fed_mean, fed_std = None, None

    if method != 'Local':
        
        from fedbiomed.researcher.aggregators.fedstandard import FedStandard

        # NOTE: we need to perform only 1 round of 1 epoch to recover global mean and std
        model_args = {'n_features':data_size}

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
        
        fed_mean = fed_mean_std.aggregated_params()[0]['params']['mean']
        fed_std = fed_mean_std.aggregated_params()[0]['params']['std']
        
    
    ###########################################################
    #Define the hyperparameters for miwae                     #
    ###########################################################

    h = args.hidden # number of hidden units in (same for all MLPs)
    d = args.latent # dimension of the latent space
    K = args.K # number of IS during training

    n_epochs=args.Epochs
    batch_size = args.batch_size
    #num_updates = int(np.ceil(min_n_samples/batch_size)*n_epochs)
    rounds = args.Rounds

    
    
    ###TO COMPLETE
        
        