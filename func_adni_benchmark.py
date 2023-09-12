import pandas as pd
import numpy as np
import torch
import csv, os
import torch.distributions as td
from datetime import datetime
from pathlib import Path

from sklearn.impute import SimpleImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import BayesianRidge

def mse(xhat,xtrue,mask,normalized=False): # MSE function for imputations
    xhat = np.array(xhat)
    xtrue = np.array(xtrue)
    MSE = np.mean(np.power(xhat-xtrue,2)[~mask])
    NMSE = MSE/np.mean(np.power(xtrue,2)[~mask])
    return NMSE if normalized else MSE

def mse_old(xhat,xtrue,mask): # MSE function for imputations
    xhat = np.array(xhat)
    xtrue = np.array(xtrue)
    return np.mean(np.power(xhat-xtrue,2)[~mask])


def testing_func(data_missing, data_full, mask, method,
                 mean = None, std = None, imp='mean'):

    #features = data_full.columns.values.tolist()
    xhat = np.copy(data_missing)
    xhat_0 = np.copy(data_missing)
    xfull = np.copy(data_full)

    p = data_full.shape[1] # number of features
    
    xhat[~mask] = method.transform(xhat)[~mask]
    
    if imp=='mice':
        xhat = np.copy(data_missing)
        imputations = method.transform(xhat)
        nb_imp = 49
        for i in range(nb_imp):
            imputations += method.transform(xhat)
        xhat[~mask] = imputations[~mask]/(nb_imp+1)
            

    if ((mean is not None) and (std is not None)):
        if ((type(mean) != np.ndarray) and (type(std) != np.ndarray)):
            mean, std = mean.numpy(), std.numpy()
        xhat_destd = np.copy(xhat)
        xhat_destd = xhat_destd*std + mean
        xfull_destd = np.copy(xfull)
        xfull_destd = xfull_destd*std + mean
        err_standardized = np.array([mse(xhat,xfull,mask)])
        err = np.array([mse(xhat_destd,xfull_destd,mask,normalized=True)])
        normalized = True
        print('MSE (standardized data)',err_standardized)
        print('MSE (de-standardized data)',err)
    else:
        normalized = False
        err = np.array([mse(xhat,xfull,mask)])

    return float(err)

def testing_func_old(data_missing, data_full, mask, method):

    xhat = np.copy(data_missing)
    xfull = np.copy(data_full)
    
    xhat[~mask] = method.transform(xhat)[~mask]
    err = np.array([mse(xhat,xfull,mask)])

    return float(err)

def save_results(result_folder, Split_type,Train_data,Test_data,
                perc_missing_train,perc_missing_test,model,MSE,imputation,tags):

    os.makedirs(result_folder, exist_ok=True) 
    #exp_id = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    output_file_name = 'output_'+tags+'_'+imputation+'.csv' #+str(exp_id)+'_'+str(np.random.randint(9999, dtype=int))+'.csv'
    fieldnames=['Split_type', 'Train_data', 'Test_data', 'perc_missing_train', 
                'perc_missing_test','model', 'MSE', 'imputation']
    if not os.path.exists(result_folder+'/'+output_file_name):
        output = open(result_folder+'/'+output_file_name, "w")
        writer = csv.DictWriter(output, fieldnames=fieldnames, delimiter = ';')
        writer.writeheader()
        output.close()

    # Dictionary to be added
    dict_out={'Split_type': Split_type, 'Train_data': Train_data, 'Test_data': Test_data, 
                'perc_missing_train': perc_missing_train, 'perc_missing_test': perc_missing_test, 
                'model': model, 'MSE': MSE, 'imputation': imputation}

    with open(result_folder+'/'+output_file_name, 'a') as output_file:
        dictwriter_object = csv.DictWriter(output_file, fieldnames=fieldnames, delimiter = ';')
        dictwriter_object.writerow(dict_out)
        output_file.close()

        
def databases(data_folder,Split_type,idx_clients,idx_Test_data,N_cl,root_dir=None):

    if Split_type == 'notiid':
        data_folder += 'ADNI_notiid'
    elif Split_type == 'site_1':
        data_folder += 'ADNI_site_1'
    elif Split_type == 'site_2':
        data_folder += 'ADNI_site_2'

    if root_dir is not None:
        root_dir = Path.home() if root_dir == 'home' else Path.home().joinpath( 'Documents/Federeted/Code_benchmarking', 'dataset' )
        data_folder = root_dir.joinpath(data_folder)
     
    if Split_type == 'site_2':
        Perc_missing = [0.3,0.2,0.4,0.1]
        Perc_missing_test = Perc_missing[idx_Test_data-1]
        Perc_missing.remove(Perc_missing_test)
    else:
        Perc_missing = [0.3 for _ in range(N_cl)]
        Perc_missing_test = 0.3

    Clients_data=[]
    Clients_missing=[]
    for i in idx_clients:
        data_full_file = os.path.join(str(data_folder), "dataset_full_"+str(i)+".csv")
        #data_full_file = data_folder.joinpath("dataset_full_"+str(i)+".csv")
        data_full = pd.read_csv(data_full_file, sep=",",index_col=False)
        Clients_data.append(data_full)
        data_file = os.path.join(str(data_folder),"dataset_"+str(i)+".csv")
        #data_file = data_folder.joinpath("dataset_"+str(i)+".csv")
        data = pd.read_csv(data_file, sep=",",index_col=False)
        Clients_missing.append(data)

    test_file = os.path.join(str(data_folder),"dataset_full_"+str(idx_Test_data)+".csv")
    #test_file = data_folder.joinpath("dataset_full_"+str(idx_Test_data)+".csv")
    data_test = pd.read_csv(test_file, sep=",",index_col=False)
    test_missing_file = os.path.join(str(data_folder),"dataset_"+str(idx_Test_data)+".csv")
    #test_missing_file = data_folder.joinpath("dataset_"+str(idx_Test_data)+".csv")
    data_test_missing = pd.read_csv(test_missing_file, sep=",",index_col=False)

    return Clients_data, Clients_missing, data_test, data_test_missing, Perc_missing, Perc_missing_test
