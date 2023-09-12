import numpy as np
import torch

import pandas as pd
import torch
import torch.nn as nn
import csv, os
import torch.distributions as td
from datetime import datetime
from pathlib import Path



def gain_loss(G,D,data,mask,it):


    p_hint = 0.9 ###BY DEFAULT

    m = data.shape[0] #mb_size in the original code
    n = data.shape[1] #self.Dim in the original code
    Z_mb = np.random.uniform(0., 1, size=[m, n])

    X_mb = data
    M_mb = mask


    def sample_M(m, n, p):
        A = np.random.uniform(0., 1., size=[m, n])
        B = A > p
        C = 1. * B
        return C

    H_mb1 = sample_M(m, n, 1 - p_hint)


    H_mb = M_mb * H_mb1

    New_X_mb = M_mb * X_mb + (1 - M_mb) * Z_mb


    def compute_D_loss(G,D,M,New_X,H):
        M = torch.tensor(np.float64(M)) 
        New_X = torch.tensor(np.float64(New_X)) 
        inputs = torch.cat(dim=1, tensors=(New_X, M))
        G_sample = G(inputs)
        Hat_New_X = New_X * M + G_sample * (1 - M)
        inputs = torch.cat(dim=1, tensors=(Hat_New_X, H))
        D_prob = D(inputs)
        D_loss = -torch.mean(M * torch.log(D_prob + 1e-8) + (1 - M) * torch.log(1. - D_prob))
        return D_loss

    ##############
    ###TorchTrainingPlan _train_over_batch
    ##############

    def compute_G_loss(G,D,X,M,New_X,H,it):
        inputs = torch.cat(dim=1, tensors=(New_X, M))
        G_sample = G(inputs)
        Hat_New_X = New_X * M + G_sample * (1 - M)

        # Discriminator
        inputs = torch.cat(dim=1, tensors=(Hat_New_X, H))
        D_prob = D(inputs)

        # %% Loss
        G_loss1 = -torch.mean((1 - M) * torch.log(D_prob))

        MSE_train_loss = torch.mean((M * New_X - M * G_sample) ** 2) / torch.mean(M)

        alpha = 10 ###BY DEFAULT

        d_lr_decay_step = 200 ###BY DEFAULT: D learning rate decay after N step
        if (it + 1) % d_lr_decay_step == 0:
            alpha = alpha * 0.9

        G_loss = G_loss1 + alpha * MSE_train_loss

        # %% MSE Performance metric
        MSE_test_loss = torch.mean(((1 - M) * X - (1 - M) * G_sample) ** 2) / torch.mean(1 - M)
        RMSE_test_loss = torch.sqrt(MSE_test_loss)

        return G_loss, alpha * MSE_train_loss, RMSE_test_loss

    D_loss_curr = compute_D_loss(G, D, M=M_mb, New_X=New_X_mb, H=H_mb)

    G_loss_curr, MSE_train_loss_curr, MSE_test_loss_curr = compute_G_loss(G, D, X=X_mb, M=M_mb,New_X=New_X_mb, H=H_mb,it=it)
    return D_loss_curr, G_loss_curr



def gain_impute(G,data,mask):
    
    if type(data)==np.ndarray:
        data=torch.from_numpy(data).float()
        mask=torch.from_numpy(mask).float()
    
    m = data.shape[0] #mb_size in the original code
    n = data.shape[1] #self.Dim in the original code
    Z = np.random.uniform(0., 1, size=[m, n])
    
    Z_tilde = mask * data + (1 - mask) * Z
    
    with torch.no_grad():
        inputs = torch.cat(dim=1, tensors=(Z_tilde, mask))
        X_tilde = G(inputs)
    
    X_imp = mask * data + (1 - mask) * X_tilde

    return X_imp


def gain_impute_fed(G,data,mask):
    
    if type(data)==np.ndarray:
        data=torch.from_numpy(data).float()
        mask=torch.from_numpy(mask).float()
    
    m = data.shape[0] #mb_size in the original code
    n = data.shape[1] #self.Dim in the original code
    Z = np.random.uniform(0., 1, size=[m, n])
    
    Z_tilde = mask * data + (1 - mask) * Z
    
    with torch.no_grad():
        X_tilde = G(Z_tilde,mask)
    
    X_imp = mask * data + (1 - mask) * X_tilde

    return X_imp



def Discr_Gener_optD_optG(hidden_dim,output_size,fed_d_lr,fed_g_lr):

    input_size = output_size

    D = nn.Sequential(
            nn.Linear(input_size * 2, hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_dim//2, hidden_dim // 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_dim//2, input_size),
            nn.Sigmoid()).double()

    G = nn.Sequential(
            nn.Linear(input_size+output_size, hidden_dim//2),
            nn.ReLU(True).double(),
            nn.Linear(hidden_dim//2, hidden_dim//2),
            nn.ReLU(True),
            nn.Linear(hidden_dim//2, output_size),
            nn.Sigmoid()).double()

    optimizer_D = torch.optim.Adam(params=D.parameters(), lr=fed_d_lr)
    optimizer_G = torch.optim.Adam(params=G.parameters(), lr=fed_g_lr)
    
    def xavier_init(size):
        in_dim = size[0]
        xavier_stddev = 1. / np.sqrt(in_dim / 2.)
        return np.random.normal(size=size, scale=xavier_stddev)

    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv2d') != -1:
            nn.init.xavier_normal_(m.weight.data)
            nn.init.constant_(m.bias.data, 0.0)
        elif classname.find('Linear') != -1:
            nn.init.xavier_normal_(m.weight)
            nn.init.constant_(m.bias, 0.0)

    G.apply(weights_init)
    D.apply(weights_init)   


    return(D, G, optimizer_D, optimizer_G)


def mse(xhat,xtrue,mask,normalized=False): # MSE function for imputations
    xhat = np.array(xhat)
    xtrue = np.array(xtrue)
    MSE = np.mean(np.power(xhat-xtrue,2)[~mask])
    NMSE = MSE/np.mean(np.power(xtrue,2)[~mask])
    return NMSE if normalized else MSE


def gain_testing_func(data_imp0, data_full, mask, G,
                 mean = None, std = None, imp='GAIN'):
    
    xhat = np.copy(data_imp0)
    xfull = np.copy(data_full)

    p = data_full.shape[1] # number of features
    
    if imp=='FedGAIN':
        xhat[~mask] = gain_impute_fed(G,xhat,mask)[~mask]
    else:
        xhat[~mask] = gain_impute(G,xhat,mask)[~mask]
    
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
    