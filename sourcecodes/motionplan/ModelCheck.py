#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 22:18:20 2020

@author: hiroyasu
"""

import torch
from torch import nn
import numpy as np
import TrainRNN as trnn
import SCPmulti as scp
import matplotlib.pyplot as plt

def Np2Var(X):
    X = X.astype(np.float32)
    X = torch.from_numpy(X)
    return X

def preprocessY(Y):
    N = Y.shape[0]
    YC = 0
    for i in range(N):
        Yn = np.linalg.norm(Y[i,:])
        if Yn > YC:
            YC = Yn
            idx = i
    Y = Y/YC
    return Y,YC,idx

n_input = trnn.n_input
n_hidden = trnn.n_hidden
n_output = trnn.n_output
n_layers = trnn.n_layers
DT = scp.DT
TSPAN = scp.TSPAN

if __name__ == "__main__":
    Nrnn = 500
    Xall = np.load('data/training_data/XhisNN.npy')
    cP = np.load('data/training_data/cMhisNN.npy')
    Yall = cP
    Yall,YC,idx = preprocessY(Yall)
    loss_func = nn.MSELoss()
    net = trnn.RNN(n_input,n_hidden,n_output,n_layers)
    net.load_state_dict(torch.load('data/trained_nets/RNNLorenz.pt'))
    net.eval()
    YC =  np.load('data/trained_nets/Y_params.npy')
    t = np.arange(0,50.1,0.1)
    nYs = []
    for s in range(100):
        Xs = Xall[(Nrnn+1)*s:(Nrnn+1)*(s+1),:]
        Xs = Np2Var(Xs)
        Xs = Xs.view(Nrnn+1,1,-1)
        Ys = Yall[(Nrnn+1)*s:(Nrnn+1)*(s+1),:]
        Ys = Np2Var(Ys)
        Ys = Ys.view(Nrnn+1,1,-1)
        Yshat = net(Xs)
        nYs.append(loss_func(Ys,Yshat).data.numpy())
    print(sum(nYs)/100)
    
    Xs = Np2Var(Xall)
    Xs = Xs.view(100*(Nrnn+1),1,-1)
    Ys = Np2Var(Yall)
    Ys = Ys.view(100*(Nrnn+1),1,-1)
    Yshat = net(Xs)
    print(loss_func(Ys,Yshat).data.numpy())