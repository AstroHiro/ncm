#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 14 19:47:03 2020

@author: hiroyasu
"""

import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
import random

n_input = 6 # rnn input size
n_hidden = 64 # number of rnn hidden units
n_output = 21 # rnn input size
n_layers = 2 # rnn number of layers

class RNN(nn.Module):
    def __init__(self,n_input,n_hidden,n_output,n_layers):
        super(RNN, self).__init__()
        self.rnn = nn.LSTM(
                input_size=n_input,
                hidden_size=n_hidden, # rnn hidden unit
                num_layers=n_layers, # number of rnn layer
                batch_first=True, # input & output will has batch size as 1s dimension. e.g. (batch, time_step, input_size)
                )
        self.out = nn.Linear(n_hidden,n_output)
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        
    def forward(self,x):
        # x (batch, time_step, input_size)
        # h_state (n_layers, batch, hidden_size)
        # r_out (batch, time_step, hidden_size)
        r_out,_ = self.rnn(x)
        out = self.out(r_out)
        return out
    '''
    def init_hidden(self, batch_size):
        # Initializes hidden state
        # Create two new tensors with sizes n_layers x batch_size x hidden_dim,
        # initialized to zero, for hidden state and cell state of LSTM
        weight = next(self.parameters()).data
        hidden = (weight.new(self.n_layers, batch_size, self.n_hidden).zero_(),weight.new(self.n_layers, batch_size, self.n_hidden).zero_())
        return hidden
    '''
    
def Np2Var(X):
    X = X.astype(np.float32)
    X = torch.from_numpy(X)
    return X

def NCMLoss(Pnet,P):
    loss = 0
    B = P.shape[0] # batch size
    T = P.shape[1] # time step
    for b in range(B):
        Pb = P[b,:,:]
        Pnetb = Pnet[b,:,:]
        for t in range (T):
            Pt = Pb[t,:]
            Pt = Pt.view(-1,3)
            Pnett = Pnetb[t,:]
            Pnett = Pnett.view(-1,3)
            Et = Pt-torch.mm(Pnett,torch.t(Pnett))
            loss += torch.norm(Et,keepdim=True)
    return loss

def GetTrainData(X,Y,N,Nrnn,numX0,BatchSize,time_step):
    # Note that training data = original data - test data
    idx_rnn = random.sample(range(numX0-1),1)
    idx_rnn = idx_rnn[0]
    idx = random.sample(range(Nrnn+1-time_step),BatchSize)
    Xtrain = None
    for i in range(BatchSize):
        Xtraini = X[idx_rnn*(Nrnn+1)+idx[i]:idx_rnn*(Nrnn+1)+idx[i]+time_step,:]
        Ytraini = Y[idx_rnn*(Nrnn+1)+idx[i]:idx_rnn*(Nrnn+1)+idx[i]+time_step,:]
        if i == 0:
            Xtrain = Xtraini
            Ytrain = Ytraini
        else:
            Xtrain = np.vstack((Xtrain,Xtraini))
            Ytrain = np.vstack((Ytrain,Ytraini))
    Xtrain = Np2Var(Xtrain)
    Xtrain = Xtrain.view(BatchSize,time_step,-1)
    Ytrain = Np2Var(Ytrain)
    Ytrain = Ytrain.view(BatchSize,time_step,-1)
    return Xtrain, Ytrain

def RandomizeTrainData(X,Y,N,Nrnn,numX0,time_step):
    # Note that training data = original data - test data
    Ntime = int(Nrnn/time_step)
    Xtrain = np.zeros((Ntime*(numX0-1),time_step,6))
    Ytrain = np.zeros((Ntime*(numX0-1),time_step,21))
    for i in range(numX0-1):
        Xi = X[(Nrnn+1)*i:(Nrnn+1)*(i+1),:]
        Yi = Y[(Nrnn+1)*i:(Nrnn+1)*(i+1),:]
        for j in range(Ntime):
            Xij = Xi[(time_step)*j:(time_step)*(j+1),:]
            Yij = Yi[(time_step)*j:(time_step)*(j+1),:]
            Xtrain[i*Ntime+j,:,:] = Xij 
            Ytrain[i*Ntime+j,:,:] = Yij 
    idx = np.random.permutation(Ntime*(numX0-1))
    Xtrain = Xtrain[idx,:,:]
    Ytrain = Ytrain[idx,:,:]
    return Xtrain, Ytrain

def preprocessX(X,Y):
    N = X.shape[0]
    Xbar = np.sum(X, axis=0)/N
    Xsig = np.sqrt(np.sum((X-Xbar)**2,axis=0)/N)
    Xnor = (X-Xbar)/Xsig
    Ybar = np.sum(Y, axis=0)/N
    Ysig = np.sqrt(np.sum((Y-Ybar)**2,axis=0)/N)
    Ynor = (Y-Ybar)/Ysig
    return Xnor,Ynor

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

if __name__ == "__main__":
    torch.manual_seed(19950808)
    np.random.seed(19950808)
    random.seed(19950808)
    Nrnn = 500
    Xall = np.load('data/training_data/XhisNN.npy')
    cP = np.load('data/training_data/cMhisNN.npy')
    Yall = cP
    #Xall,Yall = preprocessX(Xall,Yall)
    Yall,YC,idx = preprocessY(Yall)
    # Hyper Parameters
    time_step = 10 # rnn time step
    rnn = RNN(n_input,n_hidden,n_output,n_layers)
    LR = 0.2 # learning rate
    N = np.size(Xall,0)
    BatchSize = np.floor(Nrnn/10)
    BatchSize = BatchSize.astype(np.int32)
    numX0 = int(N/(Nrnn+1))
    Xtest = Xall[0:Nrnn+1,:]
    Ytest = Yall[0:Nrnn+1,:]
    Xtest = Np2Var(Xtest)
    Ytest = Np2Var(Ytest)
    Xtest = Xtest.view(Nrnn+1,1,-1)
    Ytest = Ytest.view(Nrnn+1,1,-1)
    X = Xall[Nrnn+1:,:]
    Y = Yall[Nrnn+1:,:]
    #optimizer = torch.optim.Adam(rnn.parameters(),lr=LR) # optimize all cnn parameters
    optimizer = torch.optim.SGD(rnn.parameters(),lr=LR) # optimize all cnn parameters
    loss_func = nn.MSELoss()
    h_state = None # for initial hidden state
    
    Nitrs = 1000000
    e_list = []
    t_loss_list = []
    loss_list = []
    Nepochs = 10000
    
    for epoch in range(Nepochs):
        Xtrain,Ytrain = RandomizeTrainData(X,Y,N,Nrnn,numX0,time_step)
        Ntrain = Xtrain.shape[0]/BatchSize
        Ntrain = Ntrain.astype(np.int32)
        for i in range(Ntrain):
            Xi = Xtrain[(BatchSize)*i:(BatchSize)*(i+1),:,:]
            Yi = Ytrain[(BatchSize)*i:(BatchSize)*(i+1),:,:]
            Xi = Np2Var(Xi)
            Yi = Np2Var(Yi)
            prediction = rnn.forward(Xi) # rnn output
            loss = loss_func(prediction,Yi)
            optimizer.zero_grad() # clear gradients for this training step
            loss.backward() # backpropagation, compute gradients
            optimizer.step() # apply gradients
            # !! next step is important !!
            """
            h_state = rnn.init_hidden(BatchSize)
            h_state[0].detach_()
            h_state[1].detach_()
            """
        e_list.append(epoch)
        Ypred = rnn(Xtest)
        loss_test = loss_func(Ypred,Ytest)
        t_loss_list.append(loss.data.numpy())
        loss_list.append(loss_test.data.numpy())
        print('current loss at epoch = ',epoch,': ',loss_test.data.numpy())
        if (np.absolute(loss.data.numpy()) <= 0.0001) or (epoch == 500):
            break
    fig = plt.figure()
    plt.plot(e_list,loss_list)
    plt.plot(e_list,t_loss_list)
    plt.xlabel('Epochs')
    plt.ylabel('error')
    plt.legend(['Test eroor','Training eroor'],loc='best')
    plt.savefig('MatlabCodes/data/Lorenz/plots/rnn_errors.png')
    plt.show()
    np.save('data/trained_nets/n_input.npy',n_input)
    np.save('data/trained_nets/n_hidden.npy',n_hidden)
    np.save('data/trained_nets/n_output.npy',n_output)
    np.save('data/trained_nets/n_layers.npy',n_layers)
    torch.save(rnn.state_dict(),'data/trained_nets/RNNLorenz.pt')
    np.save('data/trained_nets/Y_params',YC)
    
    np.save('data/trained_nets/e_list.npy',e_list)
    np.save('data/trained_nets/t_loss_list.npy',t_loss_list)
    np.save('data/trained_nets/loss_list.npy',loss_list)