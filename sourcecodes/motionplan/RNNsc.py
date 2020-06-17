#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 28 16:40:34 2019

@author: hiroyasu
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 20 11:46:31 2019

@author: hiroyasu
"""

import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
import random

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

def GetTrainData(X,Y,N,BatchSize,time_step):
    idx = random.sample(range(N-time_step),BatchSize)
    Xtrain = None
    for i in range(BatchSize):
        Xtraini = X[idx[i]:idx[i]+time_step,:]
        Ytraini = Y[idx[i]:idx[i]+time_step,:]
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

if __name__ == "__main__":
    torch.manual_seed(19950808)
    np.random.seed(19950808)
    random.seed(19950808)
    X = np.genfromtxt('data/MotionPlan/XhisNN.csv',delimiter=',')
    cM = np.genfromtxt('data/MotionPlan/cMhisNN.csv', delimiter=',')
    Y = cM
    # Hyper Parameters
    time_step = 10 # rnn time step
    n_input = np.size(X,1) # rnn input size
    n_hidden = 128 # number of rnn hidden units
    n_output = np.size(Y,1) # rnn input size
    n_layers = 2 # rnn number of layers
    rnn = RNN(n_input,n_hidden,n_output,n_layers)
    LR = 0.2 # learning rate
    N = np.size(X,0)
    Ntest = 100
    BatchSize = np.floor(N/100)
    BatchSize = BatchSize.astype(np.int32)
    idx_test = random.sample(range(N),Ntest)
    Xtest = X[idx_test,:]
    Ytest = Y[idx_test,:]
    Xtest = Np2Var(Xtest)
    Ytest = Np2Var(Ytest)
    Xtest = Xtest.view(Ntest,1,-1)
    Ytest = Ytest.view(Ntest,1,-1)
    
    #optimizer = torch.optim.Adam(rnn.parameters(),lr=LR) # optimize all cnn parameters
    optimizer = torch.optim.SGD(rnn.parameters(),lr=LR) # optimize all cnn parameters
    loss_func = nn.MSELoss()
    h_state = None # for initial hidden state
    
    Nitrs = 1000000
    i_list = []
    loss_list = []
    for i in range(Nitrs):
        Xtrain, Ytrain = GetTrainData(X,Y,N,BatchSize,time_step)
        prediction = rnn.forward(Xtrain) # rnn output
        loss = loss_func(prediction,Ytrain)
        optimizer.zero_grad() # clear gradients for this training step
        loss.backward() # backpropagation, compute gradients
        optimizer.step() # apply gradients
        i_list.append(i)
        # !! next step is important !!
        Ypred = rnn(Xtest)
        loss_test = loss_func(Ypred,Ytest)
        loss_list.append(loss_test.data.numpy())
        if np.remainder(i,1000) == 0:
            print('current loss at i = ',i,': ',loss_test.data.numpy())
        if np.absolute(loss.data.numpy()) <= 0.00001:
            break
        """
        h_state = rnn.init_hidden(BatchSize)
        h_state[0].detach_()
        h_state[1].detach_()
        """
        
    fig = plt.figure()
    plt.plot(i_list,loss_list)
    plt.xlabel('Number of iteration')
    plt.ylabel('Loss')
    plt.show()
    torch.save(rnn,'data/TrainedNets/RNNsc.pt')