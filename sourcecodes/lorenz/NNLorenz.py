# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 16:19:52 2019

@author: Hiroyasu
"""

"""
Regression of P: R6 -> R6*6
"""

import torch
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import random

class Net(torch.nn.Module):
    def __init__(self, n_input, n_hidden, n_output):
        super(Net, self).__init__()
        self.hidden1 = torch.nn.Linear(n_input, n_hidden)
        self.hidden2 = torch.nn.Linear(n_hidden, n_hidden)
        self.hidden3 = torch.nn.Linear(n_hidden, n_hidden)
        self.hidden4 = torch.nn.Linear(n_hidden, n_hidden)
        self.predict = torch.nn.Linear(n_hidden, n_output)

    def forward(self, x):
        x = torch.relu(self.hidden1(x))
        x = torch.relu(self.hidden2(x))
        x = torch.relu(self.hidden3(x))
        x = torch.relu(self.hidden4(x))
        x = self.predict(x)
        return x
    
def testfun(X):
    x1 = X[:,0]
    x2 = X[:,1]
    y1 = np.sin(np.sqrt(x1**2+x2**2))/np.sqrt(x1**2+x2**2)
    y2 = x1**2-x2**2
    y3 = x1**2+x2**2
    Y = np.array([y1,y2,y3])
    Y = Y.T
    return Y

def Np2Var(X):
    X = X.astype(np.float32)
    X = torch.from_numpy(X)
    return X

def NCMLoss(Pnet,P):
    loss = 0
    N = P.shape[0]
    for i in range(N):
        Pi = P[i,:]
        Pi = Pi.view(-1,3)
        Pneti = Pnet[i,:]
        Pneti = Pneti.view(-1,3)
        Ei = Pi-torch.mm(Pneti,torch.t(Pneti))
        loss += torch.norm(Ei,keepdim=True)
    return loss
 
def NCMLoss2(Pnet,P):
    loss = 0
    N = P.shape[0]
    for i in range(N):
        Pi = P[i,:]
        Pi = Pi.view(-1,3)
        Pneti = Pnet[i,:]
        Pneti = Pneti.view(-1,3)
        Ei = Pi-torch.mm(Pneti,Pneti.t())
        loss += torch.norm(Ei,keepdim=True)
    return loss    

if __name__ == "__main__":
    X = np.genfromtxt('data/XhisNN.csv',delimiter=',')
    #K = np.genfromtxt('data/LhisNN.csv', delimiter=',')
    #P = np.genfromtxt('data/PhisNN.csv', delimiter=',')
    cP = np.genfromtxt('data/cMhisNN_l.csv', delimiter=',')
    Y = cP
    torch.manual_seed(19950808)
    np.random.seed(19950808)
    random.seed(19950808)
    N = np.size(X,0)
    #print(N)
    Ntest = 100
    n_input = np.size(X,1)
    n_hidden = 100
    BatchSize = np.floor(N/10)
    BatchSize = BatchSize.astype(np.int32)
    n_output = np.size(Y,1)
    print(n_output)
    idx_test = random.sample(range(N),Ntest)
    Xtest = X[idx_test,:]
    Ytest = Y[idx_test,:]
    Xtest = Np2Var(Xtest)
    Ytest = Np2Var(Ytest)
    #print(Ytest)
    #print(Ytest.view(-1,3))
    #print(torch.t(Ytest))
    #print(torch.norm(test2,keepdim=True))
    #print(test2b)
    #print(torch.norm(test2-test2b,keepdim=True))
    net = Net(n_input,n_hidden,n_output)
    optimizer = torch.optim.Adam(net.parameters())
    loss_func = torch.nn.MSELoss()
    i_list = []
    loss_list = []
    for i in range(500000):
        idx = random.sample(range(N-1),BatchSize)
        Xtrain = X[idx,:]
        Ytrain = Y[idx,:]
        Xtrain = Np2Var(Xtrain)
        Ytrain = Np2Var(Ytrain)
        prediction = net(Xtrain)
        loss = loss_func(prediction,Ytrain)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        i_list.append(i)
        Ypred = net(Xtest)
        loss_test = loss_func(Ypred,Ytest)
        loss_list.append(loss_test.data.numpy())
        if np.remainder(i,100) == 0:
            print('current loss at i = ',i,': ',loss_test.data.numpy())
        if np.absolute(loss_test.data.numpy()) <= 0.2:
            break
        
    Ypred = net(Xtest)
    Xplot = Xtest.data.numpy()
    Yplot = Ypred.data.numpy()
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.scatter3D(Xplot[:,0],Xplot[:,1],Yplot[:,0])
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.scatter3D(Xplot[:,0],Xplot[:,1],Yplot[:,1])
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.scatter3D(Xplot[:,0],Xplot[:,1],Yplot[:,2])
    
    fig = plt.figure()
    plt.plot(i_list,loss_list)
    plt.xlabel('Number of iteration')
    plt.ylabel('Loss')
    plt.show()
    torch.save(net,'data/TrainedNets/NNLorenz.pt')