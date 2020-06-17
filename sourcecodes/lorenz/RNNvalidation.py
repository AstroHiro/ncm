#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 21:14:42 2020

@author: hiroyasu
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import RNNtraining as rnnl
import CCMsampling as sccm
import pandas as pd
import pickle

dt = sccm.dt*0.1
tspan = sccm.tspan
#tspan = np.array([0,20])
s = sccm.s
b = sccm.b
r = sccm.r
ratio_p = sccm.ratio_p*1
C = sccm.C
B = sccm.B
D = sccm.D
dim = sccm.dim
d1_over = sccm.d1_over
d2_over = sccm.d2_over
b_over = sccm.b_over
c_over = sccm.c_over
d_over = sccm.d_over
RungeNum = sccm.RungeNum
RungeNum = 1

N = np.load('data/params/optimal/N.npy')
alp = alp = np.load('data/params/optimal/alp.npy')
optvals = np.load('data/params/optimal/optvals.npy')
optval = np.mean(optvals)
n_input = rnnl.n_input
n_hidden = rnnl.n_hidden
n_output = rnnl.n_output
n_layers = rnnl.n_layers

class Lorenz: 
    def __init__(self,X,t,dt,runge_num=RungeNum):
        self.X = X
        self.t = t
        self.dt = dt
        self.runge_num = runge_num
        self.h = dt/runge_num

    def dynamics(self,state,time):
        x1 = state[0]                                                       
        x2 = state[1]                                                        
        x3 = state[2]
        dx1dt = s*(x2-x1);
        dx2dt = x1*(r-x3)-x2;
        dx3dt = x1*x2-b*x3;
        return np.array([dx1dt,dx2dt,dx3dt])

    def measurement(self):
        Xv = (np.array([self.X])).T
        d2 = (np.random.rand(1,1)*2-1)*ratio_p
        y = C@Xv+D@d2
        y = y[:,0]
        return y
    
    def rk4(self):
        X = self.X
        t = self.t
        h = self.h
        k1 = self.dynamics(X,t)
        k2 = self.dynamics(X+k1*h/2.,t+h/2.)
        k3 = self.dynamics(X+k2*h/2.,t+h/2.)
        k4 = self.dynamics(X+k3*h,t+h)
        return X+h*(k1+2.*k2+2.*k3+k4)/6., t+h

    def one_step_sim(self):
        runge_num = self.runge_num
        for num in range(0, runge_num): #self.X, self.tを更新することで全体で更新
            self.X, self.t = self.rk4()
        d1 = (np.random.rand(3,1)*2-1)*ratio_p
        dist_dt = B@d1*self.dt
        self.X = self.X+dist_dt[:,0]
        return self.X, self.t
    
    
class Estimator: 
    def __init__(self,Z,t,dt,runge_num=RungeNum):
        self.Z = Z
        self.t = t
        self.ns = Z.shape[0]
        self.runge_num = runge_num
        self.h = dt/runge_num
        self.net = rnnl.RNN(n_input,n_hidden,n_output,n_layers)
        self.net.load_state_dict(torch.load('data/trained_nets/RNNLorenz.pt'))
        self.net.eval()
        self.YC = np.load('data/trained_nets/Y_params.npy')
        #self.net = torch.load('data/TrainedNets/NNLorenz.pt')
        self.RNN = 1
        
    def rk4(self,L,y):
        Z = self.Z
        t = self.t
        h = self.h
        k1 = self.estimator(Z,t,L,y)
        k2 = self.estimator(Z+k1*h/2.,t+h/2.,L,y)
        k3 = self.estimator(Z+k2*h/2.,t+h/2.,L,y)
        k4 = self.estimator(Z+k3*h,t+h,L,y)
        return Z+h*(k1+2.*k2+2.*k3+k4)/6.,t+h
    
    def dynamics(self,state,time):
        x1 = state[0]                                                       
        x2 = state[1]                                                        
        x3 = state[2]
        dx1dt = s*(x2-x1);
        dx2dt = x1*(r-x3)-x2;
        dx3dt = x1*x2-b*x3;
        return np.array([dx1dt,dx2dt,dx3dt])
    
    def estimator(self,state,time,L,y):
        yv = (np.array([y])).T
        Zv = (np.array([state])).T
        de = L@(yv-C@Zv)
        de = de[:,0]
        f = self.dynamics(state,time)
        dZdt = f+de
        return dZdt
    
    def one_step_sim(self,y):
        runge_num = self.runge_num
        L = self.GetL()
        P = self.GetP()
        for num in range(0, runge_num): #self.X, self.tを更新することで全体で更新
            self.Z, self.t = self.rk4(L,y)
        return self.Z, self.t,P
    
    def GetP(self):
        Znet = self.Np2Var(self.Z)
        if self.RNN == 1:
            cP = self.net(Znet.view(1,1,-1))
            cP = cP.data.numpy()*self.YC
            P = self.cP2P(cP[0,0,:])
        else:
            cP = self.net(Znet)
            cP = cP.data.numpy()
            P = self.cP2P(cP)
        return P
    
    def GetL(self):
        P = self.GetP()
        Ct = C.T
        L = P@Ct
        return L
    
    def Np2Var(self,X):
        X = X.astype(np.float32)
        X = torch.from_numpy(X)
        return X
        
    def cP2P(self,cP):
        cPnp = 0
        for i in range(self.ns):
            lb = i*(i+1)/2
            lb = int(lb)
            ub = (i+1)*(i+2)/2
            ub = int(ub)
            Di = cP[lb:ub]
            Di = np.diag(Di,self.ns-(i+1))
            cPnp += Di
        P = (cPnp.T)@cPnp
        return P

def M2cMvec(M):
    cholM = np.linalg.cholesky(M)
    cholM = cholM.T # upper triangular
    ns = M.shape[0]
    cM = np.zeros(int(ns*(ns+1)/2))
    for ii in range(ns):
        jj = (ns-1)-ii;
        di = np.diag(cholM,jj)
        cM[int(1/2*ii*(ii+1)):int(1/2*(ii+1)*(ii+2))] = di
    return cM

def SaveDict(filename,var):
    output = open(filename,'wb')
    pickle.dump(var,output)
    output.close()
    pass
    
def LoadDict(filename):
    pkl_file = open(filename,'rb')
    varout = pickle.load(pkl_file)
    pkl_file.close()
    return varout

if __name__ == "__main__":
    np.random.seed(seed=5)
    numX0 = 10
    WindowSize = 50
    X0s = (np.random.rand(numX0,3)*2-1)*10
    Z0s = (np.random.rand(numX0,3)*2-1)*10
    ccm = sccm.CCM(X0s,tspan,0.1,1,np.array([alp]))
    X0 = np.array([1,2,0])
    Z0 = np.array([10,2,0])
    net = rnnl.RNN(n_input,n_hidden,n_output,n_layers)
    net.load_state_dict(torch.load('data/trained_nets/RNNLorenz.pt'))
    net.eval()
    YC = np.load('data/trained_nets/Y_params.npy')
    
        
    t0 = tspan[0]
    tf = tspan[1]
    N = np.floor((tf-t0)/dt+1)
    N = N.astype(np.int32)
    dcMMs = {}
    for iX0 in range(numX0):
        est = Estimator(Z0,t0,dt)
        lor = Lorenz(X0,t0,dt)
        Xhis01 = X0
        Z1his01 = Z0
        cholP1 = {}
        t = 0
        j = 0
        cholP1[0] = M2cMvec(est.GetP())/YC
        for i in range(N):
            y = lor.measurement()
            X,t = lor.one_step_sim()
            Z1,t1,P1 = est.one_step_sim(y)
            
            if np.remainder(i+1,10) == 0:
                cholP1[j+1] = M2cMvec(est.GetP())/YC
                Xhis01 = np.vstack((Xhis01,X))
                Z1his01 = np.vstack((Z1his01,Z1))
                j += 1
                
        cvx_status,cvx_optval,tt,XX,WWout,chi,nu = ccm.CCMXhis(Z1his01,alp)
        dcP = []
        cholMM = {}
        dcMM = []
        for i in range(len(WWout)-1):
            Mi = np.linalg.inv(WWout[i])
            cMi = M2cMvec(Mi)/YC
            cholMM[i] = cMi
            dcMM.append((np.linalg.norm(cholMM[i]-cholP1[i]))**2)
        
        dcMMs[iX0] = dcMM
    
    fig = plt.figure()    
    for iX0 in range(numX0):
        dcMM = dcMMs[iX0]
        plt.plot(np.linspace(0,50,500),dcMM)
        plt.xlabel('t')
        plt.ylabel('Error')
    plt.show()

    fig = plt.figure()    
    for iX0 in range(numX0):
        dcMM = dcMMs[iX0]
        datadM = {'score': dcMM}
        dfdM = pd.DataFrame(datadM)
        dfdMhis = dfdM.rolling(window=WindowSize).mean()
        
        plt.plot(np.linspace(0,50,500),dfdMhis)
        plt.xlabel('t')
        plt.ylabel('Error')
    plt.show()
        
    SaveDict('data/trained_nets/dcMMs.pkl',dcMMs)