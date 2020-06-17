#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 20 20:44:38 2019

@author: hiroyasu
"""

import cvxpy as cp
import torch
import numpy as np
import matplotlib.pyplot as plt
import control
import RNNtraining as rnnl
import CCMsampling as sccm
import pandas as pd

dt = sccm.dt
tspan = sccm.tspan
s = sccm.s
b = sccm.b
r = sccm.r
ratio_p = sccm.ratio_p*20
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
        for num in range(0, runge_num):
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
        self.YC =  np.load('data/trained_nets/Y_params.npy')
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
        for num in range(0, runge_num):
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
    
class EstimatorEKF: 
    def __init__(self,Z,t,dt,runge_num=RungeNum):
        self.Z = Z
        self.t = t
        self.ns = Z.shape[0]
        self.runge_num = runge_num
        self.h = dt/runge_num
        
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
    
    def Aekf(self,state,time):
        x1 = state[0]                                                       
        x2 = state[1]                                                        
        x3 = state[2]
        A = np.array([[-s,s,0],[(r-x3),-1,-x1],[x2,x1,-b]])
        return A
    
    def estimator(self,state,time,L,y):
        yv = (np.array([y])).T
        Zv = (np.array([state])).T
        de = L@(yv-C@Zv)
        de = de.flatten()
        f = self.dynamics(state,time)
        dZdt = f+de
        return dZdt
    
    def one_step_sim(self,y):
        runge_num = self.runge_num
        A = self.Aekf(self.Z,self.t)
        R = 1*np.eye(1)
        Q = 100*np.eye(3)
        K,P,E = control.lqr(A.T,C.T,Q,R)
        L = K.T
        for num in range(0, runge_num)
            self.Z, self.t = self.rk4(L,y)
        return self.Z, self.t
    
class EstimatorCVSTEM: 
    def __init__(self,Z0,Z,t,dt,tspan,alp,runge_num=RungeNum):
        self.Z0 = Z0
        self.Z = Z
        self.t = t
        self.dt = dt
        self.ns = Z.shape[0]
        self.runge_num = runge_num
        self.h = dt/runge_num
        self.init_Qp(self.Z,self.t)
        self.tspan = tspan
        self.alp = alp
        self.Midx = 0
        self.CCM()
        
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
    
    def Aekf(self,state,time):
        x1 = state[0]                                                       
        x2 = state[1]                                                        
        x3 = state[2]
        A = np.array([[-s,s,0],[(r-x3),-1,-x1],[x2,x1,-b]])
        return A
    
    def estimator(self,state,time,L,y):
        yv = (np.array([y])).T
        Zv = (np.array([state])).T
        de = L@(yv-C@Zv)
        de = de[:,0]
        f = self.dynamics(state,time)
        dZdt = f+de
        return dZdt

    def init_Qp(self,state,time):
        A = self.Aekf(state,time)
        I = np.eye(3)
        Q = cp.Variable((3,3),PSD=True)
        nu = cp.Variable((1,1),nonneg=True)
        chi = cp.Variable((1,1),nonneg=True)
        M11 = -(Q@A+A.T@Q-2*nu*C.T@C+2*alp*Q)
        constraints = [M11 >> 0]
        constraints += [chi*I >> Q]
        constraints += [Q >> I]
        prob = cp.Problem(cp.Minimize(d1_over*b_over*chi+d2_over*c_over*d_over*nu),constraints)
        prob.solve()
        Q = Q.value/nu.value
        self.Qp = Q
        #print(prob.status)
        pass

    def CV_STEM(self,state,time):
        Qp = self.Qp
        dt = self.dt
        A = self.Aekf(state,time)
        I = np.eye(3)
        Q = cp.Variable((3,3),PSD=True)
        nu = cp.Variable((1,1),nonneg=True)
        chi = cp.Variable((1,1),nonneg=True)
        dQdt = (Q-nu*Qp)/dt
        M11 = -(dQdt+Q@A+A.T@Q-2*nu*C.T@C+2*alp*Q)
        constraints = [M11 >> 0]
        constraints += [chi*I >> Q]
        constraints += [Q >> I]
        prob = cp.Problem(cp.Minimize(d1_over*b_over*chi+d2_over*c_over*d_over*nu),constraints)
        prob.solve(solver=cp.MOSEK)
        Q = Q.value/nu.value
        P = np.linalg.inv(Q)
        L = P@C.T
        self.Qp = Q
        print(prob.status)
        return L,P
    
    def GenerateXX(self):
        X0 = self.Z0
        tspan = self.tspan
        dt = self.dt
        N = (tspan[1]-tspan[0])/dt
        N = N.astype(np.int32)
        tt = np.zeros(N+1)
        XX = np.zeros((N+1,3))
        tt[0] = 0
        XX[0,:] = X0
        X = X0
        t = 0
        for i in range(N):
            X,t = self.one_step_sim_X(X,t)
            tt[i+1] = t
            XX[i+1,:] = X
        return XX,tt
    
    def CCM(self):
        alp = self.alp
        epsilon = 0.
        XX,tt = self.GenerateXX()
        N = XX.shape[0]-1
        I = np.identity(3)
        WW = {}
        for i in range(N+1):
          WW[i] = cp.Variable((3,3),PSD=True)
        nu = cp.Variable(nonneg=True)
        chi = cp.Variable(nonneg=True)
        constraints = [chi*I-WW[0] >> epsilon*I,WW[0]-I >> epsilon*I]
        for k in range(N):
            Xk = XX[k,:]
            t = tt[k]
            Ax = self.Aekf(Xk,t)
            Wk = WW[k]
            Wkp1 = WW[k+1]
            constraints += [-2*alp*Wk-((Wkp1-Wk)/dt+Wk@Ax+Ax.T@Wk-2*nu*C.T@C) >> epsilon*I]
            constraints += [chi*I-Wkp1 >> epsilon*I,Wkp1-I >> epsilon*I]
        prob = cp.Problem(cp.Minimize(d1_over*b_over*chi+d2_over*c_over*d_over*nu),constraints)
        prob.solve(solver=cp.MOSEK)
        cvx_status = prob.status
        cvx_optval = d1_over*b_over*chi.value+d2_over*c_over*d_over*nu.value
        WWout = {}
        for i in range(N+1):
          WWout[i] = WW[i].value/nu.value
        chi = chi.value
        nu = nu.value
        print(cvx_status)
        self.WW = WWout
        self.chi = chi
        self.nu = nu
        self.cvx_optval = cvx_optval
        pass
    
    def one_step_sim(self,y):
        runge_num = self.runge_num
        P = np.linalg.inv(self.WW[self.Midx])
        L = P@C.T
        L,P = self.CV_STEM(self.Z,self.t)
        self.Midx += 1
        for num in range(0, runge_num):
            self.Z, self.t = self.rk4(L,y)
        return self.Z, self.t,P
    
    def one_step_sim_X(self,state,time):
        runge_num = self.runge_num
        for num in range(0, runge_num):
            state,time = self.rk4X(state,time)
        state = state
        return state,time

    def rk4X(self,state,time):
        h = self.h
        k1 = self.dynamics(state,time)
        k2 = self.dynamics(state+k1*h/2.,time+h/2.)
        k3 = self.dynamics(state+k2*h/2.,time+h/2.)
        k4 = self.dynamics(state+k3*h,time+h)
        return state+h*(k1+2.*k2+2.*k3+k4)/6.,time+h
    
    def CCMfinal(self,Xhis):
        alp = self.alp
        epsilon = 0.
        XX,tt = self.GenerateXX()
        XX = Xhis
        N = XX.shape[0]-1
        I = np.identity(3)
        WW = {}
        for i in range(N+1):
          WW[i] = cp.Variable((3,3),PSD=True)
        nu = cp.Variable(nonneg=True)
        chi = cp.Variable(nonneg=True)
        constraints = [chi*I-WW[0] >> epsilon*I,WW[0]-I >> epsilon*I]
        for k in range(N):
            Xk = XX[k,:]
            t = tt[k]
            Ax = self.Aekf(Xk,t)
            Wk = WW[k]
            Wkp1 = WW[k+1]
            constraints += [-2*alp*Wk-((Wkp1-Wk)/dt+Wk@Ax+Ax.T@Wk-2*nu*C.T@C) >> epsilon*I]
            constraints += [chi*I-Wkp1 >> epsilon*I,Wkp1-I >> epsilon*I]
        prob = cp.Problem(cp.Minimize(d1_over*b_over*chi+d2_over*c_over*d_over*nu),constraints)
        prob.solve(solver=cp.MOSEK)
        cvx_status = prob.status
        cvx_optval = d1_over*b_over*chi.value+d2_over*c_over*d_over*nu.value
        WWout = {}
        MMout = {}
        for i in range(N+1):
          WWout[i] = WW[i].value/nu.value
          MMout[i] = np.linalg.inv(WWout[i])
        chi = chi.value
        nu = nu.value
        print(cvx_status)
        self.WW = WWout
        self.MM = MMout
        self.chi = chi
        self.nu = nu
        self.cvx_optval = cvx_optval
        pass
    
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
 
if __name__ == "__main__":
    np.random.seed(seed=5)
    X0 = np.array([-1,2,3])
    Z0 = np.array([150.1,-1.5,-6])
    t0 = 0
    tf = 50
    N = np.floor((tf-t0)/dt+1)
    N = N.astype(np.int32)
    est = Estimator(Z0,t0,dt)
    estCVSTEM = EstimatorCVSTEM(Z0,Z0,t0,dt,tspan,alp)
    estEKF = EstimatorEKF(Z0,t0,dt)
    lor = Lorenz(X0,t0,dt)
    this = t0
    this = np.array([t0])
    Xhis = X0
    Z1his = Z0
    Z2his = Z0
    Z3his = Z0
    e1his = np.array([np.linalg.norm(Z0-X0)])
    e2his = np.array([np.linalg.norm(Z0-X0)])
    e3his = np.array([np.linalg.norm(Z0-X0)])
    P1his = {}
    P2bhis = {}
    for i in range(N):
        y = lor.measurement()
        X,t = lor.one_step_sim()
        this = np.hstack((this,t))
        Xhis = np.vstack((Xhis,X))
        
        Z1,t,P1 = est.one_step_sim(y)
        Z1his = np.vstack((Z1his,Z1))
        e1 = np.linalg.norm(Z1-X)
        e1his = np.hstack((e1his,e1))
        
        Z2,t,P2 = estCVSTEM.one_step_sim(y)
        Z2his = np.vstack((Z2his,Z2))
        e2 = np.linalg.norm(Z2-X)
        e2his = np.hstack((e2his,e2))
        
        Z3,t = estEKF.one_step_sim(y)
        Z3his = np.vstack((Z3his,Z3))
        e3 = np.linalg.norm(Z3-X)
        e3his = np.hstack((e3his,e3))
        
        P1his[i] = P1
        P2bhis[i] = P2
        
    estCVSTEM.CCMfinal(Z1his)
    P2his = estCVSTEM.MM
    dPhis = []
    YC =  np.load('data/trained_nets/Y_params.npy')
    for i in range(N):
        P1 = P1his[i]
        cP1 = M2cMvec(P1)/YC
        P2 = P2his[i]
        cP2 = M2cMvec(P2)/YC
        dPhis.append((np.linalg.norm(cP1-cP2))**2)
        
    data1 = {'score': e1his}
    data2 = {'score': e2his}
    data3 = {'score': e3his}
    # Create dataframe
    df1 = pd.DataFrame(data1)
    df2 = pd.DataFrame(data2)
    df3 = pd.DataFrame(data3)
    WindowSize= 50
    dfe1his = df1.rolling(window=WindowSize).mean()
    dfe2his = df2.rolling(window=WindowSize).mean()
    dfe3his = df3.rolling(window=WindowSize).mean()
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(Xhis[:,0],Xhis[:,1],Xhis[:,2],color='blue')
    ax.plot(Z1his[:,0],Z1his[:,1],Z1his[:,2],color='red')  
    ax.plot(Z2his[:,0],Z2his[:,1],Z2his[:,2],color='green')  
    ax.plot(Z3his[:,0],Z3his[:,1],Z3his[:,2],color='yellow')  
    fig = plt.figure()
    plt.plot(this,dfe1his)
    plt.plot(this,dfe2his)
    plt.plot(this,optval*np.ones(this.size))  
    plt.plot(this,dfe3his)
    plt.xlabel('t')
    plt.ylabel('Error')
    plt.legend(['CV-STEM with RNN','CV-STEM','EKF','Expected upper bound'],loc='best')
    plt.show()
    
    fig = plt.figure()
    plt.plot(this,Z1his[:,0]-Xhis[:,0])
    plt.plot(this,Z2his[:,0]-Xhis[:,0])
    plt.plot(this,Z3his[:,0]-Xhis[:,0])
    plt.xlabel('t')
    plt.ylabel('Error')
    plt.legend(['CV-STEM with RNN','CV-STEM','EKF','Expected upper bound'],loc='best')
    plt.show()
    
    fig = plt.figure()
    plt.plot(this[0:N],dPhis)
    plt.xlabel('t')
    plt.ylabel('Error')
    plt.show()
    
    np.save('data/simulation/optval.npy',optval)
    np.save('data/simulation/this.npy',this)
    np.save('data/simulation/Xhis.npy',Xhis)
    np.save('data/simulation/Z1his.npy',Z1his)
    np.save('data/simulation/Z2his.npy',Z2his)
    np.save('data/simulation/Z3his.npy',Z3his)
    np.save('data/simulation/e1his.npy',e1his)
    np.save('data/simulation/e2his.npy',e2his)
    np.save('data/simulation/e3his.npy',e3his)
    
    nP1his = []
    nP2his = []
    nP2bhis = []
    for i in range(N):
        P1 = P1his[i]
        P2 = P2his[i]
        P2b = P2bhis[i]
        nP1his.append(np.linalg.norm(P1,ord=2))
        nP2his.append(np.linalg.norm(P2,ord=2))
        nP2bhis.append(np.linalg.norm(P2b,ord=2))
    dchi = abs(np.max(nP1his)/np.min(nP1his)-np.max(nP2his)/np.min(nP2his))
    dnu = abs(np.max(nP1his)-np.max(nP2his))
    dchi = abs(np.max(nP1his)/np.min(nP1his)-np.max(nP2bhis)/np.min(nP2bhis))
    dnu = abs(np.max(nP1his)-np.max(nP2bhis))