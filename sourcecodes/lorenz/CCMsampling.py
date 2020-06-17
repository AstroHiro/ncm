#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  7 16:41:50 2020

@author: hiroyasu
"""

import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt
import pickle

dt = 0.1
tspan = np.array([0,50])
N = tspan[1]/dt
N = N.astype(np.int32)
s = 10
b = 8/3
r = 28
ratio_p = 1
C = np.array([[1,0,0]])
B = np.identity(3)
D = np.array([[1]])
dim = 3
d1_over = ratio_p*np.sqrt(dim)
d2_over = ratio_p
b_over = np.linalg.norm(B,ord=2)
c_over = np.linalg.norm(C,ord=2)
d_over = np.linalg.norm(D,ord=2)
RungeNum = 10

class CCM:
    def __init__(self,X0s,tspan,dt,numX0,alp_list,runge_num=RungeNum):
        self.X0s = X0s
        self.tspan = tspan
        self.dt = dt
        self.numX0 = numX0
        self.alp_list = alp_list
        self.runge_num = runge_num
        self.h = dt/runge_num

    def dynamics(self,state,time):
        x1 = state[0]                                                       
        x2 = state[1]                                                        
        x3 = state[2]
        dx1dt = s*(x2-x1)
        dx2dt = x1*(r-x3)-x2
        dx3dt = x1*x2-b*x3
        return np.array([dx1dt,dx2dt,dx3dt])

    def Aekf(self,state,time):
        x1 = state[0]                                                       
        x2 = state[1]                                                        
        x3 = state[2]
        A = np.array([[-s,s,0],[(r-x3),-1,-x1],[x2,x1,-b]])
        return A
    
    def rk4(self,state,time):
        h = self.h
        k1 = self.dynamics(state,time)
        k2 = self.dynamics(state+k1*h/2.,time+h/2.)
        k3 = self.dynamics(state+k2*h/2.,time+h/2.)
        k4 = self.dynamics(state+k3*h,time+h)
        return state+h*(k1+2.*k2+2.*k3+k4)/6.,time+h

    def one_step_sim(self,state,time):
        runge_num = self.runge_num
        for num in range(0, runge_num):
            state,time = self.rk4(state,time)
        state = state
        return state,time

    def GenerateXX(self,X0):
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
            X,t = self.one_step_sim(X,t)
            tt[i+1] = t
            XX[i+1,:] = X
        return XX,tt

    def CCM(self,X0,alp):
        epsilon = 0.
        XX,tt = self.GenerateXX(X0)
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
        return cvx_status,cvx_optval,tt,XX,WWout,chi,nu

    def CCMXhis(self,Xhis,alp):
        epsilon = 0.
        XX,tt = self.GenerateXX(Xhis[0,:])
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
        for i in range(N+1):
          WWout[i] = WW[i].value/nu.value
        chi = chi.value
        nu = nu.value
        print(cvx_status)
        return cvx_status,cvx_optval,tt,XX,WWout,chi,nu
        
    def GetCCMs(self,logidx):
        numX0 = self.numX0
        alp_list = self.alp_list
        a_len = alp_list.size
        optvals = np.zeros((numX0,a_len))
        alp_list_iX0 = np.zeros(numX0)
        X0s = self.X0s
        chi_s = np.zeros((numX0,a_len))
        nu_s = np.zeros((numX0,a_len))
        a_len = alp_list.size
        status_str = []
        plt.figure()
        for iX0 in range(numX0):
            X0 = X0s[iX0,:]
            print('iX0 = ',iX0)
            for al in range(a_len):
                alp = alp_list[al]
                print('alpha =',alp)
                cvx_status,cvx_optval,tt,XX,WW,chi,nu = self.CCM(X0,alp)
                optvals[iX0,al] = cvx_optval/alp
                chi_s[iX0,al] = chi
                nu_s[iX0,al] = nu
                status_str.append(cvx_status)
            if logidx == 0:
                plt.semilogx((alp_list),optvals[iX0,:])
            else:
                plt.semilogy((alp_list),optvals[iX0,:])
            optidx = np.argmin(optvals[iX0,:])
            alp_list_iX0[iX0] = alp_list[optidx]
        plt.show()
        self.optalp = np.mean(alp_list_iX0)
        return optvals,chi_s,nu_s,status_str
    
    
    def GetOptimalCCMs(self,alp):
        numX0 = self.numX0
        optvals = np.zeros(numX0)
        X0s = self.X0s
        chi_s = np.zeros(numX0)
        nu_s = np.zeros(numX0)
        status_str = []
        XXs = {}
        WWs = {}
        for iX0 in range(numX0):
            X0 = X0s[iX0,:]
            print('iX0 = ',iX0)
            cvx_status,cvx_optval,tt,XX,WW,chi,nu = self.CCM(X0,alp)
            optvals[iX0] = cvx_optval/alp
            chi_s[iX0] = chi
            nu_s[iX0] = nu
            XXs[iX0] = XX
            WWs[iX0] = WW
            status_str.append(cvx_status)
        return optvals,chi_s,nu_s,status_str,tt,XXs,WWs
    
    def SaveTrainingData(self,XXs,WWs):
        ns = WWs[0][0].shape[0]
        N = len(WWs[0])-1
        NX0 = len(WWs)
        XhisNN = XXs[0]
        MMs = np.zeros((ns,ns,N+1,NX0))
        NN = NX0*(N+1)
        WhisNN = np.zeros((NN,ns**2))
        MhisNN = np.zeros((NN,ns**2))
        nc = int(1/2*ns*(ns+1))
        cMhisNN = np.zeros((NN,nc))
        for j in range(NX0):
            if j > 0:
                XhisNN = np.vstack((XhisNN,XXs[j]))
            for i in range(N+1):
                Wi = WWs[j][i]
                Mi = np.linalg.inv(Wi)
                MMs[:,:,i,j] = Mi
                WhisNN[i+(N+1)*j,:] = np.reshape(Wi,ns**2)
                MhisNN[i+(N+1)*j,:] = np.reshape(Mi,ns**2)
                cholMi = np.linalg.cholesky(Mi)
                cholMi = cholMi.T # upper triangular
                for ii in range (ns):
                    jj = (ns-1)-ii;
                    di = np.diag(cholMi,jj)
                    print(di)
                    cMhisNN[i+(N+1)*j,int(1/2*ii*(ii+1)):int(1/2*(ii+1)*(ii+2))] = di
        return XhisNN,MMs,WhisNN,MhisNN,cMhisNN

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
    '''
    # macro optvals
    numX0 = 10
    alp_list = np.array([0.05,0.1,1,3,6])
    X0s = (np.random.rand(numX0,3)*2-1)*10
    ccm = CCM(X0s,tspan,dt,numX0,alp_list)
    optvals,chi_s,nu_s,status_str = ccm.GetCCMs(0)
    alp = ccm.optalp
    np.save('data/params/alpha_macro/alp.npy',alp)
    np.save('data/params/alpha_macro/alp_list.npy',alp_list)
    np.save('data/params/alpha_macro/X0s.npy',X0s)
    np.save('data/params/alpha_macro/optvals.npy',optvals)
    np.save('data/params/alpha_macro/chi_s.npy',chi_s)
    np.save('data/params/alpha_macro/nu_s.npy',nu_s)
    np.save('data/params/alpha_macro/status_str.npy',status_str)
    '''
    
    '''
    # micro optvals
    numX0 = 100
    alp_list = np.arange(1,5.1,0.1)
    X0s = (np.random.rand(numX0,3)*2-1)*10
    ccm = CCM(X0s,tspan,dt,numX0,alp_list)
    optvals,chi_s,nu_s,status_str = ccm.GetCCMs(1)
    alp = ccm.optalp
    np.save('data/params/alpha_micro/alp.npy',alp)
    np.save('data/params/alpha_micro/alp_list.npy',alp_list)
    np.save('data/params/alpha_micro/X0s.npy',X0s)
    np.save('data/params/alpha_micro/optvals.npy',optvals)
    np.save('data/params/alpha_micro/chi_s.npy',chi_s)
    np.save('data/params/alpha_micro/nu_s.npy',nu_s)
    np.save('data/params/alpha_micro/status_str.npy',status_str)
    
    '''
    # CCM sampling
    numX0 = 100
    alp = np.load('data/params/alpha_micro/alp.npy')
    alp_list = np.array([alp])
    X0s = (np.random.rand(numX0,3)*2-1)*10
    ccm = CCM(X0s,tspan,dt,numX0,alp_list)
    optvals,chi_s,nu_s,status_str,tt,XXs,WWs = ccm.GetOptimalCCMs(alp)
    '''
    np.save('data/params/optimal/alp.npy',alp)
    np.save('data/params/optimal/X0s.npy',X0s)
    np.save('data/params/optimal/optvals.npy',optvals)
    np.save('data/params/optimal/chi_s.npy',chi_s)
    np.save('data/params/optimal/nu_s.npy',nu_s)
    np.save('data/params/optimal/status_str.npy',status_str)
    np.save('data/params/optimal/tt.npy',tt)
    SaveDict('data/params/optimal/XXs.pkl',XXs)
    SaveDict('data/params/optimal/WWs.pkl',WWs)
    
    XhisNN,MMs,WhisNN,MhisNN,cMhisNN = ccm.SaveTrainingData(XXs,WWs)
    np.save('data/training_data/XhisNN.npy',XhisNN)
    np.save('data/training_data/WWs.npy',WWs)
    np.save('data/training_data/MMs.npy',MMs)
    np.save('data/training_data/WhisNN.npy',WhisNN)
    np.save('data/training_data/MhisNN.npy',MhisNN)
    np.save('data/training_data/cMhisNN.npy',cMhisNN)
    np.save('data/params/optimal/N.npy',N)
    '''

