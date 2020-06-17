#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 14 13:03:16 2020

@author: hiroyasu
"""

import numpy as np
import control
import cvxpy as cp
import matplotlib.pyplot as plt
import SCPmulti as scp
import pickle

DT = scp.DT
TSPAN = scp.TSPAN
M = scp.M
II = scp.II
L = scp.L
bb = scp.bb
FMIN = scp.FMIN
FMAX = scp.FMAX
RungeNum = scp.RungeNum
AA = scp.AA

linnum = 25
xmax = 20
ymax = 16
X01 = np.linspace(0,ymax,num=linnum)
X01 = np.vstack((np.zeros(linnum),X01)).T
X02 = np.linspace(0,xmax,num=linnum)
X02 = np.vstack((X02,ymax*np.ones(linnum))).T
X03 = np.flipud(np.linspace(0,ymax,num=linnum))
X03 = np.vstack((xmax*np.ones(linnum),X03)).T
X04 = np.flipud(np.linspace(0,xmax,num=linnum))
X04 = np.vstack((X04,np.zeros(linnum))).T
Xf1 = X03
Xf2 = X04
Xf3 = X01
Xf4 = X02
X0s = np.vstack((np.vstack((X01,X02)),np.vstack((X03,X04))))
Xfs = np.vstack((np.vstack((Xf1,Xf2)),np.vstack((Xf3,Xf4))))

class DesiredTrajectories: 
    def __init__(self,X0s,Xfs,dt=DT,tspan=TSPAN,runge_num=RungeNum,m=M,I=II,l=L,b=bb,fmin=FMIN,fmax=FMAX,A=AA):
        self.X0s  = X0s
        self.Xfs  = Xfs
        self.dt  = dt
        self.tspan = tspan
        N = (tspan[1]-tspan[0])/dt
        self.N = N.astype(np.int32)
        self.runge_num = runge_num
        self.h = dt/runge_num
        self.m = m
        self.I = I
        self.l = l
        self.b = b
        self.fmin = fmin
        self.fmax = fmax
        self.A = A

    def dynamics(self,t,states,inputs):
        A = self.A
        B = self.GetB(states)
        Xv = np.transpose(np.array([states]))
        Uv = np.transpose(np.array([inputs]))
        dXdt = A.dot(Xv)+B.dot(Uv)
        dXdt = dXdt[:,0]
        return dXdt
    
    def GetB(self,states):
        m = self.m
        I = self.I
        l = self.l
        b = self.b
        th = states[2]
        T = np.array([[np.cos(th)/m,np.sin(th)/m,0],[-np.sin(th)/m,np.cos(th)/m,0],[0,0,1/2./I]])
        H = np.array([[-1,-1,0,0,1,1,0,0],[0,0,-1,-1,0,0,1,1],[-l,l,-b,b,-l,l,-b,b]])
        B = np.vstack((np.zeros((3,8)),T@H))
        return B

    def GetdBdX(self,X):
        m = self.m
        l = self.l
        b = self.b
        th = X[2]
        T = np.array([[-np.sin(th)/m,np.cos(th)/m,0],[-np.cos(th)/m,-np.sin(th)/m,0],[0,0,0]])
        H = np.array([[-1,-1,0,0,1,1,0,0],[0,0,-1,-1,0,0,1,1],[-l,l,-b,b,-l,l,-b,b]])
        dBdX = np.vstack((np.zeros((3,8)),T@H))
        return dBdX
    
    def rk4(self,t,X,U):
        h = self.h
        k1 = self.dynamics(t,X,U)
        k2 = self.dynamics(t+h/2.,X+k1*h/2.,U)
        k3 = self.dynamics(t+h/2.,X+k2*h/2.,U)
        k4 = self.dynamics(t+h,X+k3*h,U)
        return t+h,X+h*(k1+2.*k2+2.*k3+k4)/6.

    def one_step_sim(self,t,X,U):
        runge_num = self.runge_num
        for num in range(0, runge_num): #self.X, self.tを更新することで全体で更新
             t,X = self.rk4(t,X,U)
        return t,X

    def NominalControl(self,states,Xf):
        Xd = Xf
        A = self.A
        B = self.GetB(states)
        Q = 1*np.identity(states.size)
        R = 1*np.identity(B.shape[1])
        K,P,E = control.lqr(A,B,Q,R)
        X = np.array([states]).T
        Xdv = np.array([Xd]).T
        u = -K@(X-Xdv)
        u = np.ravel(u)
        return u
    
    def NominalTrajectory(self,X0,Xf):
        N = self.N
        B = self.GetB(X0)
        X = X0
        Xhis = np.zeros((N+1,X0.size))
        Xhis[0,:] = X0
        t = 0
        Uhis = np.zeros((N,B.shape[1]))
        for i in range(N):
            U = self.NominalControl(X,Xf)
            t,X = self.one_step_sim(t,X,U)
            Xhis[i+1,:] = X
            Uhis[i,:] = U
        return Xhis,Uhis
    
    def SCPnormalized(self,X0,Xf):
        dt = self.dt
        N = self.N
        B = self.GetB(X0)
        fmin = self.fmin
        fmax = self.fmax
        Xhis,Uhis = self.NominalTrajectory(X0,Xf)
        Tregion = 6
        for i in range(10):
            XX = cp.Variable((N+1,X0.size))
            UU = cp.Variable((N,B.shape[1]))
            p = cp.Variable((6))
            constraints = [XX[0,:] == X0]
            constraints += [p == XX[N,:]-Xf]
            for k in range(N):
                Xkp1 = XX[k+1,:]
                Xk = XX[k,:]
                Uk = UU[k,:]
                constraints += [fmin <= Uk, Uk <= fmax]
                Xs_k = Xhis[k,:]
                Us_k = Uhis[k,:]
                #constraints += [cp.norm(Xk-Xs_k) <= Tregion]
                constraints += [Xkp1 == self.DynamicsLD(dt,Xk,Uk,Xs_k,Us_k)]
            prob = cp.Problem(cp.Minimize(cp.sum_squares(UU)+100*cp.norm(p)**2),constraints)
            prob.solve(solver=cp.MOSEK)
            print(prob.status)
            print(np.linalg.norm(p.value))
            Xhis = XX.value
            Uhis = UU.value
            #this,Xhis,Uhis = self.FinalTrajectory(UU.value)        
        self.XXd = XX.value
        self.UUd = UU.value
        return XX.value,UU.value
    
    def DynamicsLD(self,dt,Xk,Uk,Xs_k,Us_k):
        th_k = Xk[2]
        ths_k = Xs_k[2]
        A = self.A
        Bs = self.GetB(Xs_k)
        dBdths = self.GetdBdX(Xs_k)
        dXdt = A@Xk+Bs@Uk+(dBdths*(th_k-ths_k))@Us_k
        Xkp1 = Xk+dXdt*dt
        return Xkp1
    
    def FinalTrajectory(self,X0):
        UUd = self.UUd
        N = self.N
        B = self.GetB(X0)
        t = 0
        X = X0
        this = np.zeros(N+1)
        Xhis = np.zeros((N+1,X0.size))
        this[0] = t
        Xhis[0,:] = X
        Uhis = np.zeros((N,B.shape[1]))
        for i in range(N):
            U = UUd[i]
            t,X = self.one_step_sim(t,X,U)
            this[i+1] = t
            Xhis[i+1,:] = X
            Uhis[i,:] = U
        return this,Xhis,Uhis
    
    def GetSCPs(self):
        X0s = self.X0s
        Xfs = self.Xfs
        XXds = {}
        UUds = {}
        for i in range(X0s.shape[0]):
            X0 = np.zeros(6)
            Xf = np.zeros(6)
            X0[0:2] = X0s[i,:]
            Xf[0:2] = Xfs[i,:]
            XXd,UUd = self.SCPnormalized(X0,Xf)
            this,Xhis,Uhis = self.FinalTrajectory(X0)
            XXds[i] = Xhis
            UUds[i] = Uhis
        return this,XXds,UUds
        
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
    dtra = DesiredTrajectories(X0s,Xfs)
    this,XXds,UUds = dtra.GetSCPs()
    plt.figure()
    for i in range(X0s.shape[0]):
        Xhis = XXds[i]
        plt.plot(Xhis[:,0],Xhis[:,1])
    plt.axes().set_aspect('equal')
    plt.show()
    np.save('data/params/desireds/this.npy',this)
    SaveDict('data/params/desireds/XXds.pkl',XXds)
    SaveDict('data/params/desireds/UUds.pkl',UUds)