#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 29 08:41:48 2019

@author: hiroyasu
"""

import cvxpy as cp
import numpy as np
import torch
import control
import matplotlib.pyplot as plt

M = 1;
II = 1
L = 1
bl = 1
FMIN = 0
FMAX = 1
RungeRum = 10
RHO = 1
AA = np.vstack((np.hstack((np.zeros((3,3)),np.eye(3))),np.hstack((np.zeros((3,3)),np.zeros((3,3))))))

class Spacecraft: 
    def __init__(self,t,X,dt,runge_num=RungeRum,m=M,I=II,l=L,b=bl,A=AA):
        self.t = t
        self.X = X
        self.runge_num = runge_num
        self.h = dt/runge_num
        self.m = m
        self.I = I
        self.l = l
        self.b = b
        self.A = A

    def dynamics(self,t,states,inputs):
        A = self.A
        B = self.GetB(states)
        Xv = np.transpose(np.array([states]))
        Uv = np.transpose(np.array([inputs]))
        dXdt = A.dot(Xv)+B.dot(Uv)
        dXdt = dXdt[:,0]
        return dXdt
    
    def GetA(self,states):
        return AA
    
    def GetB(self,states):
        m = self.m
        I = self.I
        l = self.l
        b = self.b
        th = states[2]
        T = np.array([[np.cos(th)/m,np.sin(th)/m,0],[-np.sin(th)/m,np.cos(th)/m,0],[0,0,1/2./I]])
        H = np.array([[-1,-1,0,0,1,1,0,0],[0,0,-1,-1,0,0,1,1],[-l,l,-b,b,-l,l,-b,b]])
        B = np.vstack((np.zeros((3,8)),T.dot(H)))
        return B
    
    def rk4(self,U):
        t = self.t
        X = self.X
        h = self.h
        k1 = self.dynamics(t,X,U)
        k2 = self.dynamics(t+h/2.,X+k1*h/2.,U)
        k3 = self.dynamics(t+h/2.,X+k2*h/2.,U)
        k4 = self.dynamics(t+h,X+k3*h,U)
        return t+h,X+h*(k1+2.*k2+2.*k3+k4)/6.

    def one_step_sim(self,U):
        runge_num = self.runge_num
        for num in range(0, runge_num): #self.X, self.tを更新することで全体で更新
             self.t,self.X = self.rk4(U)
        return self.t,self.X

class Controller: 
    def __init__(self,dt,Ax,Bx,rho=RHO,runge_num=RungeRum,fmin=FMIN,fmax=FMAX):
        self.dt = dt
        self.Ax = Ax
        self.Bx = Bx
        self.runge_num = runge_num
        self.h = dt/runge_num
        self.rho = rho
        self.ns = 6
        self.net = torch.load('data/TrainedNets/RNNsc.pt')
        self.XXd = np.genfromtxt('data/MotionPlan/XhisNN.csv', delimiter=',')
        self.UUd = np.genfromtxt('data/MotionPlan/UhisNN.csv', delimiter=',')
        
    def GetU(self,X,kstep):
        K = self.GetK(X)
        Xv = np.transpose(np.array([X]))
        Xd = self.XXd[kstep,:]
        Xdv = np.transpose(np.array([Xd]))
        Ud = self.UUd[kstep,:]
        Udv = np.transpose(np.array([Ud]))
        U = -K@(Xv-Xdv)+Udv
        U = U[:,0]
        return U
        
    def GetP(self,X):
        Pnet = self.Np2Var(X)
        cP = self.net(Pnet.view(1,1,-1))
        cP = cP.data.numpy()
        P = self.cP2P(cP[0,0,:])
        return P
    
    def GetK(self,X):
        B = self.Bx(X)
        rho = self.rho
        Bt = np.transpose(B)
        P = self.GetP(X)
        K = 1/2*rho*Bt@P
        return K
    
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
        P = np.transpose(cPnp).dot(cPnp)
        return P
    
        
        
if __name__ == "__main__":
    Q0 = np.zeros((6,6))
    t0 = 0
    X0 = np.array([1,2,np.pi/6,0,0,0])
    dt = 0.1
    tf = 50
    N = np.floor((tf-t0)/dt)
    N = N.astype(np.int32)
    sc = Spacecraft(t0,X0,dt)
    Ax = sc.GetA
    Bx = sc.GetB
    con = Controller(dt,Ax,Bx)
    this = t0
    this = np.array([t0])
    Xhis = X0
    ehis = np.array([np.linalg.norm(X0)])
    X = X0
    for i in range(N):
        A = Ax(X)
        B = Bx(X)
        U = con.GetU(X,i)
        t,X = sc.one_step_sim(U)
        this = np.hstack((this,t))
        Xhis = np.vstack((Xhis,X))
        e = np.linalg.norm(X)
        ehis = np.hstack((ehis,e))
        
        print(e)
        
    fig = plt.figure()
    plt.plot(Xhis[:,0],Xhis[:,1])
    plt.plot(con.XXd[:,0],con.XXd[:,1])
    plt.xlabel('Number of iteration')
    plt.ylabel('Loss')
    plt.show()