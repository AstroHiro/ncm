#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 14 19:50:56 2020

@author: hiroyasu
"""

import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import control
import SCPmulti as scp
import pickle
import TrainRNN as trnn
import torch
import pandas as pd

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
Robs = scp.Robs
Rsafe = scp.Rsafe
XOBSs = scp.XOBSs
XXd0 = np.load('data/params/desired_n/Xhis.npy')
UUd0 = np.load('data/params/desired_n/Uhis.npy')
dratio = 0.2
d_over = np.sqrt(3)*dratio
X0 = scp.X0
Xf = scp.Xf
n_input = trnn.n_input
n_hidden = trnn.n_hidden
n_output = trnn.n_output
n_layers = trnn.n_layers

class Spacecraft:
    def __init__(self,XXd0,UUd0,tspan=TSPAN,dt=DT,runge_num=RungeNum,m=M,I=II,l=L,b=bb,A=AA,fmin=FMIN,fmax=FMAX):
        self.XXd = XXd0
        self.UUd = UUd0
        self.tspan = tspan
        self.dt = dt
        self.runge_num = runge_num
        self.h = dt/runge_num
        self.m = m
        self.I = I
        self.l = l
        self.b = b
        self.A = A
        self.fmin = fmin
        self.fmax = fmax
        self.net = trnn.RNN(n_input,n_hidden,n_output,n_layers)
        self.net.load_state_dict(torch.load('data/trained_nets/RNNLorenz.pt'))
        self.net.eval()
        self.YC =  np.load('data/trained_nets/Y_params.npy')
        self.ns = 6
        self.RNN = 1

    def GetP(self,X):
        Xnet = self.Np2Var(X)
        if self.RNN == 1:
            cP = self.net(Xnet.view(1,1,-1))
            cP = cP.data.numpy()*self.YC
            P = self.cP2P(cP[0,0,:])
        else:
            cP = self.net(Xnet)
            cP = cP.data.numpy()
            P = self.cP2P(cP)
        return P
    
    def GetK(self,X):
        P = self.GetP(X)
        B = self.GetB(X)
        K = B.T@P
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
        P = (cPnp.T)@cPnp
        return P
    
    
    def GetPdot(self,X,Ud,Xdp1):
        dX = Xdp1-X
        P = self.GetP(X)
        Pdot = 0
        dXdt = self.dynamics(0,X,Ud)
        for i in range(self.ns):
            dx = np.zeros(self.ns)
            dx[i] = dX[i]
            Pdot += (self.GetP(X+dx)-P)/np.linalg.norm(dx)*dXdt[i]
        return Pdot
    
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
    
    def rk4(self,t,X,U):
        h = self.h
        k1 = self.dynamics(t,X,U)
        k2 = self.dynamics(t+h/2.,X+k1*h/2.,U)
        k3 = self.dynamics(t+h/2.,X+k2*h/2.,U)
        k4 = self.dynamics(t+h,X+k3*h,U)
        return t+h,X+h*(k1+2.*k2+2.*k3+k4)/6.

    def one_step_sim(self,t,X,U):
        runge_num = self.runge_num
        for num in range(0, runge_num):
             t,X = self.rk4(t,X,U)
        return t,X

    def GetCCM(self,alp):
        dt = self.dt
        epsilon = 0.
        XX = self.XXd
        N = XX.shape[0]-1
        I = np.identity(6)
        WW = {}
        for i in range(N+1):
          WW[i] = cp.Variable((6,6),PSD=True)
        nu = cp.Variable(nonneg=True)
        chi = cp.Variable(nonneg=True)
        constraints = [chi*I-WW[0] >> epsilon*I,WW[0]-I >> epsilon*I]
        for k in range(N):
            Xk = XX[k,:]
            Ax = self.A
            Bx = self.GetB(Xk)
            Wk = WW[k]
            Wkp1 = WW[k+1]
            constraints += [-2*alp*Wk-(-(Wkp1-Wk)/dt+Ax@Wk+Wk@Ax.T-2*nu*Bx@Bx.T) >> epsilon*I]
            constraints += [chi*I-Wkp1 >> epsilon*I,Wkp1-I >> epsilon*I]
        prob = cp.Problem(cp.Minimize(chi),constraints)
        prob.solve(solver=cp.MOSEK)
        cvx_status = prob.status
        print(cvx_status)
        WWout = {}
        MMout = {}
        for i in range(N+1):
          WWout[i] = WW[i].value/nu.value
          MMout[i] = np.linalg.inv(WWout[i])
        chi = chi.value
        nu = nu.value
        cvx_optval = chi/alp
        return cvx_status,cvx_optval,WWout,MMout,chi,nu

    def CLFQP(self,X,Xd,Xdp1,M,Ud,alp):
        dt = self.dt
        U = cp.Variable((8,1))
        Ud = np.array([Ud]).T
        p = cp.Variable((1,1))
        fmin = self.fmin
        fmax = self.fmax
        A = self.A
        Bx = self.GetB(X)
        Bxd = self.GetB(Xd)
        evec = np.array([X-Xd]).T
        Mdot = self.GetPdot(X,Ud,Xdp1)
        constraints = [evec.T@(Mdot+M@A+A.T@M)@evec+2*evec.T@M@Bx@U-2*evec.T@M@Bxd@Ud <= -2*alp*evec.T@M@evec+p]
        for i in range(8):
            constraints += [U[i,0] <= fmax, U[i,0] >= fmin]
        prob = cp.Problem(cp.Minimize(cp.sum_squares(U-Ud)+p**2),constraints)
        prob.solve()
        cvx_status = prob.status
        U = U.value
        U = np.ravel(U)
        return U

    def FinalTrajectory(self,MM,alp,XXdRCT,UUdRCT):
        dt = self.dt
        XXd = self.XXd
        UUd = self.UUd
        N = UUd.shape[0]
        X0 = XXd[0,:]
        B = self.GetB(X0)
        t = 0
        t1 = 0
        t2 = 0
        X1 = X0
        X2 = X0
        X3 = X0
        Xd = XXd[0,:]
        XdRCT = XXdRCT[0,:]
        this = np.zeros(N+1)
        X1his = np.zeros((N+1,X0.size))
        X2his = np.zeros((N+1,X0.size))
        X3his = np.zeros((N+1,X0.size))
        this[0] = t
        X1his[0,:] = X1
        U1his = np.zeros((N,B.shape[1]))
        X2his[0,:] = X2
        U2his = np.zeros((N,B.shape[1]))
        X3his[0,:] = X3
        U3his = np.zeros((N,B.shape[1]))
        U1hisN = np.zeros(N)
        U2hisN = np.zeros(N)
        U3hisN = np.zeros(N)
        U1hisND = np.zeros(N)
        U2hisND = np.zeros(N)
        U3hisND = np.zeros(N)
        dnMs = np.zeros(N)
        dnUs = np.zeros(N)
        dnXs = np.zeros(N)
        for i in range(N):
            M = MM[i]
            A = self.A
            Bx3 = self.GetB(X3)
            Q = 2.4*np.identity(6)
            R = 1*np.identity(8)
            K,P,E = control.lqr(A,Bx3,Q,R)
            P1 = self.GetP(X1)
            U3 = UUd[i,:]-Bx3.T@P@(X3-Xd)
            for j in range(8):
                if U3[j] >= self.fmax:
                    U3[j] = self.fmax
                elif U3[j] <= self.fmin:
                    U3[j] = self.fmin
            #U1 = self.CLFQP(X1,Xd,XXd[i+1,:],P,UUd[i,:],alp)
            U1 = self.CLFQP(X1,XdRCT,XXdRCT[i+1,:],P1,UUdRCT[i,:],alp)
            U2 = self.CLFQP(X2,XdRCT,XXdRCT[i+1,:],M,UUdRCT[i,:],alp)
            t,X1 = self.one_step_sim(t,X1,U1)
            t1,X2 = self.one_step_sim(t1,X2,U2)
            t2,X3 = self.one_step_sim(t2,X3,U3)
            Xd = XXd[i+1,:]
            XdRCT = XXdRCT[i+1,:]
            d1 = np.random.choice(np.array([-1,1]),1)[0]
            d2 = np.random.choice(np.array([-1,1]),1)[0]
            d3 = np.random.choice(np.array([-1,1]),1)[0]
            d1 = np.array([0,0,0,d1,d2,d3])*2
            #d1 = 0
            #d1 = np.hstack((np.zeros(3),(np.random.rand(3)*2-1)))*dratio*20
            #d1 = (np.random.rand(6)*2-1)*0.1
            X1 = X1+d1*dt
            X2 = X2+d1*dt
            X3 = X3+d1*dt
            this[i+1] = t
            X1his[i+1,:] = X1
            U1his[i,:] = U1
            X2his[i+1,:] = X2
            U2his[i,:] = U2
            X3his[i+1,:] = X3
            U3his[i,:] = U3
            U1hisN[i] = np.linalg.norm(U1)
            U2hisN[i] = np.linalg.norm(U2)
            U3hisN[i] = np.linalg.norm(U3)
            U1hisND[i] = np.linalg.norm(U1-UUd[i,:])
            U2hisND[i] = np.linalg.norm(U2-UUdRCT[i,:])
            U3hisND[i] = np.linalg.norm(U3-UUdRCT[i,:])
            dnMs[i] = np.linalg.norm(M-P1,ord=2)
            dnUs[i] = np.linalg.norm(U1-U2)
            dnXs[i] = np.linalg.norm(X1-X2)
        return this,X1his,U1his,X2his,U2his,X3his,U3his,U1hisN,U2hisN,U3hisN,U1hisND,U2hisND,U3hisND,dnMs,dnUs,dnXs

def Rot2d(th):
    R = np.array([[np.cos(th),-np.sin(th)],[np.sin(th),np.cos(th)]])
    return R
    
def GetTubePlot(XXdRCT,alp,chi,d_over,th):
    Rtube = d_over*np.sqrt(chi)/alp
    xxdRCT = XXdRCT[:,0:2]
    dxxdRCT = np.diff(xxdRCT,axis=0)
    for i in range(dxxdRCT.shape[0]):
        dxxdRCT[i,:] = Rot2d(th)@(dxxdRCT[i,:]/np.linalg.norm(dxxdRCT[i,:]))
    return xxdRCT[0:dxxdRCT.shape[0],:]+dxxdRCT*Rtube

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
    sc = Spacecraft(XXd0,UUd0)
    alp = np.load('data/params/alpha_mmicro/alp.npy')
    XXdRCT = np.load('data/params/desiredRCT/XXdRCT.npy')
    UUdRCT = np.load('data/params/desiredRCT/UUdRCT.npy')
    np.random.seed(seed=32)
    cvx_status,cvx_optval,WW,MM,chi,nu = sc.GetCCM(alp)
    this,X1his,U1his,X2his,U2his,X3his,U3his,U1hisN,U2hisN,U3hisN,U1hisND,U2hisND,U3hisND,dnMs,dnUs,dnXs = sc.FinalTrajectory(MM,alp,XXdRCT,UUdRCT)
    xxTube1 = GetTubePlot(XXdRCT,alp,chi,d_over,np.pi/2)
    xxTube2 = GetTubePlot(XXdRCT,alp,chi,d_over,-np.pi/2)
    Nplot = xxTube1.shape[0]
    plt.figure()
    plt.plot(X1his[:,0],X1his[:,1])
    plt.plot(X2his[:,0],X2his[:,1])
    plt.plot(X3his[:,0],X3his[:,1])
    #plt.plot(xxTube1[:,0],xxTube1[:,1])
    #plt.plot(xxTube2[:,0],xxTube2[:,1])
    plt.plot(XXdRCT[:,0],XXdRCT[:,1],'--k')
    plt.fill_between(xxTube1[:,0],xxTube1[:,1],np.zeros(Nplot),facecolor='black',alpha=0.2)
    plt.fill_between(xxTube2[:,0],xxTube2[:,1],np.zeros(Nplot),facecolor='white')
    plt.fill_between(np.linspace(19.347,22,100),16.9*np.ones(100),np.zeros(100),facecolor='white')
    plt.fill_between(np.linspace(-1.03228,1.03228,100),-0.2*np.ones(100),np.zeros(100),facecolor='black',alpha=0.2)
    for i in range(XOBSs.shape[0]):
        x,y=[],[]
        for _x in np.linspace(0,2*np.pi):
            x.append(Robs*np.cos(_x)+XOBSs[i,0])
            y.append(Robs*np.sin(_x)+XOBSs[i,1])
            plt.plot(x,y,'k')
    plt.axes().set_aspect('equal')
    plt.show()
    
    
    data1 = {'score': U1hisN}
    data2 = {'score': U2hisN}
    data3 = {'score': U3hisN}
    # Create dataframe
    df1 = pd.DataFrame(data1)
    df2 = pd.DataFrame(data2)
    df3 = pd.DataFrame(data3)
    dfU1hisN = df1.rolling(window=50).mean()
    dfU2hisN = df2.rolling(window=50).mean()
    dfU3hisN = df3.rolling(window=50).mean()
    
    plt.figure()
    plt.plot(this[0:500],np.cumsum(U1hisN**2)*DT)
    plt.plot(this[0:500],np.cumsum(U2hisN**2)*DT)
    plt.plot(this[0:500],np.cumsum(U3hisN**2)*DT)
    plt.show()
    
    plt.figure()
    plt.plot(this[0:500],U1hisND)
    plt.plot(this[0:500],U2hisND)
    plt.plot(this[0:500],U3hisND)
    plt.show()
    
    plt.figure()
    plt.plot(this[0:500],U1his)
    plt.plot(this[0:500],U2his)
    plt.plot(this[0:500],U3his)
    plt.show()
    
    this,X1his,U1his,X2his,U2his,X3his,U3his,U1hisN,U2hisN,U3hisN,U1hisND,U2hisND,U3hisND
    np.save('data/simulation/this.npy',this)
    np.save('data/simulation/X1his.npy',X1his)
    np.save('data/simulation/U1his.npy',U1his)
    np.save('data/simulation/X2his.npy',X2his)
    np.save('data/simulation/U2his.npy',U2his)
    np.save('data/simulation/X3his.npy',X3his)
    np.save('data/simulation/U3his.npy',U3his)
    np.save('data/simulation/U1hisN.npy',U1hisN)
    np.save('data/simulation/U2hisN.npy',U2hisN)
    np.save('data/simulation/U3hisN.npy',U3hisN)
    np.save('data/simulation/U1hisND.npy',U1hisND)
    np.save('data/simulation/U2hisND.npy',U2hisND)
    np.save('data/simulation/U3hisND.npy',U3hisND)
    np.save('data/simulation/xxTube1.npy',xxTube1)
    np.save('data/simulation/xxTube2.npy',xxTube2)
    '''
    dnMs_ave = []
    dnUs_ave = []
    dnXs_ave = []
    for r in range(100):
        np.random.seed(seed=r)
        this,X1his,U1his,X2his,U2his,X3his,U3his,U1hisN,U2hisN,U3hisN,U1hisND,U2hisND,U3hisND = sc.FinalTrajectory(MM,alp,XXdRCT,UUdRCT)
        np.sum(dnMs)*dT/TSPAN[1]
    '''