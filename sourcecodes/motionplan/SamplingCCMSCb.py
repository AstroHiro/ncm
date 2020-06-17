#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 29 22:43:52 2020

@author: hiroyasu
"""

import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import control
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
Robs = scp.Robs
Rsafe = scp.Rsafe
XOBSs = scp.XOBSs
XXd0 = np.load('data/params/desired_n/Xhis.npy')
UUd0 = np.load('data/params/desired_n/Uhis.npy')
dratio = 0.15
d_over = np.sqrt(3)*dratio
X0 = scp.X0
Xf = scp.Xf

class CCM:
    def __init__(self,XXd0,UUd0,XXds,UUds,tspan=TSPAN,dt=DT,runge_num=RungeNum,m=M,I=II,l=L,b=bb,A=AA,fmin=FMIN,fmax=FMAX):
        self.XXd = XXd0
        self.UUd = UUd0
        self.XXds = XXds
        self.UUds = UUds
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
        prob = cp.Problem(cp.Minimize(chi+nu),constraints)
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

    def GetCCM2(self):
        dt = self.dt
        epsilon = 0.
        XX = self.XXd
        N = XX.shape[0]-1
        I = np.identity(6)
        WW = {}
        for i in range(N+1):
          WW[i] = cp.Variable((6,6),PSD=True)
        nu = cp.Variable(nonneg=True)
        pp = cp.Variable((2,2),symmetric=True)
        tau = pp[0,0]
        chi = pp[1,0]
        alp = pp[1,1]
        constraints = [chi*I-WW[0] >> epsilon*I,WW[0]-I >> epsilon*I]
        constraints += [tau >= 0, alp >= 0, chi >= 0]
        constraints += [pp >> epsilon*np.identity(2)]
        for k in range(N):
            Xk = XX[k,:]
            Ax = self.A
            Bx = self.GetB(Xk)
            Wk = WW[k]
            Wkp1 = WW[k+1]
            constraints += [-2*alp*I-(-(Wkp1-Wk)/dt+Ax@Wk+Wk@Ax.T-2*nu*Bx@Bx.T) >> epsilon*I]
            constraints += [chi*I-Wkp1 >> epsilon*I,Wkp1-I >> epsilon*I]
        prob = cp.Problem(cp.Minimize(tau),constraints)
        prob.solve()
        cvx_status = prob.status
        print(cvx_status)
        cvx_optval = tau.value
        WWout = {}
        MMout = {}
        for i in range(N+1):
          WWout[i] = WW[i].value/nu.value
          MMout[i] = np.linalg.inv(WWout[i])
        chi = chi.value
        nu = nu.value
        tau = tau.value
        alp = alp.value
        return cvx_status,cvx_optval,WWout,MMout,chi,nu,tau,alp

    def CLFQP(self,X,Xd,M,Ud,alp):
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
        constraints = [evec.T@(M@A+A.T@M)@evec+2*evec.T@M@Bx@U-2*evec.T@M@Bxd@Ud <= -2*alp*evec.T@M@evec+p]
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
        X1 = X0
        X2 = X0
        Xd = XXd[0,:]
        XdRCT = XXdRCT[0,:]
        this = np.zeros(N+1)
        X1his = np.zeros((N+1,X0.size))
        X2his = np.zeros((N+1,X0.size))
        this[0] = t
        X1his[0,:] = X1
        U1his = np.zeros((N,B.shape[1]))
        X2his[0,:] = X2
        U2his = np.zeros((N,B.shape[1]))
        for i in range(N):
            M = MM[i]
            A = self.A
            Bx1 = self.GetB(X1)
            Bx2 = self.GetB(X2)
            Q = 0.1*np.identity(6)
            R = 1*np.identity(8)
            K,P,E = control.lqr(A,Bx1,Q,R)
            #U = UUd[i,:]-Bx.T@M@(X-Xd)
            U1 = self.CLFQP(X1,Xd,M,UUd[i,:],alp)
            U2 = self.CLFQP(X2,XdRCT,M,UUdRCT[i,:],alp)
            t,X1 = self.one_step_sim(t,X1,U1)
            t1,X2 = self.one_step_sim(t1,X2,U2)
            Xd = XXd[i+1,:]
            XdRCT = XXdRCT[i+1,:]
            d1 = np.hstack((np.zeros(3),(np.random.rand(3)*2-1)))*dratio*10
            #d1 = (np.random.rand(6)*2-1)*dratio
            X1 = X1+d1*dt
            X2 = X2+d1*dt
            this[i+1] = t
            X1his[i+1,:] = X1
            U1his[i,:] = U1
            X2his[i+1,:] = X2
            U2his[i,:] = U2
        return this,X1his,U1his,X2his,U2his
    
    def GetOptimalCCMs(self,alp):
        XXds = self.XXds
        UUds = self.UUds
        numX0 = len(XXds)
        optvals = np.zeros(numX0)
        chi_s = np.zeros(numX0)
        nu_s = np.zeros(numX0)
        status_str = []
        WWs = {}
        for iX0 in range(numX0):
            print('iX0 = ',iX0)        
            self.XXd = XXds[iX0]
            self.UUd = UUds[iX0]
            cvx_status,cvx_optval,WW,MM,chi,nu = self.GetCCM(alp)
            optvals[iX0] = cvx_optval/alp
            chi_s[iX0] = chi
            nu_s[iX0] = nu
            WWs[iX0] = WW
            status_str.append(cvx_status)
        self.WWs = WWs
        XXs = self.XXds
        return optvals,chi_s,nu_s,status_str,XXs,WWs  
    
    def SaveTrainingData(self):
        XXs = self.XXds
        WWs = self.WWs
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
                    cMhisNN[i+(N+1)*j,int(1/2*ii*(ii+1)):int(1/2*(ii+1)*(ii+2))] = di
        return XhisNN,MMs,WhisNN,MhisNN,cMhisNN

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
    np.random.seed(seed=19940808)
    # np.random.seed(seed=6881994)
    #np.random.seed(seed=1995226)
    XXds = LoadDict('data/params/desireds/XXds.pkl')
    UUds = LoadDict('data/params/desireds/UUds.pkl')
    ccm = CCM(XXd0,UUd0,XXds,UUds)
    '''
    # macro alpha
    alp_list = np.array([0.01,0.1,0.5,1,1.5,2,3,10])
    a_len = alp_list.size
    optvals = np.zeros(a_len)
    for i in range(a_len):
        print(i)
        alp = alp_list[i]
        cvx_status,cvx_optval,WW,MM,chi,nu = ccm.GetCCM(alp)
        optvals[i] = cvx_optval
    plt.figure()
    plt.semilogx(alp_list,optvals)
    alp = alp_list[np.argmin(optvals)]
    np.save('data/params/alpha_macro/alp.npy',alp)
    np.save('data/params/alpha_macro/alp_list.npy',alp_list)
    np.save('data/params/alpha_macro/optvals.npy',optvals)
    '''
    
    '''
    # micro alpha
    alp_list = np.arange(0.1,1.5,0.1)
    a_len = alp_list.size
    optvals = np.zeros(a_len)
    for i in range(a_len):
        print(i)
        alp = alp_list[i]
        cvx_status,cvx_optval,WW,MM,chi,nu = ccm.GetCCM(alp)
        optvals[i] = cvx_optval
    plt.figure()
    plt.semilogx(alp_list,optvals)
    alp = alp_list[np.argmin(optvals)]
    np.save('data/params/alpha_micro/alp.npy',alp)
    np.save('data/params/alpha_micro/alp_list.npy',alp_list)
    np.save('data/params/alpha_micro/optvals.npy',optvals)
    '''
    
    '''
    # mmicro alpha
    alp_list = np.arange(0.5,0.71,0.01)
    a_len = alp_list.size
    optvals = np.zeros(a_len)
    for i in range(a_len):
        print(i)
        alp = alp_list[i]
        cvx_status,cvx_optval,WW,MM,chi,nu = ccm.GetCCM(alp)
        optvals[i] = cvx_optval
    plt.figure()
    plt.semilogx(alp_list,optvals)
    alp = alp_list[np.argmin(optvals)]
    np.save('data/params/alpha_mmicro/alp.npy',alp)
    np.save('data/params/alpha_mmicro/alp_list.npy',alp_list)
    np.save('data/params/alpha_mmicro/optvals.npy',optvals)
    '''
    
    '''
    alp = np.load('data/params/alpha_mmicro/alp.npy')
    optvals,chi_s,nu_s,status_str,XXs,WWs = ccm.GetOptimalCCMs(alp)
    np.save('data/params/optimal/alp.npy',alp)
    np.save('data/params/optimal/optvals.npy',optvals)
    np.save('data/params/optimal/chi_s.npy',chi_s)
    np.save('data/params/optimal/nu_s.npy',nu_s)
    np.save('data/params/optimal/status_str.npy',status_str)
    SaveDict('data/params/optimal/XXs.pkl',XXs)
    SaveDict('data/params/optimal/WWs.pkl',WWs)
    '''
    
    '''
    XXs = LoadDict('data/params/optimal/XXs.pkl')
    WWs = LoadDict('data/params/optimal/WWs.pkl')
    ccm = CCM(XXd0,UUd0,XXs,UUds)
    ccm.WWs = WWs
    ccm.XXds = XXs
    XhisNN,MMs,WhisNN,MhisNN,cMhisNN = ccm.SaveTrainingData()
    np.save('data/training_data/XhisNN.npy',XhisNN)
    np.save('data/training_data/WWs.npy',WWs)
    np.save('data/training_data/MMs.npy',MMs)
    np.save('data/training_data/WhisNN.npy',WhisNN)
    np.save('data/training_data/MhisNN.npy',MhisNN)
    np.save('data/training_data/cMhisNN.npy',cMhisNN)
    '''
    
    '''
    # robust tube desired trajectory
    alp = np.load('data/params/alpha_mmicro/alp.npy')
    dtra = scp.DesiredTrajectories(X0,Xf)
    ccm.XXd = XXd0
    ccm.UUd = UUd0
    cvx_status,cvx_optval,WW,MM,chi,nu = ccm.GetCCM(alp)
    dtra.SCPRCT(alp,chi,d_over)
    this,XXdRCT,UUdRCT = dtra.FinalTrajectory()
    np.save('data/params/desiredRCT/this.npy',this)
    np.save('data/params/desiredRCT/XXdRCT.npy',XXdRCT)
    np.save('data/params/desiredRCT/UUdRCT.npy',UUdRCT)
    '''
    
    alp = np.load('data/params/alpha_mmicro/alp.npy')
    #cvx_status,cvx_optval,WW,MM,chi,nu = ccm.GetCCM(alp)
    XXdRCT = np.load('data/params/desiredRCT/XXdRCT.npy')
    UUdRCT = np.load('data/params/desiredRCT/UUdRCT.npy')
    ccm.XXd = XXd0
    ccm.UUd = UUd0
    this,X1his,U1his,X2his,U2his = ccm.FinalTrajectory(MM,alp,XXdRCT,UUdRCT)
    xxTube1 = GetTubePlot(XXdRCT,alp,chi,d_over,np.pi/2)
    xxTube2 = GetTubePlot(XXdRCT,alp,chi,d_over,-np.pi/2)
    Nplot = xxTube1.shape[0]
    plt.figure()
    plt.plot(X1his[:,0],X1his[:,1])
    plt.plot(X2his[:,0],X2his[:,1])
    #plt.plot(xxTube1[:,0],xxTube1[:,1])
    #plt.plot(xxTube2[:,0],xxTube2[:,1])
    plt.plot(XXdRCT[:,0],XXdRCT[:,1],'--k')
    plt.fill_between(xxTube1[:,0],xxTube1[:,1],np.zeros(Nplot),facecolor='black',alpha=0.2)
    plt.fill_between(xxTube2[:,0],xxTube2[:,1],np.zeros(Nplot),facecolor='white')
    plt.fill_between(np.linspace(19.52,22,100),17.2528*np.ones(100),np.zeros(100),facecolor='white')
    for i in range(XOBSs.shape[0]):
        x,y=[],[]
        for _x in np.linspace(0,2*np.pi):
            x.append(Robs*np.cos(_x)+XOBSs[i,0])
            y.append(Robs*np.sin(_x)+XOBSs[i,1])
            plt.plot(x,y,'k')
    plt.axes().set_aspect('equal')
    plt.show()
    