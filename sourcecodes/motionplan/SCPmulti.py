#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 16:59:34 2020

@author: hiroyasu
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 13:46:00 2020

@author: Hiroyasu
"""
import numpy as np
import control
import cvxpy as cp
import matplotlib.pyplot as plt

DT = 0.1
TSPAN = np.array([0,50])
#M = 19.890
#II = 0.68
#L = 0.4572
#bb = 0.4572
M = 1
II = 1
L = 1
bb = 1
FMIN = 0
FMAX = 1
RungeNum = 10
Z = np.zeros((3,3))
eye3 = np.identity(3)
AA = np.vstack((np.hstack((Z,eye3)),np.hstack((Z,Z))))
Robs = 3
Rsafe = 0.
XOBSs = np.array([[0,11],[5,3],[8,11],[13,3],[16,11],[21,3]])
X0 = np.array([0,0,np.pi/12,0,0,0])
Xf = np.array([20,18,0,0,0,0])

class DesiredTrajectories: 
    def __init__(self,X0,Xf,dt=DT,tspan=TSPAN,runge_num=RungeNum,m=M,I=II,l=L,b=bb,fmin=FMIN,fmax=FMAX,A=AA,Xobs=XOBSs,robs=Robs,rsafe=Rsafe):
        self.X0  = X0
        self.Xf  = Xf
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
        self.Xobs = Xobs
        self.robs = robs
        self.rsafe = Rsafe

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
        for num in range(0, runge_num):
             t,X = self.rk4(t,X,U)
        return t,X

    def NominalControl(self,states):
        Xd = self.Xf
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
    
    def NominalTrajectory(self):
        N = self.N
        X0 = self.X0
        B = self.GetB(X0)
        X = X0
        Xhis = np.zeros((N+1,X0.size))
        Xhis[0,:] = X0
        t = 0
        Uhis = np.zeros((N,B.shape[1]))
        for i in range(N):
            U = self.NominalControl(X)
            t,X = self.one_step_sim(t,X,U)
            Xhis[i+1,:] = X
            Uhis[i,:] = U
        return Xhis,Uhis
    
    def DynamicsLD(self,dt,Xk,Uk,Xs_k,Us_k):
        th_k = Xk[2]
        ths_k = Xs_k[2]
        A = self.A
        Bs = self.GetB(Xs_k)
        dBdths = self.GetdBdX(Xs_k)
        dXdt = A@Xk+Bs@Uk+(dBdths*(th_k-ths_k))@Us_k
        Xkp1 = Xk+dXdt*dt
        return Xkp1
    
    def SCPnormalized(self,Xhis,Uhis,R,epochs,idxobs,Tidx,Xfidx):
        dt = self.dt
        N = self.N
        X0 = self.X0
        Xf = self.Xf
        B = self.GetB(X0)
        fmin = self.fmin
        fmax = self.fmax
        Tregion = 0.1
        for i in range(epochs):
            XX = cp.Variable((N+1,X0.size))
            UU = cp.Variable((N,B.shape[1]))
            p = cp.Variable((6))
            p2 = cp.Variable(N)
            constraints = [XX[0,:] == X0]
            if Xfidx == 0:
                constraints += [p == XX[N,:]-Xf]
            elif Xfidx == 1:
                constraints += [XX[N,:] == Xf]
            for k in range(N):
                Xkp1 = XX[k+1,:]
                Xk = XX[k,:]
                Uk = UU[k,:]
                constraints += [fmin <= Uk, Uk <= fmax]
                Xs_k = Xhis[k,:]
                Us_k = Uhis[k,:]
                C = self.SafetyConstraint(Xk,Xs_k,R)
                if idxobs == 1:
                    constraints += [-C[idxobs-1] <= p2[k]]
                elif idxobs > 1:
                    for j in range(idxobs-1):
                        constraints += [-C[j] <= 0]
                    constraints += [-C[idxobs-1] <= p2[k]]
                if Tidx == 1:
                    constraints += [cp.norm(Xk-Xs_k) <= Tregion]
                constraints += [Xkp1 == self.DynamicsLD(dt,Xk,Uk,Xs_k,Us_k)]
            prob = cp.Problem(cp.Minimize(cp.sum_squares(UU)+100*cp.norm(p)**2+100*cp.norm(p2)**2),constraints)
            prob.solve(solver=cp.MOSEK)
            print(prob.status)
            #print(np.linalg.norm(p.value))
            print(np.linalg.norm(p2.value))
            Xhis = XX.value
            Uhis = UU.value
        return XX.value,UU.value
    
    def SCP0(self):
        R = self.robs+self.rsafe
        XXd0,UUd0 = self.NominalTrajectory()
        XXd,UUd = XXd0,UUd0  
        for i in range(3):
            XXd,UUd = self.SCPnormalized(XXd,UUd,R,10,i+1,0,0)
        XXdF,UUdF = self.SCPnormalized(XXd,UUd,R,25,5,1,1)
        self.XXd = XXdF
        self.UUd = UUdF
        pass
    
    def SCPRCT(self,alp,chi,d_over):
        Rtube = d_over*np.sqrt(chi)/alp
        R = self.robs+self.rsafe+Rtube
        XXd0,UUd0 = self.NominalTrajectory()
        XXd,UUd = XXd0,UUd0  
        for i in range(3):
            XXd,UUd = self.SCPnormalized(XXd,UUd,R,10,i+1,0,0)
        XXdF,UUdF = self.SCPnormalized(XXd,UUd,R,50,5,1,1)
        self.XXd = XXdF
        self.UUd = UUdF
        pass
    
    def SafetyConstraint(self,Xk,Xs_k,R):
        Xobs = self.Xobs
        C = {}
        for i in range(Xobs.shape[0]):
            C[i] = (Xs_k[0:2]-Xobs[i,:])@(Xk[0:2]-Xobs[i,:])-R*cp.atoms.norm(Xs_k[0:2]-Xobs[i,:])
        return C
    
    def FinalTrajectory(self):
        UUd = self.UUd
        N = self.N
        X0 = self.X0
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
    
    
if __name__ == "__main__":
    dtra = DesiredTrajectories(X0,Xf)
    dtra.SCP0()
    #dtra.SCPRCT(1,0.5,1)
    this,Xhis,Uhis = dtra.FinalTrajectory()
    plt.figure()
    for i in range(XOBSs.shape[0]):
        x,y=[],[]
        for _x in np.linspace(0,2*np.pi):
            x.append(Robs*np.cos(_x)+XOBSs[i,0])
            y.append(Robs*np.sin(_x)+XOBSs[i,1])
            plt.plot(x,y)
    plt.plot(Xhis[:,0],Xhis[:,1])
    plt.axes().set_aspect('equal','datalim')
    plt.show()
    np.save('data/params/desired_n/this.npy',this)
    np.save('data/params/desired_n/Xhis.npy',Xhis)
    np.save('data/params/desired_n/Uhis.npy',Uhis)
