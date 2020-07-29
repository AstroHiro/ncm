#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MIT License

Copyright (c) 2020 Hiroyasu Tsukamoto https://hirotsukamoto.com/

Permission is hereby granted, free of charge, to any person obtaining a copy 
of this software and associated documentation files (the "Software"), to deal 
in the Software without restriction, including without limitation the rights 
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell 
copies of the Software, and to permit persons to whom the Software is 
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all 
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR 
IMPLIED, INCLUDING BUT NOT LIMI TED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL 
THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER 
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, 
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE 
SOFTWARE.

"""

import os
import cvxpy as cp
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import matplotlib
from inspect import signature as sig

class NCM:
    def __init__(self,dt,dynamicsf,h_or_g,xlims,alims,iEC,fname,d1_over=0.1,\
                 d2_over=0.1,da=0.1,Nx=1000,Nls=100,plims=np.empty((2,0))):
        """
        This class provides several objects and methods for designing a Neural 
        Contraction Metric (NCM) of a given nonliner dynamical system both for
        state estimation and feedback control.
        See the NCM paper https://arxiv.org/abs/2006.04361 and
        the CV-STEM paper https://arxiv.org/abs/2006.04359 for more details.
        See https://github.com/AstroHiro/ncm/wiki/Documentation for the
        documentation of this class file.
        
        
        Parameters
        (let n: state dimension and m: measurement or control input dimension)
        ----------
        dt : float
            discrete sampling period of CV-STEM
        dynamicsf : function - ndarray (n,n_p) -> (n, )
            vector field of given dynamical system 
            (i.e. f of dx/dt = f(x) or dx/dt = f(x)+g(x)u)
        h_or_g : function - ndarray (n,n_p) -> (m, ) for h, -> (n,m) for g
            measurement equation h or actuation matrix g
            (i.e. h of y = h(x,p) or g of dx/dt  = f(x,p)+g(x,p)u)
        xlims : ndarray - (2,n)
            lower and upper buonds of eash state
            (i.e. xlims[0,:]: lower bounds, xlims[1,:]: upper bounds)
        alims : ndarray - (2, )
            lower and upper bound of contraction rate alpha
            (i.e. alims[0]: lower bound, alims[0]: upper bound)
        iEC : str
            iEC = "est" for estimation and = "con" for control
        fname : str
            file name of your choice for storing NCM models and parameters
        d1_over : float, optional, default is 0.1
            upper bound of process noise
            (i.e. d1_over or d_over in the NCM paper)
        d2_over : float, optional, default is 0.1
            upper bound of measurement noise or penalty on feedback gains
            (i.e. d2_over or lambda in in the NCM paper)
        da : float, optional, default is 0.1
            step size of contraction rate alpha for line search in CV-STEM
        Nx : int, optional, default is 1000
            # samples of CV-STEM to be used for NCM training 
        Nls : int, optional, default is 100
            # samples to be used for line search in CV-STEM
        plims : ndarray - (2,n_p), default is np.empty((2,0))
            lower and upper bound of system parameters
            (i.e. plims[0,:]: lower bounds, plims[0,:]: upper bounds)

        Any other objects to be updated
        -------
        n : int
            state dimension
        m : int
            measurement or control input dimension
        n_p : int
            system parameter dimension
        Afun : function - ndarray (n,n_p) -> (n,n)
            Jacobian of dynamicsf (can be set to state-dependent coefficient
            matrix A s.t. f(x) = A(x)x, see the CV-STEM paper for details)
        Cfun : function - ndarray (n,n_p) -> (n,m), to be used for iEC = "est"
            Jacobian of measurement equation h (can be set to C s.t. 
            h(x) = C(x)x, see the CV-STEM paper for details)
        Bw : function - ndarray (n,n_p) -> (n,k1)
            B(x) given in equation (9) or B_2(x) in equation (17) of the NCM
            paper (B(x) = I and B_2(x) = g(x) are used by default, where g(x)
            is actuation matrix)
        Gw : function - ndarray (n,n_p) -> (m,k2), to be used for iEC = "est"
            G(x) given in equation (9) of the NCM paper (G(x) = I is used by
            default)
        c_over : numpy.float64, to be used for iEC = "est"
            approximate upper bound of Cfun(x) in given state space
        b_over : numpy.float64
            approximate upper bound of Bw(x) in given state space
        g_over : numpy.float64, to be used for iEC = "est"
            approximate upper bound of Gw(x) in given state space
        model : keras neural net model - ndarray (k,n) -> (k,int(n*(n+1)/2))
            function that returns cholesky-decomposed approximate optimal
            contraction metrics (i.e. NCMs) for given k states
        alp_opt : float
            optimal contraction rate
        chi_opt : numpy.float64
            optimal upper bound of condition number of contraction metrics
        nu_opt : numpy.float64
            optimal upper bound of induced 2-norm of contraction metrics
        Jcv_opt : numpy.float64
            optimal steady-state upper bound of estimation or tracking error
        xs_opt : ndarray - (Nx,n)
            randomized state samples
        Ws_opt : list of length Nx
            list containing inverse of ndarray (n,n) optimal contraction
            metrics sampled by CV-STEM
        Ms_opt : list of length Nx
            list containing ndarray (n,n) optimal contraction metrics sampled
            by CV-STEM
        cholMs : list of length Nx
            list containing ndarray (int(n*(n+1)/2), ) optimal contraction 
            metrics sampled by CV-STEM
        Ws : list of length Ncv
            list containing inverse of n by n optimal contraction metrics
            in current instance of CV-STEM
        chi : numpy.float64
            optimal upper bound of condition number of contraction metrics
            in current instance of CV-STEM
        nu : numpy.float64
            optimal upper bound of induced 2-norm of contraction metrics
            in current instance of CV-STEM
        Jcv : numpy.float64
            optimal steady-state upper bound of estimation or tracking error
            in current instance of CV-STEM
        cvx_status : str
            problem status of CV-STEM, "optimal", "infeasible", "unbounded",
            "infeasible_inaccurate", or "unbounded_inaccurate"
        epsilon : float, default is 0.0
            non-negative constant introduced to relax stability condition
        dt_rk : float, default is 0.01
            time step of numerical integration 
            
        """
        self.dt = dt
        if (len(sig(dynamicsf).parameters) == 1):
            self.dynamicsf = lambda x,p: dynamicsf(x)
        else:
            self.dynamicsf = dynamicsf
        if (len(sig(h_or_g).parameters) == 1):
            self.h_or_g =  lambda x,p: h_or_g(x)
        else:
            self.h_or_g = h_or_g
        self.xlims = xlims
        self.alims = alims
        self.iEC = iEC
        self.fname = fname
        self.d1_over = d1_over
        self.d2_over = d2_over
        self.da = da
        self.Nx = Nx
        self.Nls = Nls
        self.n = np.size(xlims,1)
        self.m = np.size(self.h_or_g(xlims[0,:],plims[0,:]).T,0)
        self.n_p = np.size(plims,1)
        self.Afun= lambda x,p: self.jacobian(x,p,self.dynamicsf)
        if self.iEC == "est":
            self.Cfun= lambda x,p: self.jacobian(x,p,self.h_or_g)
            self.Bw = lambda x,p: np.identity(self.n)
            self.Gw = lambda x,p: np.identity(self.m)
        elif self.iEC == "con":
            self.Bw = self.h_or_g
        else:
            raise ValueError('Invalid iEC: iEC = "est" or "con"')
        self.epsilon = 0
        self.dt_rk = 0.01
        self.plims = plims
    
    def ncm(self,x,p):
        """
        Compute Neural Contraction Metric (NCM) M(x,p) at current state x and 
        current system parameter p
        
        
        Parameters
        ----------
        x : ndarray - (n, )
            current state x
        p : ndarray - (n_p, )
            current system parameter

        Returns
        -------
        M : ndarray - (n,n)
            Neural Contraction Metric (NCM)
            
        """
        n = self.n
        n_p = self.n_p
        x = np.reshape(np.hstack((x,p)),(1,n+n_p))
        cholM = self.model.predict(x)
        cholM = np.reshape(cholM,int(n*(n+1)/2))
        M = self.cholM2M(cholM)
        return M
    
    def train(self,iTrain=1,Nbatch=32,Nlayers=3,Nunits=100,\
                 Nepochs=10000,ValidationSplit=0.1,Patience=20):
        """
        Train Neural Contraction Metric (NCM)
        

        Parameters
        ----------
        iTrain : 1 or 0, optional, default is 1
            IdXTrain = 1 for training NCM and = 0 for using pretrained NCM
        Nbatch : int, optional, default is 32
            batch size of NCM training
        Nlayers : int, optional, default is 3
            # layers of NCM 
        Nunits : int, optional, default is 100
            # units of each layers of NCM
        Nepochs : int, optional, default is 10000
            # training epochs
        ValidationSplit : int, optional, default is 0.1
            proportion of training data used as verification data
        Patience : int, optional, default is 20
            # epochs with no improvement after which training will be stopped

        Objects to be updated
        -------
        model : keras neural net model - ndarray (k,n) -> (k,int(n*(n+1)/2))
            function that returns cholesky-decomposed approximate optimal
            contraction metrics (i.e. NCMs) for given k states
            
        When iTrain = 0, follwoing objects will also be updated
        alp_opt : float
            optimal contraction rate
        chi_opt : numpy.float64
            optimal upper bound of condition number of contraction metrics
        nu_opt : numpy.float64
            optimal upper bound of induced 2-norm of contraction metrics
        Jcv_opt : numpy.float64
            optimal steady-state upper bound of estimation or tracking error
            
        """
        if iTrain == 1:
            self.cvstem()
            print("========================================================")
            print("=================== NCM CONSTRUCTION ===================")
            print("========================================================")
            n = self.n
            n_p = self.n_p
            X = np.hstack((self.xs_opt,self.ps_opt))
            y = self.cholMs
            model = Sequential(name="NCM")
            model.add(Dense(Nunits,activation="relu",input_shape=(n+n_p,)))
            for l in range(Nlayers-1):
                model.add(Dense(Nunits,activation="relu"))
            model.add(Dense(int(n*(n+1)/2)))
            model.summary()
            model.compile(loss="mean_squared_error",optimizer="adam")
            es = EarlyStopping(monitor="val_loss",patience=Patience)
            model.fit(X,y,batch_size=Nbatch,epochs=Nepochs,verbose=2,\
                      callbacks=[es],validation_split=ValidationSplit)
            self.model = model
            model.save("models/"+self.fname+".h5")
        elif iTrain == 0:
            print("========================================================")
            print("=================== NCM CONSTRUCTION ===================")
            print("========================================================")
            self.model = load_model("models/"+self.fname+".h5")
            path = "models/optvals/"+self.fname
            self.alp_opt = np.load(path+"/alp_opt.npy")
            self.chi_opt = np.load(path+"/chi_opt.npy")
            self.nu_opt = np.load(path+"/nu_opt.npy")
            self.Jcv_opt = np.load(path+"/Jcv_opt.npy")
            print("Loading pre-trained NCM ...")
            print("Loading pre-trained NCM END")
        else:
            raise ValueError("Invalid iTrain: iTrain = 1 or 0")
        print("========================================================")
        print("================= NCM CONSTRUCTION END =================")
        print("========================================================")
        pass
      
    def cvstem(self):
        """        
        Sample optimal contraction metrics by CV-STEM for constructing NCM


        Objects to be updated
        -------
        c_over : numpy.float64, to be used for iEC = "est"
            Approximate upper bound of Cfun(x) in given state space
        b_over : numpy.float64
            Approximate upper bound of Bw(x) in given state space
        g_over : numpy.float64, to be used for iEC = "est"
            Approximate upper bound of Gw(x) in given state space
        xs_opt : ndarray - (Nx,n), where Nx is # samples to be used for NCM
            randomized state samples
        Ws_opt : list of length Nx
            list containing inverse of ndarray (n,n) optimal contraction
            metrics
        chi_opt : numpy.float64
            optimal upper bound of condition number of contraction metrics
        nu_opt : numpy.float64
            optimal upper bound of induced 2-norm of contraction metrics
        Jcv_opt : numpy.float64
            optimal steady-state upper bound of estimation or tracking error
        
        """
        if (self.iEC == "est") and (len(sig(self.Cfun).parameters) == 1):
            fun1 = self.Cfun
            self.Cfun = lambda x,p: fun1(x)
        if (self.iEC == "est") and (len(sig(self.Gw).parameters) == 1):
            fun2 = self.Gw
            self.Gw = lambda x,p: fun2(x)
        if self.iEC == "est":
            self.c_over = self.matrix_2bound(self.Cfun)
            self.g_over = self.matrix_2bound(self.Gw)
        if (len(sig(self.Bw).parameters) == 1):
            fun3 = self.Bw
            self.Bw = lambda x,p: fun3(x)
        self.b_over = self.matrix_2bound(self.Bw)
        self.linesearch()
        alp = self.alp_opt
        Nx = self.Nx
        Nsplit = 1
        Np = int(Nx/Nsplit)
        Nr = np.remainder(Nx,Nsplit)
        xpmin = np.hstack((self.xlims[0,:],self.plims[0,:]))
        xpmax = np.hstack((self.xlims[1,:],self.plims[1,:]))
        Nxp = self.n+self.n_p
        xps = np.random.uniform(xpmin,xpmax,size=(Nx,Nxp))
        xs_opt,ps_opt,_ = np.hsplit(xps,np.array([self.n,Nxp]))
        Ws_opt = []
        chi_opt = 0
        nu_opt = 0
        print("========================================================")
        print("====== SAMPLING OF CONTRACTION METRICS BY CV-STEM ======")
        print("========================================================")
        for p in range(Np):
            if np.remainder(p,int(Np/10)) == 0:
                print("# sampled metrics: ",p*Nsplit,"...")
            xs_p = xs_opt[Nsplit*p:Nsplit*(p+1),:]
            ps_p = ps_opt[Nsplit*p:Nsplit*(p+1),:]
            self.cvstem0(xs_p,ps_p,alp)
            Ws_opt += self.Ws
            if self.nu >= nu_opt:
                nu_opt = self.nu
            if self.chi >= chi_opt:
                chi_opt = self.chi
        if Nr != 0:
            print("# samples metrics: ",Nx,"...")
            xs_p = xs_opt[Nsplit*(p+1):Nx,:]
            ps_p = ps_opt[Nsplit*(p+1):Nx,:]
            self.cvstem0(xs_p,ps_p,alp)
            Ws_opt += self.Ws
            if self.nu >= nu_opt:
                nu_opt = self.nu
            if self.chi >= chi_opt:
                chi_opt = self.chi
        self.xs_opt = xs_opt
        self.ps_opt = ps_opt
        self.Ws_opt = Ws_opt
        self.chi_opt = chi_opt
        self.nu_opt = nu_opt
        if self.iEC == "est":
            self.Jcv_opt = (self.d1_over*self.b_over*np.sqrt(chi_opt)\
                            +self.d2_over*self.c_over*self.g_over*nu_opt)/alp
            print("Optimal steady-state estimation error =",\
                  "{:.2f}".format(self.Jcv_opt))
        elif self.iEC == "con":
            self.Jcv_opt = self.d1_over*self.b_over*np.sqrt(chi_opt)/alp
            print("Optimal steady-state tracking error =",\
                  "{:.2f}".format(self.Jcv_opt))
        else:
            raise ValueError('Invalid iEC: iEC = "est" or "con"')
        self.M2cholM()
        path = "models/optvals/"+self.fname
        if os.path.exists(path) == False:
            try:
                os.makedirs(path)
            except: 
                raise OSError("Creation of directory %s failed" %path)
            else:
                print ("Successfully created directory %s " %path)
        else:
            print ("Directory %s already exists" %path)
        np.save(path+"/alp_opt.npy",alp)
        np.save(path+"/chi_opt.npy",self.chi_opt)
        np.save(path+"/nu_opt.npy",self.nu_opt)
        np.save(path+"/Jcv_opt.npy",self.Jcv_opt)
        print("========================================================")
        print("==== SAMPLING OF CONTRACTION METRICS BY CV-STEM END ====")
        print("========================================================\n\n")
        pass
    
    def linesearch(self):
        """
        Perform line search of optimal contraction rate in CV-STEM


        Objects to be updated
        -------
        alp_opt : float
            optimal contraction rate

        """
        alp = self.alims[0]
        da = self.da
        Na = int((self.alims[1]-self.alims[0])/da)+1
        Jcv_prev = np.Inf
        Ncv = self.Nls
        xpmin = np.hstack((self.xlims[0,:],self.plims[0,:]))
        xpmax = np.hstack((self.xlims[1,:],self.plims[1,:]))
        Nxp = self.n+self.n_p
        xps = np.random.uniform(xpmin,xpmax,size=(Ncv,Nxp))
        xs,ps,_ = np.hsplit(xps,np.array([self.n,Nxp]))
        print("========================================================")
        print("============= LINE SEARCH OF OPTIMAL ALPHA =============")
        print("========================================================")
        for k in range(Na):
            self.cvstem0(xs,ps,alp)
            print("Optimal value: Jcv =","{:.2f}".format(self.Jcv),\
                  "( alpha =","{:.3f}".format(alp),")")
            if Jcv_prev <= self.Jcv:
                alp = alp-da
                break
            alp += da
            Jcv_prev = self.Jcv
        self.alp_opt = alp
        print("Optimal contraction rate: alpha =","{:.3f}".format(alp))
        print("========================================================")
        print("=========== LINE SEARCH OF OPTIMAL ALPHA END ===========")
        print("========================================================\n\n")
        pass
    
    def cvstem0(self,xs,ps,alp):
        """
        Run one single instance of CV-STEM algorithm for given states xs and
        contraction rate alpha


        Parameters
        ----------
        xs : ndarray - (Ncv,n), where Ncv is # state samples
            state samples for solving CV-STEM
        ps : ndarray - (Ncv,n_p), where Ncv is # state samples
            system parameter samples for solving CV-STEM
        alp : float
            contraction rate of interest

        Objects to be updated
        -------
        Ws : list of length Ncv
            list containing inverse of n by n optimal contraction metrics
            in current instance of CV-STEM
        chi : numpy.float64
            optimal upper bound of condition number of contraction metrics
            in current instance of CV-STEM
        nu : numpy.float64
            optimal upper bound of induced 2-norm of contraction metrics
            in current instance of CV-STEM
        Jcv : numpy.float64
            optimal steady-state upper bound of estimation or tracking error
            in current instance of CV-STEM
        cvx_status : str
            problem status of CV-STEM, "optimal", "infeasible", "unbounded",
            "infeasible_inaccurate", or "unbounded_inaccurate"

        """
        epsilon = self.epsilon
        Ncv = np.size(xs,0)
        n = self.n
        I = np.identity(n)
        Ws = []
        for k in range(Ncv):
            Ws.append(cp.Variable((n,n),PSD=True))
        nu = cp.Variable(nonneg=True)
        chi = cp.Variable(nonneg=True)
        errtxt = "https://github.com/AstroHiro/ncm#troubleshooting"
        if len(sig(self.Afun).parameters) == 1:
            fun1 = self.Afun
            self.Afun = lambda x,p: fun1(x)
        if (self.iEC == "est") and (len(sig(self.Cfun).parameters) == 1):
            fun2 = self.Cfun
            self.Cfun = lambda x,p: fun2(x)
        if self.iEC == "est":
            Af = self.Afun
            Cf = self.Cfun
            J = (self.d1_over*self.b_over*chi\
                 +self.d2_over*self.c_over*self.g_over*nu)/alp
        elif self.iEC == "con":
            Af = lambda x,p: self.Afun(x,p).T
            Cf = lambda x,p: self.h_or_g(x,p).T
            J = self.d1_over*self.b_over*chi/alp+self.d2_over*nu
        else:
            raise ValueError('Invalid iEC: iEC = "est" or "con"')
        constraints = []
        for k in range(Ncv):
            x = xs[k,:]
            p = ps[k,:]
            Ax = Af(x,p)
            Cx = Cf(x,p)
            W = Ws[k]
            constraints += [chi*I-W >> 0,W-I >> 0]
            constraints += [-2*alp*W-((W-I)/self.dt+W@Ax+Ax.T@W-2*nu*Cx.T@Cx)\
                            >> epsilon*I]
        prob = cp.Problem(cp.Minimize(J),constraints)
        prob.solve(solver=cp.MOSEK)
        cvx_status = prob.status
        if cvx_status in ["infeasible","infeasible_inaccurate"]:
            raise ValueError("Problem infeasible: see "+errtxt)
        elif cvx_status in ["unbounded","unbounded_inaccurate"]:
            raise ValueError("Problem unbounded: "+errtxt)
        Wsout = []
        for k in range(Ncv):
            Wk = Ws[k].value/nu.value
            Wsout.append(Wk)
        self.Ws = Wsout
        self.nu = nu.value
        self.chi = chi.value
        self.Jcv = prob.value
        self.cvx_status = cvx_status
        pass
    
    def M2cholM(self):
        """
        Compute cholesky-decomposed optimal contraction metrics obtained by
        CV-STEM
        
        
        Objects to be updated
        -------
        Ms_opt : list of length Nx, where Nx is # samples to be used for NCM
            list containing ndarray (n,n) optimal contraction metrics
        cholMs : list of length Nx
            list containing ndarray (int(n*(n+1)/2), ) optimal contraction 
            metrics

        """
        Nx = self.Nx
        n = self.n
        Ms_opt = []
        cholMs = []
        for k in range(Nx):
            Mk = np.linalg.inv(self.Ws_opt[k])
            cholMk = np.linalg.cholesky(Mk)
            cholMk = cholMk.T # upper triangular
            cholMk_vec = np.zeros(int(n*(n+1)/2)) 
            for i in range (n):
                j = (n-1)-i;
                di = np.diag(cholMk,j)
                cholMk_vec[int(1/2*i*(i+1)):int(1/2*(i+1)*(i+2))] = di
            Ms_opt.append(Mk)
            cholMs.append(cholMk_vec)
        self.Ms_opt = Ms_opt
        self.cholMs = np.array(cholMs)
        pass
    
    def cholM2M(self,cholM):
        """
        Convert cholesky-decomposed optimal contraction metrics to original
        form in R^(n x n)


        Parameters
        ----------
        cholM : ndarray - (int(n*(n+1)/2), )
            cholesky-decomposed optimal contraction metrics

        Returns
        -------
        M : ndarray - (n,n)
            optimal contraction metrics

        """
        cMnp = 0
        n = self.n
        for i in range(n):
            lb = int(i*(i+1)/2)
            ub = int((i+1)*(i+2)/2)
            Di = cholM[lb:ub]
            Di = np.diag(Di,n-(i+1))
            cMnp += Di
        M = (cMnp.T)@cMnp
        return M  
        
    def jacobian(self,x,p,fun):
        """
        Compute Jacobian of given vector field


        Parameters
        ----------
        x : ndarray - (n, )
            current state x
        fun : function - ndarray (n, ) -> (nout, )
            given vector field

        Returns
        -------
        dfdx : ndarray - (ny,n)
             Jacobian of given vector field

        """
        n = self.n
        y = fun(x,p)
        h = 1e-4
        nout = np.size(y)
        dfdx = np.zeros((nout,n))
        for j in range(n):
            dx1 = np.zeros(n)
            dx2 = np.zeros(n)
            dx1[j] = -h
            dx2[j] = h
            dfdx[:,j] = (fun(x+dx2,p)-fun(x+dx1,p))/(2*h)
        return dfdx
    
    def matrix_2bound(self,fun):
        """
        Compute approximate upper bound of induced 2-norm of given matrix 
        function in given state space


        Parameters
        ----------
        fun : function - ndarray (n, ) -> (n1,n2)
            given matrix function

        Returns
        -------
        mat_over_out : numpy.float64
            upper bound of induced 2-norm of given matrix function in given 
            state space

        """
        xpmin = np.hstack((self.xlims[0,:],self.plims[0,:]))
        xpmax = np.hstack((self.xlims[1,:],self.plims[1,:]))
        Nxp = self.n+self.n_p
        xps = np.random.uniform(xpmin,xpmax,size=(self.Nx,Nxp))
        xs,ps,_ = np.hsplit(xps,np.array([self.n,Nxp]))
        mat_over_out = 0
        for k in range(self.Nx):
            Mat = fun(xs[k,:],ps[k,:])
            mat_over = np.linalg.norm(Mat,ord=2)
            if mat_over > mat_over_out:
                mat_over_out = mat_over
        return mat_over_out
    
    def dynamics(self,x,p,dEf):
        """
        Compute vector field of given nonlinear dynamical system with state 
        feedback input
        

        Parameters
        ----------
        x : ndarray - (n, )
            current state x
        p : ndarray - (n_p, )
            current system parameter
        dEf : function - ndarray (n, ) -> (n, )
            function that returns state feedback input at current state

        Returns
        -------
        fout : ndarray - (n, )
            vector field of given nonliner dynamical system with state
            feedback input

        """
        fout = self.dynamicsf(x,p)+dEf(x,p)
        return fout

    def rk4(self,x,p,dEf,fun):
        """
        Compute state at next time step by 4th order Runge-Kutta method
        

        Parameters
        ----------
        x : ndarray - (n, )
            current state x
        p : ndarray - (n_p, )
            current system parameter
        dEf : function - ndarray (n, ) -> (n, )
            function that returns state feedback input at current state
        fun : function - ndarray (n, ) -> (n, )
            function to be integrated

        Returns
        -------
        x : ndarray - (n, )
            state at next time step

        """
        Nrk = self.Nrk
        dt_rk = self.dt_rk
        for num in range(0,Nrk):
            k1 = fun(x,p,dEf)
            k2 = fun(x+k1*dt_rk/2.,p,dEf)
            k3 = fun(x+k2*dt_rk/2.,p,dEf)
            k4 = fun(x+k3*dt_rk,p,dEf)
            x = x+dt_rk*(k1+2.*k2+2.*k3+k4)/6.
        return x
    
    def unifrand2(self,d_over,nk):
        """
        Generate nk-dimensional random point uniformally distributed in
        L2-ball of radius d_over
        

        Parameters
        ----------
        d_over : float
            radius of L2-ball
        nk : int
            dimension of output vector

        Returns
        -------
        d : ndarray - (nk, )
            nk-dimensional random point uniformally distributed in L2-ball of 
            radius d_over

        """
        d_over_out = d_over+1
        while d_over_out > d_over:
            d = np.random.uniform(-d_over,d_over,size=nk)
            d_over_out = np.linalg.norm(d)
        return d
    
    def clfqp(self,x,p):
        """
        Compute optimal control input solving Control Layapunov Fucntion
        Quadratic Program (CLFQP) using NCM as Lyapunov function
        

        Parameters
        ----------
        x : ndarray - (n, )
            current state x
        p : ndarray - (n_p, )
            current system parameter

        Returns
        -------
        u : ndarray - (m, )
            current input u

        """
        alp = self.alp_opt
        nu = self.nu_opt
        dt = self.dt
        n = self.n
        I = np.identity(n)
        M = self.ncm(x,p)
        nu = np.size(self.h_or_g(x,p),1)
        u = cp.Variable((nu,1))
        e = np.reshape(x,(n,1))
        fx = np.reshape(self.dynamicsf(x,p),(n,1))
        gx = self.h_or_g(x,p)
        dMdt = (nu*I-M)/dt
        constraints = [2*e.T@(fx+gx@u)+e.T@dMdt@e <= -2*alp*e.T@M@e]
        prob = cp.Problem(cp.Minimize(cp.sum_squares(u)),constraints)
        prob.solve()
        u = u.value
        u = np.ravel(u)
        return u
    
    def simulation(self,dt,tf,x0,z0=None,dscale=10.0,xnames="num",Ncol=1,\
                   FigSize=(20,10),FontSize=20,phis=None):
        """
        Perform NCM-based estimation or control of given nolinear dynamical
        systems and return simulation results
        

        Parameters
        ----------
        dt : float
            simulation time step
        tf : float
            terminal time
        x0 : ndarray - (n, )
            initial state
        z0 : ndarray - (n, ), to be used for iEC = "est"
            estimated initial state
        dscale : float, optional, default is 10
            scale of external disturbance 
        xnames : str, optional, default is "num"
            list containing names of each state, when xnames = "num", they are
            denoted as xnames = ["state 1","state 2",...]
        Ncol : int, optional, default is 1
            # columns of state figures to be generated
        FigSize : tuple, optional, default is (20,10)
            size of state figures to be generated
        FontSize : float, optional, default is 20
            font size of figures to be generated
        phis : system parameter history, optional, default is None
            history of system parameters

        Returns
        -------
        this : ndarray - (int(tf/dt)+1, )
            time histry 
        xhis : ndarray - (int(tf/dt)+1,n)
            state history
        zhis : ndarray - (int(tf/dt)+1,n), to be used for estimation tasks
            estimated state history

        """
        """
        
        
        1) SIMULATION
    
        
        """
        print("========================================================")
        print("====================== SIMULATIOM ======================")
        print("========================================================")
        if dt <= self.dt_rk:
            self.dt_rk = dt
        self.Nrk = int(dt/self.dt_rk)
        Nsim = int(tf/dt)
        np.set_printoptions(precision=1)
        print("time step =",dt)
        print("terminal time =",tf)
        print("initial state =",x0)
        if self.iEC == "est":
            print("estimated initial state =",z0)
            funx = lambda x,p,u: self.dynamicsf(x,p)
            funz = self.dynamics
            z = z0
            zhis = np.zeros((Nsim+1,self.n))
            zhis[0,:] = z
            tit1 = "Performance of NCM-based Estimation (1)"
            tit2 = "Performance of NCM-based Estimation (2)"
            ly = r"estimation error: $\|x-\hat{x}\|_2$"
            l1 = r"estimation error"
            bNam1 = "=================== ESTIMATION ERROR ==================="
            bNam2 = "============ ESTIMATION ERROR OF EACH STATE ============"
        elif self.iEC == "con":
            funx = self.dynamics
            zhis = np.zeros((Nsim+1,self.n))
            tit1 = "Performance of NCM-based Control (1)"
            tit2 = "Performance of NCM-based Control (2)"
            ly = r"tracking error: $\|x-x_d\|_2$"
            l1 = r"tracking error"
            bNam1 = "==================== TRACKING ERROR ===================="
            bNam2 = "============= TRACKING ERROR OF EACH STATE ============="
        else:
            raise ValueError('Invalid iEC: iEC = "est" or "con"')
        l2 = r"optimal steady-state upper bound"
        x = x0
        xhis = np.zeros((Nsim+1,self.n))
        xhis[0,:] = x
        this = np.linspace(0,tf,Nsim+1)
        if phis == None:
            phis = np.linspace(self.plims[0,:],self.plims[1,:],Nsim)
        if (self.iEC == "est") and (len(sig(self.Cfun).parameters) == 1):
            fun1 = self.Cfun
            self.Cfun = lambda x,p: fun1(x)
        if (len(sig(self.Bw).parameters) == 1):
            fun2 = self.Bw
            self.Bw = lambda x,p: fun2(x)
        if (self.iEC == "est") and (len(sig(self.Gw).parameters) == 1):
            fun3 = self.Gw
            self.Gw = lambda x,p: fun3(x)
        for k in range(Nsim):
            p = phis[k,:]
            if self.iEC == "est":
                Mx = self.ncm(z,p)
                Cx = self.Cfun(z,p)
                Lx = Mx@Cx.T
                d2 = self.unifrand2(self.d2_over,np.size(self.Gw(x,p),1))*dscale
                y = self.h_or_g(x,p)+self.Gw(x,p)@d2
                dEf = lambda z,p: Lx@(y-self.h_or_g(z,p))
                z = self.rk4(z,p,dEf,funz)
                zhis[k+1,:] = z
            elif self.iEC == "con":
                Mx = self.ncm(x,p)
                Bx = self.h_or_g(x,p)
                Kx = Bx.T@Mx
                u = -Kx@x
                dEf = lambda x,p: self.h_or_g(x,p)@u
            else:
                raise ValueError('Invalid iEC: iEC = "est" or "con"')
            d1 = self.unifrand2(self.d1_over,np.size(self.Bw(x,p),1))*dscale
            x = self.rk4(x,p,dEf,funx)+self.Bw(x,p)@d1*dt
            xhis[k+1,:] = x
        """
        
        
        2) FIGURE GENERATION
    
        
        """
        print("========================================================")
        print(bNam1)
        print("========================================================")
        matplotlib.rcParams.update({"font.size": 15})
        matplotlib.rc("text",usetex=True)
        plt.figure()
        plt.plot(this,np.sqrt(np.sum((xhis-zhis)**2,1)))
        plt.plot(this,np.ones(np.size(this))*self.Jcv_opt)
        plt.xlabel(r"time",fontsize=FontSize)
        plt.ylabel(ly,fontsize=FontSize)
        plt.legend([l1,l2],loc="best")
        plt.title(tit1,fontsize=FontSize)
        plt.show()
        print("========================================================")
        print(bNam2)
        print("========================================================")
        Nrow = int(self.n/Ncol)+np.remainder(self.n,Ncol)
        fig,ax = plt.subplots(Nrow,Ncol,figsize=FigSize)
        plt.subplots_adjust(wspace=0.25,hspace=0.25)
        if Ncol == 1:
            ax = np.reshape(ax,(self.n,1))
        elif Nrow == 1:
            ax = np.reshape(ax,(1,self.n))
        if xnames == "num":
            xnames = []
            for i in range(self.n):
                xnames += [r"state "+str(i+1)]
        for row in range(Nrow):
            for col in range(Ncol):
                i = Ncol*row+col
                if i+1 <= self.n:
                    ax[row,col].plot(this,xhis[:,i]-zhis[:,i])
                    ax[row,col].set_xlabel(r"time",fontsize=FontSize)
                    if self.iEC == "est":
                        LabelName = r"estimation error: "+xnames[i]
                    elif self.iEC == "con":
                        LabelName = r"tracking error: "+xnames[i]
                    else:
                        txterr = 'Invalid iEC: iEC = "est" or "con"'
                        raise ValueError(txterr)
                    ax[row,col].set_ylabel(LabelName,fontsize=FontSize)
        fig.suptitle(tit2,fontsize=FontSize)
        plt.show()
        print("========================================================")
        print("==================== SIMULATIOM END ====================")
        print("========================================================")
        return this,xhis,zhis
    
    def simulation_OFC(self,ncmE,ncmC,f,g,Cfun,h,dt,tf,x0,z0=None,dscale=10.0,\
                xnames="num",Ncol=1,FigSize=(20,10),FontSize=20,phis=None):
        """
        Perform NCM-based output feedback control of given nolinear dynamical
        systems and return simulation results
        

        Parameters
        ----------
        dt : float
            simulation time step
        tf : float
            terminal time
        x0 : ndarray - (n, )
            initial state
        z0 : ndarray - (n, ), to be used for iEC = "est"
            estimated initial state
        dscale : float, optional, default is 10
            scale of external disturbance 
        xnames : str, optional, default is "num"
            list containing names of each state, when xnames = "num", they are
            denoted as xnames = ["state 1","state 2",...]
        Ncol : int, optional, default is 1
            # columns of state figures to be generated
        FigSize : tuple, optional, default is (20,10)
            size of state figures to be generated
        FontSize : float, optional, default is 20
            font size of figures to be generated
        phis : system parameter history, optional, default is None
            history of system parameters

        Returns
        -------
        this : ndarray - (int(tf/dt)+1, )
            time histry 
        xhis : ndarray - (int(tf/dt)+1,n)
            state history
        zhis : ndarray - (int(tf/dt)+1,n), to be used for estimation tasks
            estimated state history

        """
        """
        
        
        1) SIMULATION
    
        
        """
        if len(sig(f).parameters) == 1:
            fun1 = f
            f = lambda x,p: fun1(x)
        if len(sig(g).parameters) == 1:
            fun2 = g
            g = lambda x,p: fun2(x)
        if len(sig(Cfun).parameters) == 1:
            fun3 = Cfun
            Cfun = lambda x,p: fun3(x)
        if len(sig(h).parameters) == 1:
            fun4 = h
            h = lambda x,p: fun4(x)
        print("========================================================")
        print("====================== SIMULATIOM ======================")
        print("========================================================")
        if dt <= self.dt_rk:
            self.dt_rk = dt
        self.Nrk = int(dt/self.dt_rk)
        Nsim = int(tf/dt)
        np.set_printoptions(precision=1)
        print("time step =",dt)
        print("terminal time =",tf)
        print("initial state =",x0)
        print("estimated initial state =",z0)
        funx = lambda x,p,dEf: f(x,p)+dEf(x,p)
        z = z0
        zhis = np.zeros((Nsim+1,self.n))
        zhis[0,:] = z
        x = x0
        xhis = np.zeros((Nsim+1,self.n))
        xhis[0,:] = x
        tit1 = "Performance of NCM-based Output Feedback (1)"
        tit2 = "Performance of NCM-based Output Feedback (2)"
        tit3 = "Performance of NCM-based Output Feedback (3)"
        tit4 = "Performance of NCM-based Output Feedback (4)"
        ly = r"estimation error: $\|x-\hat{x}\|_2$"
        l1 = r"estimation error"
        lyb = r"tracking error: $\|x-x_d\|_2$"
        l1b = r"tracking error"
        bNam1 = "=================== ESTIMATION ERROR ==================="
        bNam2 = "============ ESTIMATION ERROR OF EACH STATE ============"
        bNam3 = "==================== Tracking ERROR ===================="
        bNam4 = "============= Tracking ERROR OF EACH STATE ============="
        l2 = r"optimal steady-state upper bound"
        if phis == None:
            phis = np.linspace(self.plims[0,:],self.plims[1,:],Nsim)
        for k in range(Nsim):
            p = phis[k,:]
            Mc = ncmC.ncm(z,p)
            u = -g(z,p).T@Mc@z
            dEfC = lambda x,p: g(x,p)@u
            d1 = self.unifrand2(ncmC.d1_over,np.size(ncmC.Bw(x,p),1))*dscale
            x = self.rk4(x,p,dEfC,funx)+ncmC.Bw(x,p)@d1*dt
            xhis[k+1,:] = x
            Me = ncmE.ncm(z,p)
            Cx = Cfun(z,p)
            Lx = Me@Cx.T
            #Lx = K.T
            d2 = self.unifrand2(ncmE.d2_over,np.size(ncmE.Gw(x,p),1))*dscale
            y = h(x,u,p)+ncmE.Gw(x,p)@d2
            funz = lambda z,p,dEf: f(z,p)+g(z,p)@u+dEf(z,p)
            dEfE = lambda z,p: Lx@(y-h(z,u,p))
            z = self.rk4(z,p,dEfE,funz)
            zhis[k+1,:] = z
        this = np.linspace(0,tf,Nsim+1)
        """
        
        
        2) FIGURE GENERATION
    
        
        """
        print("========================================================")
        print(bNam1)
        print("========================================================")
        matplotlib.rcParams.update({"font.size": 15})
        matplotlib.rc("text",usetex=True)
        plt.figure()
        plt.plot(this,np.sqrt(np.sum((xhis-zhis)**2,1)))
        plt.plot(this,np.ones(np.size(this))*ncmE.Jcv_opt)
        plt.xlabel(r"time",fontsize=FontSize)
        plt.ylabel(ly,fontsize=FontSize)
        plt.legend([l1,l2],loc="best")
        plt.title(tit1,fontsize=FontSize)
        plt.show()
        print("========================================================")
        print(bNam2)
        print("========================================================")
        Nrow = int(self.n/Ncol)+np.remainder(self.n,Ncol)
        fig,ax = plt.subplots(Nrow,Ncol,figsize=FigSize)
        plt.subplots_adjust(wspace=0.25,hspace=0.25)
        if Ncol == 1:
            ax = np.reshape(ax,(self.n,1))
        elif Nrow == 1:
            ax = np.reshape(ax,(1,self.n))
        if xnames == "num":
            xnames = []
            for i in range(self.n):
                xnames += [r"state "+str(i+1)]
        for row in range(Nrow):
            for col in range(Ncol):
                i = Ncol*row+col
                if i+1 <= self.n:
                    ax[row,col].plot(this,xhis[:,i]-zhis[:,i])
                    ax[row,col].set_xlabel(r"time",fontsize=FontSize)
                    LabelName = r"estimation error: "+xnames[i]
                    ax[row,col].set_ylabel(LabelName,fontsize=FontSize)
        fig.suptitle(tit2,fontsize=FontSize)
        plt.show()
        print("========================================================")
        print(bNam3)
        print("========================================================")
        matplotlib.rcParams.update({"font.size": 15})
        matplotlib.rc("text",usetex=True)
        plt.figure()
        plt.plot(this,np.sqrt(np.sum((xhis)**2,1)))
        plt.plot(this,np.ones(np.size(this))*ncmC.Jcv_opt)
        plt.xlabel(r"time",fontsize=FontSize)
        plt.ylabel(lyb,fontsize=FontSize)
        plt.legend([l1b,l2],loc="best")
        plt.title(tit3,fontsize=FontSize)
        plt.show()
        print("========================================================")
        print(bNam4)
        print("========================================================")
        Nrow = int(self.n/Ncol)+np.remainder(self.n,Ncol)
        fig,ax = plt.subplots(Nrow,Ncol,figsize=FigSize)
        plt.subplots_adjust(wspace=0.25,hspace=0.25)
        if Ncol == 1:
            ax = np.reshape(ax,(self.n,1))
        elif Nrow == 1:
            ax = np.reshape(ax,(1,self.n))
        if xnames == "num":
            xnames = []
            for i in range(self.n):
                xnames += [r"state "+str(i+1)]
        for row in range(Nrow):
            for col in range(Ncol):
                i = Ncol*row+col
                if i+1 <= self.n:
                    ax[row,col].plot(this,xhis[:,i])
                    ax[row,col].set_xlabel(r"time",fontsize=FontSize)
                    LabelName = r"tracking error: "+xnames[i]
                    ax[row,col].set_ylabel(LabelName,fontsize=FontSize)
        fig.suptitle(tit4,fontsize=FontSize)
        plt.show()
        print("========================================================")
        print("==================== SIMULATIOM END ====================")
        print("========================================================")
        return this,xhis,zhis