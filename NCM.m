%{
MIT License

Copyright (c) 2020 Hiroyasu Tsukamoto https://hirotsukamoto.com/

Permission is hereby granted, free of charge, to any person obtaining a
copy of this software and associated documentation files (the "Software"),
to deal in the Software without restriction, including without limitation 
the rights to use, copy, modify, merge, publish, distribute, sublicense, 
and/or sell copies of the Software, and to permit persons to whom the 
Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in 
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING 
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER 
DEALINGS IN THE SOFTWARE.
%}
% This class file is under development.
classdef NCM
    properties
        dt 
        f
        g
        Xlims
        alims
        IdxT
        fname
        d_ov = 1
        b2_ov = 1
        lam = 1
        da = 0.01
        Nx = 1000
        Nls = 1
        nx
    end
    methods
        function obj = NCM(dt,f,g,Xlims,alims,IdxT,fname)
            obj.dt = dt;
            obj.f = f;
            obj.g = g;
            obj.Xlims = Xlims;
            obj.alims = alims;
            obj.IdxT = IdxT;
            obj.fname = fname;
            obj.nx = size(Xlims,1);
        end
        function [Ms_opt,Ws_opt,chi_opt,nu_opt,Jcvc_opt] = CVSTEM(obj)
            alp = obj.LineSearch();
            N = obj.Nx;
            n = obj.nx;
            Nsplit = 1;
            Np = floor(N/Nsplit);
            Nr = rem(N,Nsplit);
            Xs_opt = zeros(n,N);
            for k = 1:N
                Xs_opt(:,k) = unifrnd(obj.Xlims(:,1),obj.Xlims(:,2));
            end
            chi_opt = 0;
            nu_opt = 0;
            Ms_opt = zeros(n,n,N);
            Ws_opt = zeros(n,n,N);
            disp('=============================================================')
            disp('=========== SAMPLE CONTRACTION METRICS BY CV-STEM ===========')
            disp('=============================================================')
            for p = 1:Np
                if rem(p,floor(1000/Nsplit)) == 0
                    disp(["# sampled metrics: ",num2str(p*Nsplit),"..."])
                end
                Xs_p = Xs_opt(:,Nsplit*(p-1)+1:Nsplit*p);
                [Ms,Wsout,chi,nu,Jcvc,cvx_status] = obj.CVSTEM0(Xs_p,alp);
                Ms_opt(:,:,Nsplit*(p-1)+1:Nsplit*p) = Ms;
                Ws_opt(:,:,Nsplit*(p-1)+1:Nsplit*p) = Wsout;
                if nu >= nu_opt
                    nu_opt = nu;
                end
                if chi >= chi_opt
                    chi_opt = chi;
                end
            end
            if Nr ~= 0
                disp(["# samples metrics: ",num2str(N),"..."])
                Xs_p = Xs_opt(:,Nsplit*Np:N);
                [Ms,Wsout,chi,nu,Jcvc,cvx_status] = obj.CVSTEM0(Xs_p,alp);
                Ms_opt(:,:,Nsplit*(p-1)+1:Nsplit*p) = Ms;
                Ws_opt(:,:,Nsplit*Np:N) = Wsout;
                if nu >= nu_opt
                    nu_opt = nu;
                end
                if chi >= chi_opt
                    chi_opt = chi;
                end
            end
            Jcvc_opt = obj.b2_ov*obj.d_ov*chi_opt/alp+obj.lam*nu_opt;
            disp(["Contraction rate: alpha = ",num2str(alp),...
                 ", Optimal value: Jcvc = ",num2str(Jcvc_opt)])
            disp('=============================================================')
            disp('========= SAMPLE CONTRACTION METRICS BY CV-STEM END =========')
            disp('=============================================================\n\n')
        end
        function alp = LineSearch(obj)
            n = obj.nx;
            alp = obj.alims(1);
            dals = obj.da;
            Na = floor((obj.alims(2)-obj.alims(1))/dals)+1;
            Jcvc_prev = Inf;
            Ncv = obj.Nls;
            Xs = zeros(n,Ncv);
            for k = 1:Ncv
                Xs(:,k) = unifrnd(obj.Xlims(:,1),obj.Xlims(:,2));
            end
            disp('=============================================================')
            disp('=================== LINE SEARCH FOR ALPHA ===================')
            disp('=============================================================')
            for k = 1:Na
                [Ms,Wsout,chi,nu,Jcvc,cvx_status] = obj.CVSTEM0(Xs,alp);
                if Jcvc_prev <= Jcvc
                    alp = alp-dals;
                    break
                end
                disp(["Contraction rate: alpha = ",num2str(alp),...
                  ", Optimal value: Jcvc = ",num2str(Jcvc)])
                alp = alp+dals;
                Jcvc_prev = Jcvc;
            end
            disp("Optimal contraction rate: alpha = ",num2str(alp))
            disp('=============================================================')
            disp('================= LINE SEARCH FOR ALPHA END =================')
            disp('=============================================================\n\n')
        end
        function [Ms,Wsout,chi,nu,Jcvc,cvx_status] = CVSTEM0(obj,Xs,alp)
            Ncv = size(Xs,2);
            n = obj.nx;
            I = eye(n);
            cvx_begin sdp quiet
                variable Ws(n,n,Ncv) semidefinite
                variable nu nonnegative
                variable chi nonnegative
                minimize (obj.b2_ov*obj.d_ov*chi/alp+obj.lam*nu)
                subject to
                    for k = 1:Ncv
                        X = Xs(:,k);
                        Ax = obj.Afun(X);
                        Bx = obj.g(X);
                        W = Ws(:,:,k);
                        I <= W <= chi*I;
                        -(I-W)/obj.dt+Ax*W+W*Ax'-2*nu*Bx*Bx' <= -2*alp*W;
                    end
            cvx_end
            Wsout = zeros(n,n,Ncv);
            Ms = zeros(n,n,Ncv);
            for k = 1:Ncv
                Wsout(:,:,k) = Ws(:,:,k)/nu;
                Ms(:,:,k) = inv(Ws(:,:,k));
            end
            Jcvc = cvx_optval;
        end
        function dfdX = Afun(obj,X)
            n = obj.nx;
            h = 1e-4;
            dfdX = zeros(n,n);
            for j = 1:n
                dX1 = zeros(4,1);
                dX2 = zeros(4,1);
                dX1(j) = -h;
                dX2(j) = h;
                dfdX(:,j) = (obj.f(X+dX2)-obj.f(X+dX1))/2/h;
            end
        end
    end
end
