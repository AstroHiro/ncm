#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 02:39:46 2020

@author: hiroyasu
"""


import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import SCPmulti as scp

matplotlib.rcParams.update({'font.size': 15})
matplotlib.rc('font',**{'family':'serif','serif':['Times']})
matplotlib.rc('text', usetex=True)
LabelSize = 17
DPI = 300

###################### OPTIMAL VALUES ######################
alp_list = np.load('data/params/alpha_macro/alp_list.npy')
optvals = np.load('data/params/alpha_macro/optvals.npy')

###################### MACRO ALPHA LINE SEARCH ######################
plt.figure()
plt.plot(np.log10(alp_list),optvals)
plt.xlabel(r'$\alpha$',fontsize=LabelSize)
plt.ylabel(r'$\frac{\bar{d}_1\bar{b}\chi+\bar{d}_2\bar{c}\bar{d}\nu}{\alpha}$',fontsize=LabelSize+6)
plt.xticks(np.log10(alp_list),('0.05','0.1','1','3','6'))
#plt.ylabel(r'$\overline{d}_1\overline{b}\sqrt{\frac{\overline{\xi}}{\underline{\xi}}}+\frac{\overline{d}_2\overline{c}\overline{d}}{{\underline{\xi}}}$')
plt.savefig('data/plots/SC_macro.png',dpi=DPI)
plt.show()

###################### MICRO ALPHA LINE SEARCH ######################
alp_list = np.load('data/params/alpha_micro/alp_list.npy')
optvals = np.load('data/params/alpha_micro/optvals.npy')
alp = np.load('data/params/optimal/alp.npy')

plt.figure()
plt.plot(alp*np.ones(1000),np.linspace(np.log10(1),np.log10(20),1000))
plt.scatter(alp_list,np.log10(optvals),s=2,c='b')
plt.xlabel(r'$\alpha$',fontsize=LabelSize)
plt.ylabel(r'$\frac{\bar{d}_1\bar{b}\chi+\bar{d}_2\bar{c}\bar{d}\nu}{\alpha}$',fontsize=LabelSize+6)
plt.legend([r'Optimal $\alpha = 3.497$'],loc='best')
plt.savefig('data/plots/SC_micro.png',dpi=DPI)
plt.show()

###################### RNN TRAINING HISTORY ######################
plt.figure()
e_list = np.load('data/trained_nets/e_list.npy')
t_loss_list = np.load('data/trained_nets/t_loss_list.npy')
loss_list  = np.load('data/trained_nets/loss_list.npy')
plt.plot(e_list,loss_list)
plt.plot (e_list,t_loss_list)
#plt.plot(e_list,np.log10(loss_list))
#plt.plot(e_list,np.log10(t_loss_list))
plt.xlabel('Epochs',fontsize=LabelSize)
plt.ylabel('Loss',fontsize=LabelSize)
plt.legend(['Test','Train'],loc='best')
plt.savefig('data/plots/rnn_errors_sc.png',bbox_inches='tight',dpi=DPI)
plt.show()

###################### MOTION PLANNING ######################
DT = scp.DT
Robs = scp.Robs
XOBSs = scp.XOBSs
    
this = np.load('data/simulation/this.npy')
X1his = np.load('data/simulation/X1his.npy')
U1his = np.load('data/simulation/U1his.npy')
X2his = np.load('data/simulation/X2his.npy')
U2his = np.load('data/simulation/U2his.npy')
X3his = np.load('data/simulation/X3his.npy')
U3his = np.load('data/simulation/U3his.npy')
U1hisN = np.load('data/simulation/U1hisN.npy')
U2hisN = np.load('data/simulation/U2hisN.npy')
U3hisN = np.load('data/simulation/U3hisN.npy')
U1hisND = np.load('data/simulation/U1hisND.npy')
U2hisND = np.load('data/simulation/U2hisND.npy')
U3hisND = np.load('data/simulation/U3hisND.npy')
xxTube1 = np.load('data/simulation/xxTube1.npy')
xxTube2 = np.load('data/simulation/xxTube2.npy')
XXdRCT = np.load('data/params/desiredRCT/XXdRCT.npy')
UUdRCT = np.load('data/params/desiredRCT/UUdRCT.npy')

Nplot = xxTube1.shape[0]
lw = 2.5
z0 = 1
z1 = 10
plt.figure()
l1, = plt.plot(X1his[:,0],X1his[:,1],linewidth=lw,zorder=z0)
l2, = plt.plot(X2his[:,0],X2his[:,1],':',linewidth=lw,zorder=z0)
l3, = plt.plot(X3his[:,0],X3his[:,1],'-.',linewidth=lw,zorder=z0)
la, = plt.plot(XXdRCT[:,0],XXdRCT[:,1],'--k',linewidth=lw-1,alpha=0.4)
lb = plt.fill_between(xxTube1[:,0],xxTube1[:,1],np.zeros(Nplot),facecolor='black',alpha=0.2)
plt.fill_between(xxTube2[:,0],xxTube2[:,1],np.zeros(Nplot),facecolor='white')
plt.fill_between(np.linspace(19.3,22,100),16.9*np.ones(100),np.zeros(100),facecolor='white')
plt.fill_between(np.linspace(-2,20,100),0.093*np.ones(100),np.zeros(100),facecolor='white')
plt.fill_between(np.linspace(-1.03228,1.033,100),-0.2*np.ones(100),0.093*np.ones(100),facecolor='black',alpha=0.2)
for i in range(XOBSs.shape[0]):
    x,y=[],[]
    for _x in np.linspace(0,2*np.pi):
        x.append(Robs*np.cos(_x)+XOBSs[i,0])
        y.append(Robs*np.sin(_x)+XOBSs[i,1])
        plt.plot(x,y,'k')
plt.scatter([0,20],[0,18],marker='x',s=100,color='k',zorder=z1) # zorder to bring it forward
plt.annotate('start',(-1.5,-2),fontsize=LabelSize) 
plt.annotate('goal',(21,17.5),fontsize=LabelSize)
plt.axes().set_aspect('equal')
plt.xlabel(r'horizontal coordinate $p_x$',fontsize=LabelSize)
plt.ylabel(r'vertical coordinate $p_y$',fontsize=LabelSize)
legend1 = plt.legend([l1,l2,l3],[r'NCM',r'CV-STEM',r'LQR'],loc='right',bbox_to_anchor=(1.01,0.6),fontsize=12) 
plt.legend([la,lb],[r'Desired trajectory',r'Bounded error tube'],loc='upper left',fontsize=12) 
plt.gca().add_artist(legend1) 
plt.xlim([-4,29.3])
plt.ylim([-2.5,20])
#plt.legend(['NCM','CV-STEM','LQR','Desired','Tube','Tube','Tube'],loc='upper right',bbox_to_anchor=(1.45,1))
plt.grid()
plt.savefig('data/plots/SC_motion_planning.png',bbox_inches='tight',dpi=DPI)
plt.show()
    
plt.figure()
plt.plot(this[0:500],np.cumsum(U1hisN**2)*DT)
plt.plot(this[0:500],np.cumsum(U2hisN**2)*DT)
plt.plot(this[0:500],np.cumsum(U3hisN**2)*DT)
plt.show()
Uef1 = np.cumsum(U1hisN**2)*DT
Uef2 = np.cumsum(U2hisN**2)*DT
Uef3 = np.cumsum(U3hisN**2)*DT
Uef1 = Uef1[-1]
Uef2 = Uef2[-1]
Uef3 = Uef3[-1]
###################### TEST PLOTS ######################
x = np.array([1,2])
data = np.array([10,8])
err = np.array([2,1])
b1 = plt.bar(x-.2,2*err,0.4,color='b',bottom=data-err,alpha=0.3)
plt.legend([b1[0]], ['nice legend graphic','nice legend graphic'],shadow=True,fancybox=True,numpoints=1)
plt.legend([b1[0]], ['nice legend graphic','nice legend graphic'],shadow=True,fancybox=True,numpoints=1)
plt.axis([0,3,0,15])
plt.grid()
plt.show()