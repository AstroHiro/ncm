#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  8 14:01:45 2020

@author: hiroyasu
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import pickle

matplotlib.rcParams.update({'font.size': 15})
matplotlib.rc('font',**{'family':'serif','serif':['Times']})
matplotlib.rc('text', usetex=True)
LabelSize = 20
DPI = 300

###################### FUNCTIONS ######################
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


###################### OPTIMAL VALUES ######################
optvals = np.load('data/params/optimal/optvals.npy')
chi_s = np.load('data/params/optimal/chi_s.npy')
nu_s = np.load('data/params/optimal/nu_s.npy')

OPTVAL = np.mean(optvals)
CHI = np.mean(chi_s)
NU = np.mean(nu_s)


###################### MACRO ALPHA LINE SEARCH ######################
alp_list = np.load('data/params/alpha_macro/alp_list.npy')
optvals = np.load('data/params/alpha_macro/optvals.npy')
status_str = np.load('data/params/alpha_macro/status_str.npy')
plt.figure()
numX0 = optvals.shape[0]
for iX0 in range(numX0):
    plt.plot(np.log10(alp_list),optvals[iX0,:])
plt.xlabel(r'$\alpha$',fontsize=LabelSize)
plt.ylabel(r'$\frac{J_{e}^*}{\alpha} = \frac{\overline{d}_1\overline{b}\chi+\overline{d}_2\overline{c}\overline{d}\nu}{\alpha}$',fontsize=LabelSize+6)
plt.grid()
plt.xticks(np.log10(alp_list),('0.05','0.1','1','3','6'))
#plt.ylabel(r'$\overline{d}_1\overline{b}\sqrt{\frac{\overline{\xi}}{\underline{\xi}}}+\frac{\overline{d}_2\overline{c}\overline{d}}{{\underline{\xi}}}$')
plt.savefig('data/plots/lorenz_macro.png',bbox_inches='tight',dpi=DPI)
plt.show()


###################### MICRO ALPHA LINE SEARCH ######################
alp_list = np.load('data/params/alpha_micro/alp_list.npy')
optvals = np.load('data/params/alpha_micro/optvals.npy')
status_str = np.load('data/params/alpha_micro/status_str.npy')
chi_s = np.load('data/params/alpha_micro/chi_s.npy')
nu_s = np.load('data/params/alpha_micro/nu_s.npy')
alp = np.load('data/params/alpha_micro/alp.npy')
X0s = np.load('data/params/alpha_micro/X0s.npy')

plt.figure()
plt.plot(alp*np.ones(1000),np.linspace(np.log10(25),np.log10(500),1000))
numX0 = optvals.shape[0]
for iX0 in range(numX0):
    plt.scatter(alp_list,np.log10(optvals[iX0,:]),s=2)
plt.xlabel(r'contraction rate $\alpha$',fontsize=LabelSize)
#plt.ylabel(r'$\frac{J_{e}^*}{\alpha} = \frac{\overline{d}_1\overline{b}\chi+\overline{d}_2\overline{c}\overline{d}\nu}{\alpha}$',fontsize=LabelSize+6)
plt.ylabel(r'optimal upper bound $J_{CVe}^*$',fontsize=LabelSize)
plt.grid()
plt.legend([r'Optimal $\alpha = 3.4970$'],loc='best',fontsize=13)
plt.yticks(np.log10(np.array([30,50,100,200,300,400])),(r'$30$',r'$50$',r'$100$',r'$200$',r'$300$',r'$400$'))
plt.savefig('data/plots/lorenz_micro.png',bbox_inches='tight',dpi=DPI)
plt.show()


###################### RNN MODEL SELECTION ######################
nplus = 15
nticks = 30
n_hidden_list = [8,16,32,64,128]
n_layers_list = [1,2,4,8]
for n_hid in range(5):
    n_hidden = n_hidden_list[n_hid]
    fig = plt.figure()
    for n_lay in range(4):
        n_layers = n_layers_list[n_lay]
        fname1 = 'data/trained_nets/errors/e_list_'+str(n_hidden)+'hid_'+str(n_layers)+'lay'+'.npy'
        fname2 = 'data/trained_nets/errors/t_loss_list_'+str(n_hidden)+'hid_'+str(n_layers)+'lay'+'.npy'
        fname3 = 'data/trained_nets/errors/loss_list_'+str(n_hidden)+'hid_'+str(n_layers)+'lay'+'.npy'
        e_list = np.load(fname1)
        t_loss_list = np.load(fname2)
        loss_list = np.load(fname3)
        plt.plot(e_list,loss_list)
    plt.xlabel('epochs',fontsize=LabelSize+nplus)
    #plt.ylabel('Test loss',fontsize=LabelSize+nplus)
    plt.xticks(fontsize=nticks)
    plt.yticks(fontsize=nticks)
    plt.grid()
    plt.legend(['1 layers','2 layers','4 layers','8 layers'],loc='best',fontsize=22)
    fname = 'data/plots/rnn_errors_'+str(n_hidden)+'hid'+'.png'
    plt.savefig(fname,bbox_inches='tight',dpi=DPI)
    plt.show()
    
for n_lay in range(4):
    n_layers = n_layers_list[n_lay]
    fig = plt.figure()
    for n_hid in range(5):
        n_hidden = n_hidden_list[n_hid]
        fname1 = 'data/trained_nets/errors/e_list_'+str(n_hidden)+'hid_'+str(n_layers)+'lay'+'.npy'
        fname2 = 'data/trained_nets/errors/t_loss_list_'+str(n_hidden)+'hid_'+str(n_layers)+'lay'+'.npy'
        fname3 = 'data/trained_nets/errors/loss_list_'+str(n_hidden)+'hid_'+str(n_layers)+'lay'+'.npy'
        e_list = np.load(fname1)
        t_loss_list = np.load(fname2)
        loss_list = np.load(fname3)
        plt.plot(e_list,loss_list)
    plt.xlabel('epochs',fontsize=LabelSize+nplus)
    plt.ylabel('test loss',fontsize=LabelSize+nplus)
    plt.xticks(fontsize=nticks)
    plt.yticks(fontsize=nticks)
    plt.ylim(0,0.004)
    plt.grid()
    plt.legend(['8 units','16 units','32 units','64 units','128 units'],loc='best',fontsize=22)
    fname = 'data/plots/rnn_errors_'+str(n_layers)+'lay'+'.png'
    plt.savefig(fname,bbox_inches='tight',dpi=DPI)
    plt.show()

nticks_cdc = 30
fig,(ax1,ax2) = plt.subplots(ncols=2,figsize=(14,4))
fig.subplots_adjust(wspace=0.3)
for n_hid in range(5):
    n_layers = 2
    n_hidden = n_hidden_list[n_hid]
    fname1 = 'data/trained_nets/errors/e_list_'+str(n_hidden)+'hid_'+str(n_layers)+'lay'+'.npy'
    fname2 = 'data/trained_nets/errors/t_loss_list_'+str(n_hidden)+'hid_'+str(n_layers)+'lay'+'.npy'
    fname3 = 'data/trained_nets/errors/loss_list_'+str(n_hidden)+'hid_'+str(n_layers)+'lay'+'.npy'
    e_list = np.load(fname1)
    t_loss_list = np.load(fname2)
    loss_list = np.load(fname3)
    ax1.plot(e_list,loss_list)
ax1.set_xlabel(r'epochs',fontsize=LabelSize+nplus)
ax1.set_ylabel(r'test loss',fontsize=LabelSize+nplus)
ax1.set_xticks([0,250,500,750,1000])
ax1.set_yticks([0.001,0.002,0.003,0.004,0.005])
ax1.tick_params(labelsize=nticks_cdc)
ax1.tick_params(labelsize=nticks_cdc)
ax1.set_ylim(0.001,0.005+0.001/4)
ax1.grid()
ax1.legend([r'$8$ units',r'$16$ units',r'$32$ units',r'$64$ units',r'$128$ units'],loc='upper right',bbox_to_anchor=(1,1.038),fontsize=22)

for n_lay in range(4):
    n_hidden = 32
    n_layers = n_layers_list[n_lay]
    fname1 = 'data/trained_nets/errors/e_list_'+str(n_hidden)+'hid_'+str(n_layers)+'lay'+'.npy'
    fname2 = 'data/trained_nets/errors/t_loss_list_'+str(n_hidden)+'hid_'+str(n_layers)+'lay'+'.npy'
    fname3 = 'data/trained_nets/errors/loss_list_'+str(n_hidden)+'hid_'+str(n_layers)+'lay'+'.npy'
    e_list = np.load(fname1)
    t_loss_list = np.load(fname2)
    loss_list = np.load(fname3)
    ax2.plot(e_list,loss_list)
ax2.set_xlabel(r'epochs',fontsize=LabelSize+nplus)
#ax2.ylabel(r'Test loss',fontsize=LabelSize+nplus)
ax2.set_xticks([0,250,500,750,1000])
ax2.set_yticks([0.000,0.003,0.006,0.009,0.012])
ax2.tick_params(labelsize=nticks_cdc)
ax2.tick_params(labelsize=nticks_cdc)
ax2.set_ylim(0,0.012+0.003/4)
ax2.grid()
ax2.legend([r'$1$ layers',r'$2$ layers',r'$4$ layers',r'$8$ layers'],loc='upper right',bbox_to_anchor=(1,1.038),fontsize=22)
fname = 'data/plots/rnn_errors_for_cdc.png'
fig.savefig(fname,bbox_inches='tight',dpi=DPI)
fig.show()

###################### RNN TRAINING HISTORY ######################
plt.figure()
e_list = np.load('data/trained_nets/e_list.npy')
t_loss_list = np.load('data/trained_nets/t_loss_list.npy')
loss_list  = np.load('data/trained_nets/loss_list.npy')
plt.plot(e_list,loss_list)
plt.plot(e_list,t_loss_list)
#plt.plot(e_list,np.log10(loss_list))
#plt.plot(e_list,np.log10(t_loss_list))
plt.xlabel('epochs',fontsize=LabelSize+nplus)
plt.ylabel('MSE loss',fontsize=LabelSize+nplus)
plt.grid()
plt.xticks(fontsize=nticks)
plt.yticks(fontsize=nticks)
plt.legend(['Test','Train'],loc='best',fontsize=nticks)
plt.savefig('data/plots/rnn_errors.png',bbox_inches='tight',dpi=DPI)
plt.show()


###################### RNN MODEL VALIDATION ######################
numX0 = 10
WindowSize = 100
dcMMs = LoadDict('data/trained_nets/dcMMs.pkl')
fig = plt.figure()   
dcMMmeans = [] 
for iX0 in range(numX0):
    dcMM = dcMMs[iX0]
    dcMMmeans.append(np.mean(dcMM))
    plt.plot(np.linspace(0,50,500),dcMM)
plt.xlabel('t',fontsize=LabelSize)
plt.ylabel('Normalized MSE error',fontsize=LabelSize)
plt.savefig('data/plots/rnn_validation.png',bbox_inches='tight',dpi=DPI)
plt.show()

fig = plt.figure()    
for iX0 in range(numX0):
    dcMM = dcMMs[iX0]
    datadM = {'score': dcMM}
    dfdM = pd.DataFrame(datadM)
    dfdMhis = dfdM.rolling(window=WindowSize).mean()
        
    plt.plot(np.linspace(0,50,500),dfdMhis)
plt.xlabel(r'time $t$',fontsize=LabelSize+nplus)
#plt.ylabel('Normalized MSE error',fontsize=LabelSize+nplus)
plt.grid()
plt.xticks(fontsize=nticks)
plt.yticks(fontsize=nticks)
plt.savefig('data/plots/rnn_validation_filtered.png',bbox_inches='tight',dpi=DPI)
plt.show()


###################### LORENTZ OSCILLATOR SIMULATION ######################
optval = np.load('data/simulation/optval.npy')
this = np.load('data/simulation/this.npy')
Xhis = np.load('data/simulation/Xhis.npy')
Z1his = np.load('data/simulation/Z1his.npy')
Z2his = np.load('data/simulation/Z2his.npy')
Z3his = np.load('data/simulation/Z3his.npy')
e1his = np.load('data/simulation/e1his.npy')
e2his = np.load('data/simulation/e2his.npy')
e3his = np.load('data/simulation/e3his.npy')

data1 = {'score': e1his}
data2 = {'score': e2his}
data3 = {'score': e3his}
# Create dataframe
df1 = pd.DataFrame(data1)
df2 = pd.DataFrame(data2)
df3 = pd.DataFrame(data3)
WindowSize = 15
dfe1his = df1.rolling(window=WindowSize).mean()
dfe2his = df2.rolling(window=WindowSize).mean()
dfe3his = df3.rolling(window=WindowSize).mean()
lw = 2.5
fig = plt.figure()
l1, = plt.plot(this,dfe1his,linewidth=lw)
l2, = plt.plot(this,dfe2his,':',linewidth=lw) 
l3, = plt.plot(this,dfe3his,'-.',linewidth=lw)
la, = plt.plot(this,optval*np.ones(this.size),'--k',linewidth=lw) 
plt.xlabel(r'time $t$',fontsize=LabelSize)
plt.ylim([9.4,48.5])
plt.grid()
plt.ylabel(r'estimation error $||x-\hat{x}||$',fontsize=LabelSize)
legend1 = plt.legend([l1,l2,l3],[r'NCM',r'CV-STEM',r'EKF'],loc='right',bbox_to_anchor=(1,0.705),fontsize=12) 
plt.legend([la],[r'Mean of upper bounds'],loc='upper right',fontsize=13,bbox_to_anchor=(1,1.01)) 
plt.gca().add_artist(legend1) 
#plt.legend([r'NCM',r'CV-STEM',r'EKF',r'Optimal Steady-state upper bound'],loc='upper center',bbox_to_anchor=(0.5,1.5)) 
plt.savefig('data/plots/lorenz.png',bbox_inches='tight',dpi=DPI)
plt.show()