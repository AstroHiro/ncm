# [Neural Contraction Metrics for Robust Estimation and Control: A Convex Optimization Approach](https://arxiv.org/abs/2006.04361)
This repository contains "classncm.py" class file and several jupyter notebook examples for designing a Neural Contraction Metric (NCM) intruduced in [this paper](https://arxiv.org/abs/2006.04361). Here are some important facts about the NCM.
* The NCM is a global neural network-based approximation of an optimal contraction metric.
* The existence of a contraction metric is a necessary and sufficient condition for exponential stability of nonlinear systems. 
* The optimality of NCMs stems from the fact that the contraction metrics sampled offline are the solutions of a convex optimization problem (see [CV-STEM](https://arxiv.org/abs/2006.04359) for more details) to minimize an upper bound of the steady-state Euclidean distance between perturbed and unperturbed system trajectories.
* Due to the differential formulation of contraction analysis, there exists a nonlinear version of estimation and control duality, analogous to the one of the Kalman filter and LQR duality in linear systems (see the [NCM paper](https://arxiv.org/abs/2006.04361) for more).
## Files
* classncm.py: class file that contains functions required for constructing an NCM for a given nonlinear dynamical system
* NCMestimation.ipynb: jupyter notebook that illustrates how to use classncm.py for nonlinear optimal state estimation
* NCMcontrol.ipynb: jupyter notebook that illustrates how to use classncm.py for nonlinear optimal feedbck control
## Quick guide to classncm.py
The detailed explanation on its methods and objects are given in [this page](https://github.com/AstroHiro/ncm/wiki/NCM-Documentation). Here we introduce some useful mothods for the NCM design.
* [ncm(x)](https://github.com/AstroHiro/ncm/wiki/NCM-methods:-ncm)\
Returns a trained NCM with respect to a given state.
* train(iTrain=1,Nbatch=32,Nlayers=3,Nunits=100,Nepochs=10000,ValidationSplit=0.1,Patience=20)\
Trains a neural network to be used for designing an NCM and returns a Keras neural network model.
* [cvstem()](https://github.com/AstroHiro/ncm/wiki/NCM-methods:-cvstem)\
Samples optimal contraction metrics of a given dynamical system in a given state space by the [CV-STEM](https://arxiv.org/abs/2006.04359) method. These metrics will be used for the neural network training in the train method.
* [linesearch()](https://github.com/AstroHiro/ncm/wiki/NCM-methods:-linesearch)\
Finds the optimal contraction rate by line search.
* [simulation(dt,tf,x0,z0=None,dscale=10.0,xnames="num",Ncol=1,FigSize=(20,10),FontSize=20)](https://github.com/AstroHiro/ncm/wiki/NCM-methods:-simulation)\
Performs NCM-based estimation or control of a given nolinear dynamical systems and returns simulation results.
