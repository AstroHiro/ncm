# [Neural Contraction Metrics for Robust Estimation and Control: A Convex Optimization Approach](https://arxiv.org/abs/2006.04361)
This repository contains "classncm.py" class file and several jupyter notebook examples for designing a Neural Contraction Metric (NCM) intruduced in [this paper](https://arxiv.org/abs/2006.04361). Here are some important facts about the NCM.
1. The NCM is a global neural network-based approximation of an optimal contraction metric.
2. The existence of a contraction metric is a necessary and sufficient condition for exponential stability of nonlinear systems. 
3. The optimality of NCMs stems from the fact that the contraction metrics sampled offline are the solutions of a convex optimization problem (see [CV-STEM](https://arxiv.org/abs/2006.04359) for more details) to minimize an upper bound of the steady-state Euclidean distance between perturbed and unperturbed system trajectories.
4. Due to the differential formulation of contraction analysis, there exists a nonlinear version of estimation and control duality, analogous to the one of the Kalman filter and LQR duality in linear systems (see the [NCM paper](https://arxiv.org/abs/2006.04361) for more).
## Files
* classncm.py: class file that contains functions required for constructing an NCM for a given nonlinear dynamical system
* NCMestimation.ipynb: jupyter notebook that illustrates how to use classncm.py for nonlinear optimal state estimation
* NCMcontrol.ipynb: jupyter notebook that illustrates how to use classncm.py for nonlinear optimal feedbck control
## Quick guide to classncm.py
The detailed explanation on its methods and objects are given in [this page](https://github.com/AstroHiro/ncm/wiki/NCM-Documentation). Here we introduce some useful mothods for the NCM design. The trade-off between larger feedback gains and (larger nu) and smaller steady-state tracking error (smaller chi) discussed in the NCM paper can be handled by changing the values d1_over and d2_over in the NCM class.
* [ncm](https://github.com/AstroHiro/ncm/wiki/NCM-methods:-ncm)\
Returns a trained NCM with respect to a given state.
* train(iTrain=1,Nbatch=32,Nlayers=3,Nunits=100,Nepochs=10000,ValidationSplit=0.1,Patience=20)\
Trains a neural network to be used for designing an NCM and returns a Keras neural network model.
* [cvstem](https://github.com/AstroHiro/ncm/wiki/NCM-methods:-cvstem)\
Samples optimal contraction metrics of a given dynamical system in a given state space by the [CV-STEM](https://arxiv.org/abs/2006.04359) method. These metrics will be used for the neural network training in the train method.
* [linesearch](https://github.com/AstroHiro/ncm/wiki/NCM-methods:-linesearch)\
Finds the optimal contraction rate by line search.
* [simulation](https://github.com/AstroHiro/ncm/wiki/NCM-methods:-simulation)\
Performs NCM-based estimation or control of a given nolinear dynamical systems and returns simulation results.
## Troubleshooting
The convex optimization problem in the [cvstem](https://github.com/AstroHiro/ncm/wiki/NCM-methods:-cvstem) method could become infeasible for some nonlinear dynamical systems under certain choises of parameters. Here are some tips in avoiding such infeasibility in practice.
* Start from a smaller value of alpha (contraction rate)\
Since the stability constraint becomes tighter as alpha becomes larger, set alims[0] (minimum alpha) smaller.
* Select smaller state space\
Since larger state space could lead to stricter stability constraints, set Xlims to represent the smaller state space.
* Change d1_over and d2_over from their default values\
Although this will not affect the infeasibility theoretically, it may solve some numerical issues.
* Solve the CV-STEM problems along the pre-computed trajectories.\
The NCM class computes dM/dt using a lower bound of the induced 2-norm of a contraction metric. You could run the CV-STEM along given trajectories as was done in the [NCM paper](https://arxiv.org/abs/2006.04361) seeking for a less tight stability condition. The details are given in the source codes of this paper.
