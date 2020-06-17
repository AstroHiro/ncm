# [Neural Contraction Metrics for Robust Estimation and Control: A Convex Optimization Approach](https://arxiv.org/abs/2006.04361)
This repository contains "classncm.py" class file and several jupyter notebook examples for designing a Neural Cnotraction Metric (NCM) intruduced in [this paper](https://arxiv.org/abs/2006.04361). Here are some important facts about the NCM.
* The NCM is a global neural network-based approximation of an optimal contraction metric.
* The existence of a contraction metric is a necessary and sufficient condition for exponential stability of nonlinear systems. 
* The optimality of NCMs stems from the fact that the contraction metrics sampled offline are the solutions of a convex optimization problem (see [CV-STEM](https://arxiv.org/abs/2006.04359) for more details) to minimize an upper bound of the steady-state Euclidean distance between perturbed and unperturbed system trajectories.
## Files
* classncm.py: class file that contains functions required for constructing an NCM for a given nonlinear dynamical systems
* NCMestimation: jupyter notebook that illustrates how to use classncm.py for nonlinear optimal state estimation
* NCMcontrol: jupyter notebook that illustrates how to use classncm.py for nonlinear optimal feedbck control
