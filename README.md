# [Neural Contraction Metric](https://arxiv.org/abs/2006.04361) Python Package 
### Real-time robust nolinear estimation and control: a convex optimization approach
This repository contains the [NCM class](https://github.com/AstroHiro/ncm/wiki/Documentation) file and [Jupyter Notebook](https://jupyter.readthedocs.io/en/latest/install.html) examples for [estimation](https://github.com/AstroHiro/ncm/blob/master/NCMestimation.ipynb) and [control](https://github.com/AstroHiro/ncm/blob/master/NCMcontrol.ipynb) of nonlinear dynamical systems using a Neural Contraction Metric (NCM) intruduced in [this paper](https://arxiv.org/abs/2006.04361). Here are some important facts about the NCM.
1. The NCM is a global neural network-based approximation of an optimal contraction metric.
2. The existence of a contraction metric is a necessary and sufficient condition for exponential stability of nonlinear systems. 
3. The optimality of NCMs stems from the fact that the contraction metrics sampled offline are the solutions of a convex optimization problem (see [CV-STEM](https://arxiv.org/abs/2006.04359) for more details) to minimize an upper bound of the steady-state Euclidean distance between perturbed and unperturbed system trajectories.
4. Due to the differential formulation of contraction analysis, there exists a nonlinear version of estimation and control duality, analogous to the one of the Kalman filter and LQR duality in linear systems (see the [NCM paper](https://arxiv.org/abs/2006.04361) for more).
5. Note that the NCM class file assumes input affine autonomous nonlinear systems, but you can always convert input non-affine non-autonomous systems into a input affine autonomous from by treating t as a state variable and du/dt as a control input.
## [License](https://github.com/AstroHiro/ncm/blob/master/LICENSE.txt)
MIT License

Copyright (c) 2020 [Hiroyasu Tsukamoto](https://hirotsukamoto.com/)

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
## Files
* [NCMestimation.ipynb](https://github.com/AstroHiro/ncm/blob/master/NCMestimation.ipynb) : jupyter notebook that illustrates how to use classncm.py for nonlinear optimal state estimation
* [NCMcontrol.ipynb](https://github.com/AstroHiro/ncm/blob/master/NCMcontrol.ipynb) : jupyter notebook that illustrates how to use classncm.py for nonlinear optimal feedbck control
* [classncm.py](https://github.com/AstroHiro/ncm/blob/master/classncm.py) : class file that contains functions required for constructing an NCM for a given nonlinear dynamical system
## Quick guide to NCMestimation.ipynb and NCMcontrol.ipynb
### Required software
In addition to standard python packages like numpy, you need several other packages and software. The NCM class file "classncm.py" has been verified to work with CVXPY 1.1.1, Mosek 9.2.11, TensorFlow 2.2.0, and Keras 2.3.1.
* [Jupyter Notebook](https://jupyter.readthedocs.io/en/latest/install.html) to see and edit codes
* [CVXPY](https://www.cvxpy.org/install/index.html) and [Mosek](https://docs.mosek.com/9.2/install/installation.html) to solve convex optimization problems
* [Keras](https://keras.io/about/) and [TensorFlow](https://www.tensorflow.org/install) to construct neural networks (Keras comes packaged with TensorFlow 2.0 as tensorflow.keras)
### List of things you need at least
* nonlinear dynamical system
* nonlinear measurement equation for state estimation or actionation matrix for feedback control
* state space of interest
* your guess of contraction rate (just put a small number if not sure)
## Quick guide to classncm.py
The detailed explanation on its methods and objects are given in [this page](https://github.com/AstroHiro/ncm/wiki/Documentation). Here we introduce some useful mothods for the NCM design. The trade-off between larger feedback gains (larger nu) and smaller steady-state tracking error (smaller chi) discussed in the NCM paper can be handled by changing the values d1_over and d2_over in the NCM class.
* **[ncm](https://github.com/AstroHiro/ncm/wiki/NCM-methods:-ncm)** : Returns a trained NCM with respect to a given state.
* **[train](https://github.com/AstroHiro/ncm/wiki/NCM-methods:-train)** : Trains a neural network to be used for designing an NCM and returns a Keras neural network model.
* **[cvstem](https://github.com/AstroHiro/ncm/wiki/NCM-methods:-cvstem)** : Samples optimal contraction metrics of a given dynamical system in a given state space by the [CV-STEM](https://arxiv.org/abs/2006.04359) method. These metrics will be used for the neural network training in the train method.
* **[linesearch](https://github.com/AstroHiro/ncm/wiki/NCM-methods:-linesearch)** : Finds the optimal contraction rate by line search.
* **[simulation](https://github.com/AstroHiro/ncm/wiki/NCM-methods:-simulation)** : Performs NCM-based estimation or control of a given nolinear dynamical systems and returns simulation results.
## Troubleshooting
The convex optimization problem in the [cvstem](https://github.com/AstroHiro/ncm/wiki/NCM-methods:-cvstem) method could become infeasible for some nonlinear dynamical systems under certain choises of parameters. Here are some tips in avoiding such infeasibility in practice.
* **Start from smaller value of alpha** : Since the stability constraint becomes tighter as alpha (contraction rate) becomes larger, set alims[0] (minimum alpha) smaller.
* **Select smaller state space** : Since larger state space could lead to stricter stability constraints, set Xlims to represent the smaller state space.
* **Use larger CV-STEM sampling period** : It is better to use a smaller CV-STEM sampling period dt when computing dW/dt (W: normalized inverse of a contraction metric) from a theoretical point of view, but it could cause some numerical issues. You could relax dt to some larger values or solve CV-STEM along pre-computed system trajectories.
* **Solve CV-STEM problems along pre-computed trajectories** : The NCM class computes dW/dt (W: normalized inverse of a contraction metric) using a lower bound of the induced 2-norm of a contraction metric. You could run the CV-STEM along given trajectories as was done in the [NCM paper](https://arxiv.org/abs/2006.04361) seeking for a less tight stability condition. The details are given in the source codes of this paper.
* **Change d1_over and d2_over from their default values** : Although this will not affect the infeasibility theoretically, it may solve some numerical issues.
* **Set epsilon to some positive value** : There is a parameter called epsilon in the NCM class, and if you select epsilon > 0 then the stability condition can be relaxed. You could also use epsilon as a decision variable and minimize it to have tightest relaxation but stability in a sense described in the [NCM paper](https://arxiv.org/abs/2006.04361) is no longer guaranteed.
* **Try different solvers** : The default solver for the NCM class is [Mosek](https://docs.mosek.com/9.2/install/installation.html). You can change "solver=cp.MOSEK" in the class file to [something different such as cp.CVXOPT](https://www.cvxpy.org/tutorial/advanced/index.html).
* **Try [CVX](http://cvxr.com/cvx/download/) if you have [MATLAB](https://www.mathworks.com/products/matlab.html)** : The [CVX](http://cvxr.com/cvx/download/) can sometimes obtain optimal solutions even when the solvers in [CVXPY](https://www.cvxpy.org/install/index.html) fail. We have a [matlab version of the NCM class](code) although it is under development.
* If all of these approaches do not work or you have any questions, please [contact us](https://hirotsukamoto.com/).
