# TetrisCNN_for_spin_systems
**This is a repository containing code for the NeurIPS ML4PS workshop paper:**  
**"Speak so a physicist can understand you!
TetrisCNN for detecting phase transitions
and order parameters"**
by K. Cybiński, J. Enouen, A. Georges, and A. Dawid

---

The data we use are snaphots of various physical 1D systems in the ground state, generated using the [Julia programming language](https://julialang.org).   
The ground state is calculated using Density Matrix Renormalization Group (DMRG) algorithm, using the [ITensors.jl](https://docs.juliahub.com/ITensors/) library.   
The datasets are saved to the [TFIM_datasets](./TFIM_datasets/).
   
The physical system implementedin this demonstration repository is the **1-D Transverse Field Ising (TFIM) model**    
Hamiltonian (in Planck units, so $c = G = \hbar = k_B = 1$) reads:
```math
\hat{H}_{\rm TFIM} = -J \left( \sum_{i} \hat{S}^z_i \,\hat{S}^z_{i + 1} + g \sum_i \hat{S}^x_i \right)
```

---

For the machine learning part we use [PyTorch](https://pytorch.org) ML library.   
All auxiliary and helper functions are in [/src/](./src) folder. In particular:
* TetrisCNN architecture is located [/src/architectures.py](./src/architectures.py),   
* data loaders are in [/src/loaders.py](./src/loaders.py),  
* functions for unified performance metrics calculations are in [/src/metrics.py](./src/metrics.py),   
* functions for combinatorial generation of all possible n-site correlations within a given kernel are in [/src/combinatorics.py](./src/combinatorics.py) file,
* general auxiliary functions are in [/src/auxiliary_functions.py](./src/auxiliary_functions.py).   

File [1D_TFIM_train.py](./1D_TFIM_train.py) is developed for NN training and saving it into [Models](./Models/) folder. 
This file also plots the training history analogous to Fig. 2 from submission text, along with loss and $R^2$ history.

The analysis pipeline is exectuted with [analysis_pipeline_1D.py](./analysis_pipeline_1D.py) file. This file also plots the detailed report from linear and symbolic regression fittings, along with a report in a format matching Fig. 6 from the submission.

Both the training file ([1D_TFIM_train.py](./1D_TFIM_train.py)) and analysis pipeline file ([analysis_pipeline_1D.py](./analysis_pipeline_1D.py)) allow for a very detailed control, all governed by their CLI arguments, so do not hesitate to call them with `-h` flag, in order to see all customization options.   

The fittings are all done using Symbolic Regression - we use [PySR](https://github.com/MilesCranmer/PySR) Python package for this, which is a wrapper around [SymbolicRegression.jl](https://github.com/MilesCranmer/SymbolicRegression.jl) Julia backend.

The code was written by Kacper Cybiński (University of Warsaw) and James Enouen (University of South California) with help of Anna Dawid (Leiden University).

---

The packages needed for proper execution are listed below. Some of them are as of writing this documentation only pip-installable, and therefore indicated separately below.

### Conda-Installable Packages

- torch
- numba
- sympy
- numpy
- matplotlib
- scikit-learn (sklearn)
- termcolor
- tqdm
- pandas
- networkx
- scipy
- pysr

### Only Pip-Installable Packages

- prettytable
- latex2sympy2
- trimesh
- humanize
