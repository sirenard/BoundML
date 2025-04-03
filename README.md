# BoundML

BoundML is a wrapper around a [fork of ecole](https://github.com/sirenard/ecole). 
It allows to easily develop new machine learning based branching strategies based for the Branch and Bound.

## Installation

`pip install boundml`

### Troubleshooting

### Libraries not found

You need to have the following librarires installed
- SCIP 9.1.0
- fmt
- range-v3
- pybind11

It is possible to install it in a conda environment:
```
conda install scip==9.1.0 fmt pybind11

export CMAKE_PREFIX_PATH="${CONDA_PREFIX}"      
export CPLUS_INCLUDE_PATH="${CONDA_PREFIX}/include/"
export LIBRARY_PATH=${CONDA_PREFIX}/lib
export LD_LIBRARY_PATH=${CONDA_PREFIX}/lib
```

The exports commands allow the compiler to find SCIP.

To install range-v3 for a ubuntu ditribution:
```
sudo apt install librange-v3-dev librange-v3-doc
```

## Example

The file [gnn_pipeline](example/gnn_pipeline.py) shows how tu use this library to reproduce easily the work of
[Gasse et al.](http://arxiv.org/abs/1906.01629). It consists of training a GCNN to learn to imitate strong branching.
