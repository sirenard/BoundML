# BoundML

BoundML is a wrapper around a [fork of ecole](https://github.com/sirenard/ecole).
It allows to easily develop new machine learning based branching strategies based for the Branch and Bound.

## Installation

First, it is recommended to install `pyscipopt` and the different dependencies with conda to have the same scip installation
for `pyscipopt` and `ecole-fork` (on which is based `boundml`).

```
conda install pip pyscipopt fmt
pip install boundml
```

## Example

The file [gnn_pipeline](example/gnn_pipeline.py) shows how tu use this library to reproduce easily the work of
[Gasse et al.](http://arxiv.org/abs/1906.01629). It consists of training a GCNN to learn to imitate strong branching.
