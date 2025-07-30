# What is boundml ?

BoundML is a wrapper around a [fork of ecole](https://github.com/sirenard/ecole).
It allows to easily develop new machine learning based (or not) branching strategies for the Branch and Bound.

It comes with several submodules:

- **solvers**: Contains all the class of type `Solver`. They are basically wrapper around SCIP. They can be used to
  solve
  instances using custom settings or branching strategies
- **evaluation**: Contains all the tools evaluate and compare a list of `Solver` on a set of instances.
- **observers**: Contains all the builtin `Observer` class. A `Solver` calls all its observers at different moment of
  the solving process. An `Observer` can be used to define a branching strategy or simply to collects data.
- **ml**: Contains some ML tools to apply some wellknown techniques to learn branching strategies. In particular, it
  allows to collect a dataset and train a Graph Neural Network to imitate a branching strategy as in
  this [paper](http://arxiv.org/abs/1906.01629).

# Installation

First, it is recommended to install `pyscipopt` with conda to have the same scip installation for both `pyscipopt` and
`ecole-fork` (a dependency of boundml).
for `pyscipopt` and `ecole-fork` (on which is based `boundml`). `pybind11` and `fmt` can also be installed with conda if
not already installed on the system.

```bash
conda create -n myenv
conda activate myenv
conda install pip pyscipopt fmt pybind11 --channel conda-forge
```

Then, some environment variables must be set before installing boundml. By doing so, your system will see the SCIP
installation done with conda.

```bash
export CMAKE_PREFIX_PATH="${CONDA_PREFIX}"      
export CPLUS_INCLUDE_PATH="${CONDA_PREFIX}/include/"
export LIBRARY_PATH=${CONDA_PREFIX}/lib
export LD_LIBRARY_PATH=${CONDA_PREFIX}/lib
```

Finally, you can install boundml.

```bash
pip install boundml
```

# How to ...

## ... Create my own branching strategy ?

In boundml, a branching strategy is simply a `Observer` that before each decision assigns a score to the candidates
variables. Then, the associated `Solver` will branch on the candidate with the highest score.

For this example, we will implement a branching strategy where a candidate's score is its impact on the objective
function: its objective's coefficient multiplied by its current LP solution.

The full example is available [here](example/branching_strategy.py).

```python
import ecole
import numpy as np
import pyscipopt

from boundml.observers import Observer


class MyObserver(Observer):
    def extract(self, model: ecole.scip.Model, done: bool):
        """
        This method is called before each branching decision.
        It computes a score for each branching candidate
        """

        # Get the corresponding pyscipopt model
        m: pyscipopt.Model = model.as_pyscipopt()

        # List of branching candidates
        candidates, *_ = m.getLPBranchCands()

        # Array of scores. Each index correspond to a variable probindex
        # For all the variables that are not branching candidates, the score must be np.nan
        scores = np.zeros(m.getNVars())
        scores[:] = np.nan

        # Compute the score for each candidate
        var: pyscipopt.Variable
        for var in candidates:
            probindex = var.getCol().getLPPos()  # probindex of the variable as defined in SCIP

            obj_coef = var.getObj()
            val = var.getLPSol()

            scores[probindex] = obj_coef * val

        return scores
```

## ... Test my branching strategy

Once a branching strategy is built as an `Observer`, it is easy to compare it with different solvers' configuration.
For this example, we will compare 3 solvers. One that uses the default SCIP branching strategy relpscost, a second one
that uses pscost as branching strategy, and the last one that use our custom strategy defined above.

Boundml provides easy tools to run the solvers on the same instances. In addition, it is easy to summarize the raw data
by computing a report containing the metrics of interest (shifted geometric mean of a metric, number of wins, ...).

The full example is available [here](example/branching_strategy.py).

```python
import ecole

from boundml.evaluation import evaluate_solvers, SolverEvaluationResults
from boundml.solvers import EcoleSolver, ClassicSolver

scip_params = {
    "limits/time": 30
}

# List of solvers to evaluate
solvers = [
    ClassicSolver("relpscost", scip_params),
    ClassicSolver("pscost", scip_params),
    EcoleSolver(MyObserver(), scip_params),
]

# Generator of instances on which to perform the evaluation
instances = ecole.instance.CombinatorialAuctionGenerator(100, 500)

# Evaluate the solvers
# data is a SolverEvaluationResults. It can be pickled to be saved and analyzed latter
data = evaluate_solvers(
    solvers,
    instances,
    10,  # number of instances to solve
    ["nnodes", "time", "gap"],  # list of metrics of interes among ["nnodes", "time", "gap"]
    0,  # Number of cores to use in parallel. If 0, use all the available cores
)

# Compute a report from the raw data.
# The report aggregates different metrics for each solver.
report = data.compute_report(
    SolverEvaluationResults.sg_metric("nnodes", 10),  # SG mean of the number of nodes
    SolverEvaluationResults.sg_metric("time", 1),  # SG mean of the time spent
    SolverEvaluationResults.nwins("nnodes"),  # Number of time a solver has been the fastest
    SolverEvaluationResults.nsolved(),  # Number of time a solver solved an instance to optimality
    SolverEvaluationResults.auc_score("time"),  # AUC score with respect to time
)

# Display the report
# It is possible to get a latex table from it: report.to_latex()
print(report)
```

## ... Learn a branching strategy ?

boundml provides basic tools to train a Graph Convolutional Neural Network (GCNN) model to imitate any branching
strategy designed as `Observer`.

The file [gnn_pipeline](example/gnn_pipeline.py) shows how tu use this library to reproduce easily the work of
[Gasse et al.](http://arxiv.org/abs/1906.01629). It consists of training a GCNN to learn to imitate strong branching.

## ... Learn a branching strategy with my own model ?

For the moment, boundml only provides tools to train a GCNN to imitate a branching strategy. However, it is possible to
design your own workflow to train a model, and use this model in an `Observer` to use it as a branching strategy.

`DatasetGenerator` can be useful to generate a dataset.

# Troubleshooting and issue

If you encounter any unwanted behaviors or issues with boundml, do not hesitate to create an
issue [here](https://github.com/sirenard/BoundML/issues/new) 

