# What is boundml ?

BoundML is a toolbox to evaluate and compare branch and bound solvers.
It allows to easily develop new machine learning based (or not) branching strategies for the Branch and Bound.

It comes with several submodules:

- **solvers**: Contains all the class of type `Solver`. A `Solver` represent a MILP solver.
- **evaluation**: Contains all the tools to evaluate and compare a list of `Solver` on a set of instances.
- **components**: Contains all the definition of `Component` class that can be used to parametrize a `ModularSolver`. A
  `Component` can be simple observer called at a certain time in the solving process, or an active component of the
  solver depending on its subtype.
- **ml**: Contains some ML tools to apply some wellknown techniques to learn branching strategies. In particular, it
  allows to collect a dataset and train a Graph Neural Network to imitate a branching strategy as in
  this [paper](http://arxiv.org/abs/1906.01629).
- **instances**: Contains a set of MILP instances generator

# Installation

There is 2 way to install `boundml`: `pip install boundml` and `pip install boundml[all]`.
The former does not require any specific setup but does not contain the classes that depend on a fork of the library
ecole: [ecole-fork](https::github.com/sirenard/ecole).
The latter contains everything but requires some additional installation steps in order to have ecole-fork and pyscipopt
to share the same SCIP installation.

## Full Installation

`pip install boundml`

You also need to install fmt in order to be able to use `ecole-fork` which is a dependency.
It can be installed via conda: `conda install fmt`.

To test if everything works, the 2 following imports should work:

```
import boundml
import ecole
```

If `import ecole` failed, you can still use all the tools from `boundml` except for the classes that are wrapper around
`ecole`.

# How to ...

## ... Create my own branching strategy ?

In boundml, a branching strategy is simply a `Component` (in particular a `BranchingComponent`) that before each
decision assigns a score to the candidates
variables. Then, the associated `Solver` will branch on the candidate with the highest score.

For this example, we will implement a branching strategy where a candidate's score is its impact on the objective
function: its objective's coefficient multiplied by its current LP solution.

The full example is available [here](example/branching_strategy.py).

```python
import pyscipopt
from pyscipopt import Model

from boundml.components import ScoringBranchingStrategy


class MyBranchingStrategy(ScoringBranchingStrategy):
    def compute_scores(self, model: Model):
        """
        This method is called before each branching decision.
        It computes a score for each branching candidate
        """

        # List of branching candidates
        candidates, *_ = model.getLPBranchCands()

        # Compute the score for each candidate
        var: pyscipopt.Variable
        for i, var in enumerate(candidates):
            obj_coef = var.getObj()
            val = var.getLPSol()

            self.scores[i] = obj_coef * val

    def __str__(self):
        return "Custom"
```

## ... Test my branching strategy

Once a branching strategy is built as an `Observer`, it is easy to compare it with different solvers' configuration.
For this example, we will compare 3 solvers. One that uses the default SCIP branching strategy relpscost, a second one
that uses pscost as branching strategy, and the last one that use our custom strategy defined above.

Boundml provides easy tools to run the solvers on the same instances. In addition, it is easy to summarize the raw data
by computing a report containing the metrics of interest (shifted geometric mean of a metric, number of wins, ...).

The full example is available [here](example/branching_strategy.py).

```python
from boundml.evaluation import evaluate_solvers, SolverEvaluationResults
from boundml.solvers import DefaultScipSolver, ModularSolver
from boundml.instances import CombinatorialAuctionGenerator

scip_params = {
    "limits/time": 30
}

# List of solvers to evaluate
solvers = [
    DefaultScipSolver("relpscost", scip_params=scip_params),
    DefaultScipSolver("pscost", scip_params=scip_params),
    ModularSolver(MyBranchingStrategy(), scip_params=scip_params),
]

# Generator of instances on which to perform the evaluation
instances = CombinatorialAuctionGenerator(100, 500)

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

# Limitations and future development

Here is a list of features that are not yet part of `boundml` but will be one day:

- Implement `Solver` classes for different solvers (gurobi, highs, cplex)
- For the moment, only  `BranchingComponent` are really useful and called during the solving process. The idea is to
  also have specific `Component` for the different steps of the SCIP solver (node selection, heuristics, ...)
- Remove all dependencies to `ecole-fork`

Feel free to contribute by creating pool requests with new features/fixes or by creating issues with your requirement
that could improve `boundml`.

# Troubleshooting and issue

If you encounter any unwanted behaviors or issues with boundml, do not hesitate to create an
issue [here](https://github.com/sirenard/BoundML/issues/new). 

