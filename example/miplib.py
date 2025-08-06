from boundml.evaluation import evaluate_solvers
from boundml.instances import MipLibInstances
from boundml.solvers import DefaultScipSolver

scip_params = {"limits/time": 60}

# Download automatically MIPLIB instances and cache them
instances = MipLibInstances("benchmark")

solvers = [
    DefaultScipSolver("relpscost", scip_params),
    DefaultScipSolver("pscost", scip_params),
]

# Solve the ten first instances
evaluate_solvers(solvers, instances, 10, ["nnodes", "time", "gap"])