from boundml.evaluation import Evaluator
from boundml.instances import MipLibInstances
from boundml.solvers import DefaultScipSolver

scip_params = {"limits/time": 60}

# Download automatically MIPLIB instances and cache them
instances = MipLibInstances("benchmark")

solvers = [
    DefaultScipSolver("relpscost", scip_params),
    DefaultScipSolver("pscost", scip_params),
]

evaluator = Evaluator(["nnodes", "time", "gap"])

# Solve the ten first instances
evaluator.evaluate(solvers, instances, 10) # No executor is given, so run the solvers sequentially