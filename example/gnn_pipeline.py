"""
Example of pipeline of the BoundML library

It replicates the experiment of "Exact Combinatorial Optimization  with Graph Convolutional Neural Networks"
(http://arxiv.org/abs/1906.01629)
"""
import ecole

import boundml.instances
from boundml.components import StrongBranching, Pseudocosts, EcoleComponent
from boundml.components.gnn_component import GnnBranching
from boundml.evaluation import evaluate_solvers, SolverEvaluationResults
from boundml.ml import train, BranchingDatasetGenerator, load_policy
from boundml.solvers import DefaultScipSolver, ModularSolver

instances = boundml.instances.CombinatorialAuctionGenerator(100, 500)

# Generate a dataset by solving instances using Strong Branching
instances.seed(0)
state_component = EcoleComponent(ecole.observation.NodeBipartite())
generator = BranchingDatasetGenerator(instances, StrongBranching(), state_component, Pseudocosts(), expert_probability=0.1)

folder = "samples"

generator.generate(folder_name=folder, max_instances=20)

model = train(sample_folder=folder, learning_rate=0.05, n_epochs=10, output="agent.pkl")

# Evaluation of the model compared to other strategies
instances.seed(12)
n_instances = 10

scip_params = {
    "limits/time": 10,
}

solvers = [
    DefaultScipSolver("relpscost", scip_params), # A solver that use a build in strategies
    DefaultScipSolver("pscost", scip_params), # A solver that use a build in strategies
    ModularSolver(
        GnnBranching(
            "agent.pkl",
            feature_component=EcoleComponent(ecole.observation.NodeBipartite()),
            try_use_gpu = False
        ),
        scip_params=scip_params,
    )
]

metrics = ["nnodes", "time", "gap"] # metrics of interest

evaluation_results = evaluate_solvers(solvers, instances, n_instances, metrics)

report = evaluation_results.compute_report(
        SolverEvaluationResults.sg_metric("nnodes", 10),
        SolverEvaluationResults.sg_metric("time", 1),
        SolverEvaluationResults.nwins("time"),
        SolverEvaluationResults.nsolved(),
        SolverEvaluationResults.auc_score("time"))

print(report)



evaluation_results.performance_profile(metric="time")
evaluation_results.performance_profile(metric="nnodes")
