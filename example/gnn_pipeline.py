"""
Example of pipeline of the BoundML library

It replicates the experiment of "Exact Combinatorial Optimization  with Graph Convolutional Neural Networks"
(http://arxiv.org/abs/1906.01629)
"""
import ecole
import torch

from boundml import DatasetGenerator, evaluate_solvers, SolverEvaluationResults
from boundml.observers import StrongBranching, PseudoCost, GnnObserver
from boundml.model import train
from boundml.solvers import ClassicSolver, EcoleSolver

instances = ecole.instance.CombinatorialAuctionGenerator(100, 500)

# Generate a dataset by solving instances using Strong Branching
instances.seed(0)
generator = DatasetGenerator(instances, StrongBranching(), PseudoCost(), expert_probability=0.1,
                             state_observer=ecole.observation.NodeBipartite())



# generator.generate(folder_name="samples", max_instances=10)

# model = train(sample_folder="samples", learning_rate=0.05, n_epochs=5, output="agent.pkl")
model = torch.load("agent.pkl")


# Evaluation of the model compared to other strategies
instances.seed(12)
n_instances = 10

scip_params = {
    "limits/time": 10,
}

solvers = [
    ClassicSolver("relpscost", scip_params), # A solver that use a build in strategies
    ClassicSolver("pscost", scip_params), # A solver that use a build in strategies
    EcoleSolver(
        score_observer=GnnObserver(
            "agent.pkl",
            feature_observer=ecole.observation.NodeBipartite(),
            try_use_gpu = True
        ),
        scip_params=scip_params,
    )
]

metrics = ["nnodes", "time", "gap"] # metrics of interest

evaluation_results = evaluate_solvers(solvers, instances, n_instances, metrics)

report = evaluation_results.compute_report(
        SolverEvaluationResults.sg_metric("nnodes", 10),
        SolverEvaluationResults.sg_metric("time", 1),
        SolverEvaluationResults.nwins("nnodes"),
        SolverEvaluationResults.nsolved(),
        SolverEvaluationResults.auc_score("time"))

print(report)



evaluation_results.performance_profile(metric="time")
evaluation_results.performance_profile(metric="nnodes")
