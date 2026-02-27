import concurrent

import pyscipopt
from pyscipopt import Model

from boundml.components import ScoringBranchingStrategy
from boundml.evaluation import SolverEvaluationResults, Evaluator
from boundml.solvers import DefaultScipSolver, ModularSolver
from boundml.instances import CombinatorialAuctionGenerator


class MyBranchingStrategy(ScoringBranchingStrategy):
    def compute_scores(self, model: Model):
        """
        This method is called before each branching decision.
        It computes a score for each branching candidate
        """
        # Initialize scores with the good size, and a score of - infinity for each candidate
        scores = super().compute_scores(model)

        # List of branching candidates
        candidates, *_ = model.getLPBranchCands()


        # Compute the score for each candidate
        var: pyscipopt.Variable
        for i, var in enumerate(candidates):
            obj_coef = var.getObj()
            val = var.getLPSol()

            scores[i] = obj_coef * val

        return scores

    def __str__(self):
        return "Custom"


if __name__ == "__main__":
    scip_params = {
        "limits/time": 30
    }

    # List of solvers to evaluate
    solvers= [
        DefaultScipSolver("relpscost", scip_params=scip_params),
        DefaultScipSolver("pscost", scip_params=scip_params),
        ModularSolver(MyBranchingStrategy(), scip_params=scip_params),
    ]

    # Generator of instances on which to perform the evaluation
    instances = CombinatorialAuctionGenerator(100, 500)
    instances.seed(0)

    # Evaluate the solvers
    evaluator = Evaluator(["nnodes", "time", "gap"])

    # Use multiprocessing to run several solvers in parallel
    with concurrent.futures.ProcessPoolExecutor(max_workers=10) as executor:
        # data is a SolverEvaluationResults. It can be pickled to be saved and analyzed latter
        data = evaluator.evaluate(
            solvers,
            instances,
            10, # number of instances to solve
            seeds=[0, 1, 3], # Each configuration is run once with each seed
            executor=executor,
        )

    # Compute a report from the raw data.
    # The report aggregates different metrics for each solver.
    report = data.compute_report(
        SolverEvaluationResults.sg_metric("nnodes", 10), # SG mean of the number of nodes
        SolverEvaluationResults.sg_metric("time", 1), # SG mean of the time spent
        SolverEvaluationResults.sg_metric("time", 1, std=True), # SG mean of the std overall instances w.r.t time
        SolverEvaluationResults.sg_metric("gap", 1), # SG mean of the gap of instances where at least one solver has not reached optimallity
        SolverEvaluationResults.nwins("time"), # Number of time a solver has been the fastest
        SolverEvaluationResults.nwins("nnodes"), # Number of time a solver has been the fastest
        SolverEvaluationResults.nsolved(), # Number of time a solver solved an instance to optimality
        SolverEvaluationResults.auc_score("time"), # AUC score with respect to time
    )
    print()
    # Display the report
    # It is possible to get a latex table from it: report.to_latex()
    print(report)


