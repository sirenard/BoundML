import pyscipopt
from pyscipopt import Model

from boundml.components import ScoringBranchingStrategy
from boundml.evaluation import evaluate_solvers, SolverEvaluationResults
from boundml.solvers import DefaultScipSolver, ModularSolver
from boundml.instances import CombinatorialAuctionGenerator


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

    # Evaluate the solvers
    # data is a SolverEvaluationResults. It can be pickled to be saved and analyzed latter
    data = evaluate_solvers(
        solvers,
        instances,
        10, # number of instances to solve
        ["nnodes", "time", "gap"], # list of metrics of interes among ["nnodes", "time", "gap"]
        0, # Number of cores to use in parallel. If 0, use all the available cores
    )

    # Compute a report from the raw data.
    # The report aggregates different metrics for each solver.
    report = data.compute_report(
        SolverEvaluationResults.sg_metric("nnodes", 10), # SG mean of the number of nodes
        SolverEvaluationResults.sg_metric("time", 1), # SG mean of the time spent
        SolverEvaluationResults.nwins("nnodes"), # Number of time a solver has been the fastest
        SolverEvaluationResults.nsolved(), # Number of time a solver solved an instance to optimality
        SolverEvaluationResults.auc_score("time"), # AUC score with respect to time
    )

    # Display the report
    # It is possible to get a latex table from it: report.to_latex()
    print(report)


