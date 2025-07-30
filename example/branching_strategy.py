import ecole
import numpy as np
import pyscipopt

from boundml.evaluation import evaluate_solvers, SolverEvaluationResults
from boundml.observers import Observer
from boundml.solvers import EcoleSolver, ClassicSolver


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
            probindex = var.getCol().getLPPos() # probindex of the variable as defined in SCIP

            obj_coef = var.getObj()
            val = var.getLPSol()

            scores[probindex] = obj_coef * val

        return scores


if __name__ == "__main__":
    scip_params = {
        "limits/time": 30
    }

    # List of solvers to evaluate
    solvers= [
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


