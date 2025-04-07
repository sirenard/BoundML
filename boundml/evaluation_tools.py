import pickle
import tempfile
import multiprocessing

import numpy as np

from boundml.output_control import OutputControl
from boundml.solver_evaluation_results import SolverEvaluationResults
from boundml.solvers import Solver

from boundml.utils import shifted_geometric_mean

def _solve(solver, prob_file_name, metrics):
    solver.solve(prob_file_name)
    return [solver[metric] for metric in metrics]


def evaluate_solvers(solvers: [Solver], instances, n_instances, metrics, n_cpu=0):
    if n_cpu == 0:
        n_cpu = multiprocessing.cpu_count()


    output_control = OutputControl()

    data = np.zeros((n_instances, len(solvers), len(metrics)))

    files = []
    async_results = {}

    # Start the jobs
    with multiprocessing.Pool(processes=n_cpu) as pool:
        for i, instance in zip(range(n_instances), instances):
            for j, solver in enumerate(solvers):
                prob_file = tempfile.NamedTemporaryFile(suffix=".lp")
                output_control.mute()
                instance.as_pyscipopt().writeProblem(prob_file.name)
                output_control.unmute()

                files.append(prob_file)

                async_results[i,j] = pool.apply_async(_solve, [solver, prob_file.name, metrics])

        print(f"{'Instance':<15}" + "".join([f"{str(solver):<{15 * len(metrics)}}" for solver in solvers]))
        print(f"{'':<15}" + len(solvers) * ("".join([f"{metric:<15}" for metric in metrics])))
        for i, instance in zip(range(n_instances), instances):
            print(f"{i:<15}", end="")
            for j, solver in enumerate(solvers):
                line = async_results[i,j].get()
                for k, d in enumerate(line):
                    data[i, j, k] = d
                print("".join([f"{d:{'<15.3f' if type(d) == float else '<15'}}" for d in line]), end="", flush=True)

            print()

    for f in files:
        f.close()

    res = SolverEvaluationResults(data, [str(s) for s in solvers], metrics)

    print("=" * (15 * (len(solvers) * len(metrics) + 1)))

    ss = {
        "nnodes": 10,
        "time": 1,
        "gap": 1,
    }

    means = {}
    for k, metric in enumerate(metrics):
        mean = res.aggregate(metrics[k], lambda values: shifted_geometric_mean(values, shift=ss[metric]))
        means[metrics[k]] = mean

    info = []
    for j in range(len(solvers)):
        for metric in metrics:
            info.append(means[metric][j])
    print(f"{'sg mean': <15}" + "".join([f"{val: <15.3f}" for val in info]))

    prob_file.close()
    return res


if __name__ == "__main__":
    data: SolverEvaluationResults = pickle.load(open("../data", "rb"))

    r = data.compute_report(
        SolverEvaluationResults.sg_metric("nnodes", 10),
        SolverEvaluationResults.sg_metric("time", 1),
        SolverEvaluationResults.nwins("nnodes"),
        SolverEvaluationResults.nsolved(),
        SolverEvaluationResults.auc_score("time"),
        header = "easy"
    )

    r2 = data.compute_report(
        SolverEvaluationResults.sg_metric("nnodes", 10),
        SolverEvaluationResults.sg_metric("time", 1),
        SolverEvaluationResults.nwins("nnodes"),
        SolverEvaluationResults.nsolved(),
        SolverEvaluationResults.auc_score("time"),
        header="medium"
    )

    r3 = r + r2



    print(r3)

    # data.performance_profile(metric="time")
    # data.performance_profile(metric="nnodes")
