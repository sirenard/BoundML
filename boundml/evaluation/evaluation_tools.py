import tempfile
import dill
import pathos.multiprocessing as mp

import numpy as np

from boundml.evaluation.solver_evaluation_results import SolverEvaluationResults
from boundml.instances import Instances
from boundml.solvers import Solver

from boundml.core.utils import shifted_geometric_mean

def _solve(solver, prob_file_name, metrics):
    solver.solve(prob_file_name)
    return [solver[metric] for metric in metrics]


def evaluate_solvers(solvers: [Solver], instances: Instances, n_instances, metrics, n_cpu=0):
    if n_cpu == 0:
        n_cpu = mp.cpu_count()

    n_cpu = min(n_cpu, n_instances + 1)

    data = np.zeros((n_instances, len(solvers), len(metrics)))

    files = {}
    async_results = {}

    # Start the jobs
    with mp.Pool(processes=n_cpu) as pool:
        for i, instance in zip(range(n_instances), instances):
            for j, solver in enumerate(solvers):
                prob_file = tempfile.NamedTemporaryFile(suffix=".lp")
                instance.writeProblem(prob_file.name, verbose=False)

                files[i,j] = prob_file
                async_results[i,j] = pool.apply_async(_solve, [solver, prob_file.name, metrics])

        print(f"{'Instance':<15}" + "".join([f"{str(solver):<{15 * len(metrics)}}" for solver in solvers]))
        print(f"{'':<15}" + len(solvers) * ("".join([f"{metric:<15}" for metric in metrics])))
        for i, instance in zip(range(n_instances), instances):
            print(f"{i:<15}", end="")
            for j, solver in enumerate(solvers):
                line = async_results[i,j].get()
                files[i,j].close()
                for k, d in enumerate(line):
                    data[i, j, k] = d
                print("".join([f"{d:{'<15.3f' if type(d) == float else '<15'}}" for d in line]), end="", flush=True)

            print()

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

    return res


if __name__ == "__main__":
    data: SolverEvaluationResults = dill.load(open("../data", "rb"))

    r = data.compute_report(
        SolverEvaluationResults.sg_metric("nnodes", 10),
        SolverEvaluationResults.sg_metric("time", 1),
        SolverEvaluationResults.nwins("nnodes"),
        SolverEvaluationResults.nsolved(),
        SolverEvaluationResults.auc_score("time"),
    )

    print(r)

    data.performance_profile(metric="time")
    data.performance_profile(metric="nnodes")
