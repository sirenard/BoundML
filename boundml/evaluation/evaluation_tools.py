import multiprocessing
import tempfile
from typing import Any, List

import dill
import pathos.multiprocessing as mp

import numpy as np

from boundml.evaluation.solver_evaluation_results import SolverEvaluationResults
from boundml.instances import Instances
from boundml.solvers import Solver

from boundml.core.utils import shifted_geometric_mean

class TaskGenerator:
    def __init__(self, solvers, instances, n_instances, metrics, files):
        self.solvers = solvers
        self.instances = instances
        self.n_instances = n_instances
        self.metrics = metrics
        self.files = files
        self.current_instance_path = None

        self.i = 0
        self.j = 0
    def __iter__(self):
        return self

    def __next__(self):
        if self.i == self.n_instances:
            raise StopIteration

        if self.j == 0:
            instance = next(self.instances)
            if type(instance) == str:
                self.current_instance_path = instance
            else:
                prob_file = tempfile.NamedTemporaryFile(suffix=".mps")
                instance.writeProblem(prob_file.name, verbose=False)
                self.current_instance_path = prob_file.name
                self.files[self.i] = prob_file


        solver = self.solvers[self.j]

        res = (self.i, self.j, solver, self.current_instance_path, self.metrics)
        self.j += 1
        if self.j == len(self.solvers):
            self.j = 0
            self.i += 1

        return res

def _solve(solver, prob_file_name, metrics):
    solver.solve(prob_file_name)
    return [solver[metric] for metric in metrics]

def _solve_wrapper(args):
    i, j, solver, instance_path, metrics = args
    metrics_values = _solve(solver, instance_path, metrics)
    return i, j, str(solver), metrics_values

def evaluate_solvers(solvers: List[Solver], instances: Instances, n_instances, metrics, n_cpu=0):
    if n_cpu == 0:
        n_cpu = mp.cpu_count()

    n_cpu = min(n_cpu, n_instances + 1)

    data = np.zeros((n_instances, len(solvers), len(metrics)))

    files = {}

    print(f"{'Instance':<15}" + "".join([f"{str(solver):<{15 * len(metrics)}}" for solver in solvers]))
    print(f"{'':<15}" + len(solvers) * ("".join([f"{metric:<15}" for metric in metrics])))

    task_generator = TaskGenerator(
        solvers,
        iter(instances),
        n_instances,
        metrics,
        files,
    )

    # Start the jobs
    if n_cpu > 1:
        ctx = multiprocessing.get_context("spawn")
        with mp.Pool(processes=n_cpu, maxtasksperchild=1, context=ctx) as pool:
            results_stream = pool.imap(_solve_wrapper, task_generator, chunksize=1)

            for i, j, solver_name, line in results_stream:
                if j == 0: # new line
                    print(f"{i:<15}", end="")

                for k, d in enumerate(line):
                    data[i, j, k] = d
                _print_result(line)

                if j == len(solvers) - 1 and i in files:
                    files[i].close()
                    print()
    else:
        for args in task_generator:
            i, j, solver_name, line = _solve_wrapper(args)

            if j == 0:  # new line
                print(f"{i:<15}", end="")

            for k, d in enumerate(line):
                data[i, j, k] = d
            _print_result(line)

            if j == len(solvers) - 1 and i in files:
                files[i].close()
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
        s = ss[metric] if metric in ss else 1
        mean = res.aggregate(metrics[k], lambda values: shifted_geometric_mean(values, shift=s))
        means[metrics[k]] = mean

    info = []
    for j in range(len(solvers)):
        for metric in metrics:
            info.append(means[metric][j])
    print(f"{'sg mean': <15}" + "".join([f"{val: <15.3f}" for val in info]))

    return res


def _print_result(line: list[Any]):
    print("".join([f"{d:{'<15.3f' if type(d) == float else '<15'}}" for d in line]), end="", flush=True)


if __name__ == "__main__":
    pass
