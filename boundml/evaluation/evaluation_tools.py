import multiprocessing
import tempfile
from typing import Any, List

import numpy as np
import pathos.multiprocessing as mp

from boundml.core.utils import shifted_geometric_mean
from boundml.evaluation.solver_evaluation_results import SolverEvaluationResults
from boundml.instances import Instances
from boundml.solvers import Solver


class TaskGenerator:
    def __init__(self, solvers, instances, n_instances, metrics, files, save_instances_names=False):
        self.solvers = solvers
        self.instances = instances
        self.n_instances = n_instances
        self.metrics = metrics
        self.files = files
        self.current_instance_path = None
        self.current_instance_name = None
        self.save_instances_names = save_instances_names

        self.i = 0
        self.j = 0
    def __iter__(self):
        return self

    def __next__(self):
        if self.i == self.n_instances:
            raise StopIteration

        if self.j == 0:
            instance = next(self.instances)
            if type(instance) == str:  # instance is a path
                self.current_instance_path = instance
                name = instance.split("/")[-1].split(".")[0]  # get the instance name
            else:
                prob_file = tempfile.NamedTemporaryFile(suffix=".mps")
                instance.writeProblem(prob_file.name, verbose=False)
                self.current_instance_path = prob_file.name
                name = instance.getProbName()
                self.files[self.i] = prob_file

            if self.save_instances_names:
                self.current_instance_name = name
            else:
                self.current_instance_name = str(self.i)


        solver = self.solvers[self.j]

        res = (self.i, self.j, solver, self.current_instance_path, self.metrics, self.current_instance_name)
        self.j += 1
        if self.j == len(self.solvers):
            self.j = 0
            self.i += 1

        return res

def _solve(solver, prob_file_name, metrics):
    solver.solve(prob_file_name)
    return [solver[metric] for metric in metrics]

def _solve_wrapper(args):
    i, j, solver, instance_path, metrics, instance_name = args
    metrics_values = _solve(solver, instance_path, metrics)
    return i, j, metrics_values, instance_name


def evaluate_solvers(solvers: List[Solver], instances: Instances, n_instances: int, metrics: List[str], n_cpu: int = 0,
                     display_instance_names: bool = False) -> SolverEvaluationResults:
    """
    Evaluate a set of solvers against a set of instances in parallel.
    It prints as soon as possible the results for each solver on each instance.

    Parameters
    ----------
    solvers : List[Solver]
        List of solvers that will solve each instance
    instances : Instances
        Instances generator used to generate all the instances. It can be a list. It must yield either pyscipopt Model
        or a str that is a path to a problem file
    n_instances : int
        Number of instances to evaluate
    metrics : List[str]
        List of metrics reported and saved (e.g. "time", "nnodes", "gap", ...). See ScipSolver for more options.
    n_cpu :
        Number of processes to use to run the solvers in parallel
        If 0, it uses all the available cores.
        If 1, no multiprocessing is used.
        Default is 0
    display_instance_names : bool
        Whether to display instance names or simple numbering. Default is False.

    Returns
    -------
    Return a SolverEvaluationResults object which can be used to compute a report on the computed data.
    See SolverEvaluationReport for more details
    """
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
        display_instance_names,
    )

    def _process_result(i, j, line, instance_name):
        if j == 0:  # new line
            print(f"{instance_name:<15}", end="", flush=True)

        for k, d in enumerate(line):
            data[i, j, k] = d
        _print_result(line)

        if j == len(solvers) - 1:
            print()
            if i in files:
                files[i].close()
    # Start the jobs
    if n_cpu > 1:
        ctx = multiprocessing.get_context("spawn")
        with mp.Pool(processes=n_cpu, maxtasksperchild=1, context=ctx) as pool:
            results_stream = pool.imap(_solve_wrapper, task_generator, chunksize=1)

            for solve_res in results_stream:
                _process_result(*solve_res)
    else:
        for args in task_generator:
            solve_res = _solve_wrapper(args)
            _process_result(*solve_res)

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
    print(f"{'sg mean': <15}" + "".join([f"{val: <15.5g}" for val in info]))

    return res


def _print_result(line: list[Any]):
    print("".join([f"{d:<15.5g}" for d in line]), end="", flush=True)


if __name__ == "__main__":
    pass
