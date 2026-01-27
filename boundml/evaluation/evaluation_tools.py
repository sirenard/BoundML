import multiprocessing
import os
import resource
import signal
import tempfile
import threading
import time
import warnings
from typing import Any, List, Callable

import numpy as np
import multiprocessing as mp

import psutil

from boundml.core.utils import shifted_geometric_mean
from boundml.evaluation.solver_evaluation_results import SolverEvaluationResults
from boundml.instances import Instances
from boundml.solvers import Solver


class TaskGenerator:
    def __init__(self, solvers, instances, n_instances, metrics, files, save_instances_names=False, *args):
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
        self.args = args
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

        res = (self.i, self.j, solver, self.current_instance_path, self.metrics, self.current_instance_name, *self.args)
        self.j += 1
        if self.j == len(self.solvers):
            self.j = 0
            self.i += 1

        return res

def _monitor_memory(pid, limit_bytes, stop_event):
    process = psutil.Process(pid)
    while not stop_event.is_set():
        try:
            # Check strictly PHYSICAL memory (RSS)
            rss = process.memory_info().rss
            if rss > limit_bytes:
                warnings.warn(f"[{pid}] KILLED: Used {rss/1024**3:.2f} GB > Limit {limit_bytes/1024**3:.2f} GB")

                # Setting the RLIMIT_AS now will force the underlying solver to crash.
                resource.setrlimit(resource.RLIMIT_AS, (limit_bytes, limit_bytes))
                break
        except psutil.NoSuchProcess:
            break
        time.sleep(1)

def _solve(solver, prob_file_name, metrics, fail_on_error, fail_on_memory_error):
    try:
        solver.solve(prob_file_name)
        return [solver[metric] for metric in metrics]
    except MemoryError as e:
        print(fail_on_memory_error)
        if fail_on_memory_error:
            raise e
        warnings.warn(f"Memory usage reached while solvign {prob_file_name} with {solver}")
        return [0 for _ in metrics]
    except Exception as e:
        if fail_on_error:
            raise e
        warnings.warn(f"Error while solving {prob_file_name} with {solver}: {e}")
        return [0 for _ in metrics]

def _solve_wrapper(args):
    i, j, solver, instance_path, metrics, instance_name, fail_on_error, limit_rss_bytes = args

    stop_event, watcher = None, None
    if limit_rss_bytes is not None:
        stop_event = threading.Event()
        watcher = threading.Thread(target=_monitor_memory, args=(os.getpid(), limit_rss_bytes, stop_event))
        watcher.start()

    try:
        metrics_values = _solve(solver, instance_path, metrics, fail_on_error, limit_rss_bytes is None)
    finally:
        if limit_rss_bytes is not None:
            stop_event.set()
            watcher.join()


    return i, j, metrics_values, instance_name


def evaluate_solvers(solvers: List[Solver], instances: Instances, n_instances: int, metrics: List[str], n_cpu: int = 0,
                     display_instance_names: bool = False, fail_one_error: bool = False, limit_gbytes: int | None = None,
                     callback: Callable[[str, int, int, np.ndarray], None] | None = None) -> SolverEvaluationResults:
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
    fail_one_error : bool
        Whether to raise an exception when a solver fails.
        If True and an error occurs, the resulting metrics are all 0.
        Default it False.
    limit_gbytes : int | None
        Memory limit applied to the children processes in GB. If None, no limit is applied.
        When specified, if the child reach the memory limit, it catches the exception and cancel the solving process.
        All the resulting metrics are 0.
        /!\ Unexpected behavior when n_cpu is 1. As no multiprocessing is used, it will change the memory limit
        of the main process.
        Default None.
    callback: Callable[[str, int, int, np.ndarray], None] | None
        Callback function called after an instance is solved by a solver. Take as argument the instance name,
        the instance index, the solver index, the ndarray d containing all the results. d[i,j,:] contains all the
        metrics from the solving of instances i by solver j.
    Returns
    -------
    Return a SolverEvaluationResults object which can be used to compute a report on the computed data.
    See SolverEvaluationReport for more details
    """
    if n_cpu == 0:
        n_cpu = mp.cpu_count()

    limit_rss_bytes = None
    if limit_gbytes is not None:
        limit_rss_bytes = limit_gbytes * (1024**3)

    n_cpu = min(n_cpu, n_instances * len(solvers) + 1)

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
        fail_one_error,
        limit_rss_bytes
    )

    def _process_result(i, j, line, instance_name):
        if j == 0:  # new line
            print(f"{instance_name:<15}", end="", flush=True)

        for k, d in enumerate(line):
            data[i, j, k] = d

        _print_result(line)

        if callback is not None:
            callback(instance_name, i, j, data)
        if j == len(solvers) - 1:
            print()
            if i in files:
                files[i].close()
    # Start the jobs
    if n_cpu > 1:
        ctx = multiprocessing.get_context("spawn")
        with mp.Pool(processes=n_cpu) as pool:
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
