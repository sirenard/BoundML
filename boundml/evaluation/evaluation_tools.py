import concurrent.futures
import os
import signal
import tempfile
import time
import warnings
import multiprocessing
from typing import List, Callable, Optional

import numpy as np
import psutil

from boundml.evaluation.reporters import BaseReporter, ConsoleReporter
from boundml.evaluation.solver_evaluation_results import SolverEvaluationResults
from boundml.instances import Instances
from boundml.solvers import Solver


class TaskGenerator:
    def __init__(self, solvers, instances, n_instances, seeds, metrics, files, save_instances_names=False, *args):
        self.solvers = solvers
        self.instances = instances
        self.n_instances = n_instances
        self.seeds = seeds
        self.metrics = metrics
        self.files = files
        self.current_instance_path = None
        self.current_instance_name = None
        self.save_instances_names = save_instances_names

        self.i = 0
        self.j = 0
        self.s = 0
        self.args = args
    def __iter__(self):
        return self

    def __next__(self):
        if self.i == self.n_instances:
            raise StopIteration

        if self.j == 0 and self.s==0: # new instance
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
        seed = self.seeds[self.s]

        res = (self.i, self.j, self.s, seed, solver, self.current_instance_path, self.metrics, self.current_instance_name, *self.args)

        self.s += 1 # Pass to the next seed
        if self.s == len(self.seeds): # pass to the next solver
            self.s = 0
            self.j += 1
        if self.j == len(self.solvers): # pass to the next instance
            self.j = 0
            self.i += 1

        return res


class Evaluator:
    """
    Evaluates a set of solvers against a set of instances.
    Separates the configuration from the parallel execution logic.
    """

    def __init__(
            self,
            metrics: List[str],
            fail_on_error: bool = True,
            limit_gbytes: Optional[int] = None,
            reporter: Optional[BaseReporter] = None,
            callback: Callable[[str, int, int, int, np.ndarray], None] | None = None
    ):
        """
        Parameters
        ----------
        fail_one_error : bool
            Whether to raise an exception when a solver fails.
            If True and an error occurs, the resulting metrics are all 0.
            Default it False.
        limit_gbytes : int | None
            Memory limit applied to the children processes in GB. If None, no limit is applied.
            When specified, if the child reach the memory limit, it catches the exception and cancel the solving process.
            All the resulting metrics are 0.
            Default None.
        reporter: Optional[BaseReporter]
            BaseReporter used to report the results during the evalution.
            If None, a simple ComsoleReporter is built. It prints the results of the solvers on stdout
        callback: Callable[[str, int, int, int, np.ndarray], None] | None
            Callback function called after an instance is solved by a solver. Take as argument the instance name,
            the instance index, the solver index, the ndarray d containing all the results. d[i,j,s,:] contains all the
            metrics from the solving of instances i by solver j with the seed seeds[s].
        """
        self.metrics = metrics
        self.fail_on_error = fail_on_error
        self.limit_gbytes = limit_gbytes
        self.reporter = reporter if reporter is not None else ConsoleReporter()  # Default to a console reporter
        self.callback = callback

    @staticmethod
    def _monitor_memory(pid, limit_bytes, stop_event):
        try:
            process = psutil.Process(pid)
        except psutil.NoSuchProcess:
            return
        while not stop_event.is_set():
            try:
                # Check strictly PHYSICAL memory (RSS)
                rss = process.memory_info().rss
                if rss > limit_bytes:
                    os.kill(pid, signal.SIGKILL)
                    break
            except psutil.NoSuchProcess:
                break
            time.sleep(1)

    @staticmethod
    def _solve(solver, prob_file_name, metrics, seed, fail_on_error):
        try:
            solver.set_seed(seed)
            solver.solve(prob_file_name)
            return [solver[metric] for metric in metrics]
        except Exception as e:
            if fail_on_error:
                raise e
            warnings.warn(f"Error while solving {prob_file_name} with {solver}: {e}")
            return [0 for _ in metrics]

    @staticmethod
    def _solve_wrapper(args):
        i, j, s, seed, solver, instance_path, metrics, instance_name, fail_on_error, limit_rss_bytes = args

        # We create a Queue to get the results back from the isolated solver process
        result_queue = multiprocessing.Queue()

        def isolated_solve():
            try:
                res = Evaluator._solve(solver, instance_path, metrics, seed, fail_on_error)
                result_queue.put(res)
            except Exception as e:
                result_queue.put(e)

        # Start the solver in its own process
        worker = multiprocessing.Process(target=isolated_solve)
        worker.start()
        worker_pid = worker.pid

        # Start the monitor to watch the worker
        stop_event = multiprocessing.Event()
        watcher = None
        if limit_rss_bytes is not None:
            watcher = multiprocessing.Process(
                target=Evaluator._monitor_memory,
                args=(worker_pid, limit_rss_bytes, stop_event),
                daemon=True
            )
            watcher.start()

        worker.join()  # Wait for solver to finish OR be killed by monitor
        stop_event.set()  # Stop the monitor if it's still running

        # Check why the worker stopped
        if worker.exitcode == -signal.SIGKILL:
            warnings.warn(
                f"Instance {instance_name} killed: Memory limit {limit_rss_bytes / 1024 ** 3:.2f} GB exceeded.")
            metrics_values = [0 for _ in metrics]
        else:
            # Success: Get the result from the queue
            metrics_values = result_queue.get(timeout=2)
            if isinstance(metrics_values, Exception):
                if fail_on_error:
                    raise metrics_values
                else:
                    warnings.warn(metrics_values)
                    metrics_values = [0 for _ in metrics]

        if watcher and watcher.is_alive():
            watcher.terminate()

        return i, j, s, metrics_values, instance_name

    def evaluate(
            self,
            solvers: List[Solver],
            instances: Instances,
            n_instances: int,
            seeds: List[int] = (0,),
            executor: Optional[concurrent.futures.Executor] = None,
            display_instance_names: bool = False
    ) -> SolverEvaluationResults:
        """
        Executes the evaluation.

        Parameters
        ----------
        solvers : List[Solver]
            List of solvers that will solve each instance
        instances : Instances
            Instances generator. Yields either pyscipopt Model or a str path.
        n_instances : int
            Number of instances to evaluate
        seeds: List[int]
            List of seeds used to solve an instance.
        executor : concurrent.futures.Executor | None
            A pool executor for parallel processing. If None, runs sequentially.
            Compatible with ProcessPoolExecutor, ThreadPoolExecutor, or MPIPoolExecutor.
        display_instance_names : bool
            Whether to record and display instance names. Default is False.

        Returns
        -------
        Return a SolverEvaluationResults object which can be used to compute a report on the computed data.
        See SolverEvaluationReport for more details
        """
        names = []
        limit_rss_bytes = self.limit_gbytes * (1024 ** 3) if self.limit_gbytes is not None else None

        data = np.zeros((n_instances, len(solvers), len(seeds), len(self.metrics)))
        files = {}

        task_generator = TaskGenerator(
            solvers,
            iter(instances),
            n_instances,
            seeds,
            self.metrics,
            files,
            display_instance_names,
            self.fail_on_error,
            limit_rss_bytes
        )

        self.reporter.on_evaluation_start([str(s) for s in solvers], self.metrics)

        def _process_result(i, j, s, line, instance_name):
            if j == 0 and s == 0:  # new line
                names.append(instance_name)
                self.reporter.on_instance_start(instance_name)

            for k, d in enumerate(line):
                data[i, j, s, k] = d

            if s == len(seeds) - 1:
                l = data[i, j, :, :]
                mean_line = np.mean(l, axis=0)
                self.reporter.on_solver_finish(mean_line)

            if self.callback is not None:
                self.callback(instance_name, i, j, s, data)

            if j == len(solvers) - 1 and s == len(seeds) - 1:
                self.reporter.on_instance_end()
                if i in files:
                    files[i].close()

        # Execute tasks
        if executor is not None:
            # Map returns an iterator yielding results in the exact same order tasks were generated
            results_stream = executor.map(Evaluator._solve_wrapper, task_generator)
            for solve_res in results_stream:
                _process_result(*solve_res)
        else:
            # Sequential fallback
            for args in task_generator:
                solve_res = Evaluator._solve_wrapper(args)
                _process_result(*solve_res)

        res = SolverEvaluationResults(
            data,
            [str(s) for s in solvers],
            self.metrics,
            names if display_instance_names else None
        )

        self.reporter.on_evaluation_end(res, self.metrics, [str(s) for s in solvers])

        return res