from typing import List

import numpy as np

from boundml.core.utils import shifted_geometric_mean
from boundml.solvers import Solver


class BaseReporter:
    """Base class for handling evaluation display and logging."""
    def on_evaluation_start(self, solvers: List[str], metrics: List[str]): pass
    def on_instance_start(self, instance_name: str): pass
    def on_solver_finish(self, mean_metrics_across_seeds: np.ndarray): pass
    def on_instance_end(self): pass
    def on_evaluation_end(self, res, metrics: List[str], solvers: List[str]): pass

class TextReporter(BaseReporter):
    """Reporter doing nothing but returning text representation."""
    @staticmethod
    def _concat(*lines):
        return "\n".join(lines)

    def on_evaluation_start(self, solvers: List[str], metrics: List[str]):
        return TextReporter._concat(*[
            f"{'Instance':<15}" + "".join([f"{str(solver):<{15 * len(metrics)}}" for solver in solvers]),
            f"{'':<15}" + len(solvers) * ("".join([f"{metric:<15}" for metric in metrics])),
            ""
        ])

    def on_instance_start(self, instance_name: str):
        return f"{instance_name:<15}"

    def on_solver_finish(self, mean_metrics_across_seeds: np.ndarray):
        return "".join([f"{d:<15.5g}" for d in mean_metrics_across_seeds])

    def on_instance_end(self):
        return "\n"

    def on_evaluation_end(self, res, metrics: List[str], solvers: List[str]):
        lines = ["=" * (15 * (len(solvers) * len(metrics) + 1))]

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
        lines.append(f"{'sg mean': <15}" + "".join([f"{val: <15.5g}" for val in info]))

        return TextReporter._concat(*lines, "")

class ConsoleReporter(TextReporter):
    def on_evaluation_start(self, solvers: List[str], metrics: List[str]):
        print(super().on_evaluation_start(solvers, metrics), end="", flush=True)

    def on_instance_start(self, instance_name: str):
        print(super().on_instance_start(instance_name), end="", flush=True)

    def on_solver_finish(self, mean_metrics_across_seeds: np.ndarray):
        print(super().on_solver_finish(mean_metrics_across_seeds), end="", flush=True)

    def on_instance_end(self):
        print(super().on_instance_end(), end="", flush=True)

    def on_evaluation_end(self, res, metrics: List[str], solvers: List[str]):
        print(super().on_evaluation_end(res, metrics, solvers), end="", flush=True)

