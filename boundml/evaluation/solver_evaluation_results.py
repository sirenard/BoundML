from typing import List

import matplotlib
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from tabulate import tabulate

from boundml.core.utils import shifted_geometric_mean
from boundml.evaluation.reporters import TextReporter


class SolverEvaluationResults:
    def __init__(self, raw_data: np.ndarray, solvers: List[str], metrics: List[str], names: List[str] | None = None):
        self.data = raw_data
        self.solvers = solvers
        self.metrics = metrics
        self.names = names

    @property
    def metric_index(self) -> dict:
        """Returns a dictionary mapping metric names to their indices"""
        return {metric: idx for idx, metric in enumerate(self.metrics)}

    def get_metric_data(self, metric: str, std=False, count_zeros=False, min_val = -float("inf"), max_val = float("inf")) -> np.ndarray:
        """Get all data for a specific metric. Average over all the seeds (or std if std=True)"""

        if metric == "names" and self.names:
            return self.names

        data = self.data[:, :, :, self.metric_index[metric]]
        data = np.clip(data, min_val, max_val)
        if not count_zeros:
            mask = np.any(data.reshape(data.shape[0], -1) != 0, axis=1)
            data = data[mask]

        if std:
            data = np.std(data, axis=2)
            mean = self.get_metric_data(metric, False, count_zeros)
            data = data / mean * 100
        else:
            data = np.median(data, axis=2)
        return data

    def aggregate(self, metric: str, aggregation_func: callable, std=False, count_zeros=False, min_val = -float("inf"), max_val = float("inf")) -> np.ndarray:
        """
        Apply aggregation function to a specific metric
        Args:
            metric: metric name to aggregate
            aggregation_func: function to apply (e.g., np.sum, np.mean)
            std: If True the aggregation is done on the std over the seeds. Else over the mean.
        """
        res = np.array([aggregation_func(self.get_metric_data(metric, std, count_zeros, min_val, max_val)[:, i]) for i in range(len(self.solvers))])
        return np.nan_to_num(res)

    def split_instances_over(self, metric: str, condition, require_all: bool = True):
        """
        Splits instances into positives and negatives based on a condition.

        Parameters
        ----------
        metric : str
            The metric to evaluate the condition against.
        condition : callable
            A function that returns a boolean array when applied to the metric data.
        require_all : bool, default True
            If True, ALL solvers must meet the condition for an instance to be positive.
            If False, AT LEAST ONE solver must meet the condition for an instance to be positive.

        Returns
        -------
        tuple
            Two SolverEvaluationResults objects: (positives, negatives)
        """
        assert metric in self.metrics, "Cannot make a split on a non-existing metric"

        d = self.get_metric_data(metric, count_zeros=True)

        # Apply the condition to the data
        condition_met = np.apply_along_axis(condition, 1, d)

        # Filter based on the require_all flag
        if require_all:
            indexes = np.where(np.all(condition_met, axis=1))[0]
        else:
            indexes = np.where(np.any(condition_met, axis=1))[0]

        positives = self.data[indexes,]
        negatives = np.delete(self.data, indexes, axis=0)

        # For compatibility with older results
        if not hasattr(self, 'names'):
            self.names = None

        if self.names is not None:
            names_pos = [self.names[i] for i in indexes]
            names_neg = [self.names[i] for i in range(len(self.names)) if i not in indexes]
        else:
            names_pos = None
            names_neg = None

        return (
            SolverEvaluationResults(positives, self.solvers, self.metrics, names_pos),
            SolverEvaluationResults(negatives, self.solvers, self.metrics, names_neg),
        )

    def remove_solver(self, solver: str):
        index = self.solvers.index(solver)
        self.data = np.delete(self.data, index, axis=1)
        self.solvers.remove(solver)

    def remove_metric(self, metric: str):
        index = self.metric_index[metric]
        self.data = np.delete(self.data, index, axis=3)
        self.metrics.pop(index)

    def performance_profile(self, metric: str = "nnodes", ratios=np.arange(0, 1.00, .01), filename=None, plot=True, logx=True):

        if filename:
            backend = matplotlib.get_backend()
            matplotlib.use('pgf')

        n_instances = self.data.shape[0]

        data = self.get_metric_data(metric)
        min = np.min(data)
        max = np.max(data)

        xs = ratios * (max - min) + min

        res = []
        for s, solver in enumerate(self.solvers):
            ys = np.zeros(len(ratios))
            for i in range(n_instances):
                val = data[i, s]
                indexes = np.where(val <= xs)
                ys[indexes] += 1

            ys /= n_instances
            label = solver

            if logx:
                auc = np.trapezoid(ys, np.log(xs)) / np.log(max)
            else:
                auc = np.trapezoid(ys, xs) / max

            res.append(auc)
            if plot:
                plt.plot(xs, ys, label=label)

        if plot:
            plt.legend()
            plt.xlabel(metric)
            plt.ylabel("frequency")
            plt.title(f"Performance profile w.r.t. {metric}")

            if logx:
                plt.xscale("log")

            if filename:
                plt.savefig(filename)
                matplotlib.use(backend)

            else:
                plt.show()

        return np.array(res)

    def compute_report(self, *aggregations: tuple[str, callable], **kwargs):
        data = {"solver": [s for s in self.solvers]}

        for i, aggregation in enumerate(aggregations):
            data[aggregation[0]] = list(aggregation[1](self))

        return SolverEvaluationReport(data, **kwargs)

    def combine_solvers(self, other):
        """
        Combine the results of another SolverEvaluationResults.
        The 2 SolverEvaluationResults (self and other) must have the same metrics and be for the same instances.
        Parameters
        ----------
        other : SolverEvaluationResults
            The result to combine

        Returns
        -------
        The new SolverEvaluationResults object
        """
        assert self.metrics == other.metrics, "Both results have different metrics"
        assert self.data.shape[0] == other.data.shape[0], "Both results have different number of instances"
        assert self.data.shape[3] == other.data.shape[3], "Both results have different number of seeds repetitions"

        # For compatibility with older results
        self_names = getattr(self, 'names', None)
        other_names = getattr(other, 'names', None)

        assert self_names == other_names, "Both results solved instances have different names"

        solvers = self.solvers + other.solvers
        data = np.hstack((self.data, other.data))

        return SolverEvaluationResults(data, solvers, self.metrics[:], self_names)

    def combine_instances(self, other):
        """
        Combine the results of another SolverEvaluationResults.
        The 2 SolverEvaluationResults (self and other) must have the same metrics and be for the same solvers.
        Parameters
        ----------
        other : SolverEvaluationResults
            The result to combine

        Returns
        -------
        The new SolverEvaluationResults object
        """
        assert self.metrics == other.metrics, "Both results have different metrics"
        assert self.data.shape[3] == other.data.shape[3], "Both results have different number of seeds repetitions"
        assert self.solvers == other.solvers, "Both results have different solvers"

        # For compatibility with older results
        self_names = getattr(self, 'names', None)
        other_names = getattr(other, 'names', None)

        assert type(self_names) == type(other_names), \
            "Both results have different names structure. Once has names for the instances, not the other"

        data = np.vstack((self.data, other.data))

        names = None
        if self.names is not None:
            names = self_names + other_names

        return SolverEvaluationResults(data, self.solvers[:], self.metrics[:], names)

    @staticmethod
    def sg_metric(metric, s, std=False, min_val = -float("inf"), max_val = float("inf")):
        name = metric if not std else f"{metric} std (%)"
        return (name, lambda evaluationResults:
        evaluationResults.aggregate(metric, lambda values: shifted_geometric_mean(values, shift=s), std, count_zeros=False, min_val=min_val, max_val=max_val)
                )

    @staticmethod
    def nwins(metric, dir=1, count_if_not_optimal = False):
        def get_wins(evaluationResults: SolverEvaluationResults):
            data = evaluationResults.get_metric_data(metric, count_zeros=True)
            gaps = evaluationResults.get_metric_data("gap", count_zeros=True)
            res = []
            for i in range(len(evaluationResults.solvers)):
                c = 0
                for j in range(len(data[:, i])):
                    # Does not count as a win if the instance was not solved optimally.
                    if count_if_not_optimal or gaps[j, i] == 0 or metric == "gap":
                        c += dir * data[j, i] <= dir * np.min(data[j, :])
                res.append(c)
            return np.array(res)

        return f"wins ({metric})", get_wins

    @staticmethod
    def nsolved():
        return ("nsolved",
                lambda evaluationResults: evaluationResults.aggregate(
                    "gap",
                    lambda values: values.shape[0] - np.count_nonzero(values),
                    count_zeros=True
                )
            )

    @staticmethod
    def auc_score(metric, **kwargs):
        return ("AUC", lambda evaluationResults: evaluationResults.performance_profile(metric, plot=False, **kwargs))

    def get_names(self):
        return self.names

    def __str__(self):
        reporter = TextReporter()
        res = ""
        res += reporter.on_evaluation_start(self.solvers, self.metrics)

        n_instances = self.data.shape[0]
        n_solvers = len(self.solvers)

        for i in range(n_instances):
            instance_name = self.names[i] if self.names is not None else str(i)
            res += reporter.on_instance_start(instance_name)
            for j in range(n_solvers):
                line = self.data[i, j, :, :]
                line = np.mean(line, axis=0)
                res += reporter.on_solver_finish(line)

            res += reporter.on_instance_end()

        res += reporter.on_evaluation_end(self, self.metrics, self.solvers)

        return res




class SolverEvaluationReport:
    def __init__(self, data=None, header=None, df_=None):
        assert (data is None) != (df_ is None), "Only one of data and df_ must be given"

        if df_ is not None:
            self.df = df_
            return

        if header is not None:
            data_ = {}
            for key in data:
                if key != "solver":
                    data_[(header, key)] = data[key]
                else:
                    data_[("", key)] = data[key]

        else:
            data_ = data

        self.df = pd.DataFrame(data_)
        if header is not None:
            self.df.set_index(("","solver"), inplace=True)

    def __str__(self):
        return tabulate(self.df, headers="keys", tablefmt='grid', showindex=False)

    def to_latex(self, *args, **kwargs):
        return self.df.to_latex(index=False, *args, **kwargs)

    def __add__(self, other):
        print(self.df.to_dict(orient='list'))
        print(other.df.to_dict(orient='list'))
        df2 = pd.concat(
            [self.df, other.df],
            axis=1
        )

        df2 = df2.reset_index().rename(columns={'index': ('', 'solver')})
        return SolverEvaluationReport(df_ = df2)