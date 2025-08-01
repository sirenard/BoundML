import matplotlib
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from tabulate import tabulate

from boundml.core.utils import shifted_geometric_mean


class SolverEvaluationResults:
    def __init__(self, raw_data: np.array, solvers: [str], metrics: [str]):
        self.data = raw_data
        self.solvers = solvers
        self.metrics = metrics

    @property
    def metric_index(self) -> dict:
        """Returns a dictionary mapping metric names to their indices"""
        return {metric: idx for idx, metric in enumerate(self.metrics)}

    def get_metric_data(self, metric: str) -> np.array:
        """Get all data for a specific metric"""
        return self.data[:, :, self.metric_index[metric]]

    def aggregate(self, metric: str, aggregation_func: callable) -> np.array:
        """
        Apply aggregation function to a specific metric
        Args:
            metric: metric name to aggregate
            aggregation_func: function to apply (e.g., np.sum, np.mean)
        """
        return np.array([aggregation_func(self.get_metric_data(metric)[:, i]) for i in range(len(self.solvers))])

    def split_instances_over(self, metric: str, condition):
        assert metric in self.metrics, "Cannot make a split on a non-existing metric"

        index = self.metrics.index(metric)
        d = self.data[:, :, index]  # keep only the compared metrix

        indexes = np.where(np.prod(np.apply_along_axis(condition, 1, d), axis=1))[0]

        positives = self.data[indexes,]
        negatives = np.delete(self.data, indexes, axis=0)
        return SolverEvaluationResults(positives, self.solvers, self.metrics), SolverEvaluationResults(negatives,
                                                                                                       self.solvers,
                                                                                                       self.metrics)

    def remove_solver(self, solver: str):
        index = self.solvers.index(solver)
        self.data = np.delete(self.data, index, axis=1)
        self.solvers.remove(solver)

    def performance_profile(self, metric: str = "nnodes", ratios=np.arange(0, 1.00, .01), filename=None, plot=True, logx=True):

        if filename:
            backend = matplotlib.get_backend()
            matplotlib.use('pgf')

        metric_index = self.metrics.index(metric)
        n_instances = self.data.shape[0]

        data = self.data[:, :, metric_index]
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
            if label == "relpscost":
                label = "RPB"
            elif "GNN" in label and "sb" in label:
                label = "$LSB$"
            elif "GNN" in label and "ub" in label:
                x = label.split("_")[4]
                label = f"$LLB^{x}$"

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

    def __add__(self, other):
        assert self.metrics == other.metrics, "Metrics must be the same when combining Results of different solvers"
        assert self.data.shape
        solvers = self.solvers + other.solvers
        data = np.hstack((self.data, other.data))
        return SolverEvaluationResults(data, solvers, self.metrics)

    @staticmethod
    def sg_metric(metric, s):
        return (metric, lambda evaluationResults:
        evaluationResults.aggregate(metric, lambda values: shifted_geometric_mean(values, shift=s))
                )

    @staticmethod
    def nwins(metric, dir=1):
        def get_wins(evaluationResults: SolverEvaluationResults):
            data = evaluationResults.get_metric_data(metric)
            res = []
            for i in range(len(evaluationResults.solvers)):
                c = 0
                for j in range(len(data[:, i])):
                    c += dir * data[j, i] <= dir * np.min(data[j, :])
                res.append(c)
            return np.array(res)

        return f"wins ({metric})", get_wins

    @staticmethod
    def nsolved():
        return ("nsolved", lambda evaluationResults: evaluationResults.aggregate("gap", lambda values: values.shape[
                                                                                                           0] - np.count_nonzero(
            values)))

    @staticmethod
    def auc_score(metric, **kwargs):
        return ("AUC", lambda evaluationResults: evaluationResults.performance_profile(metric, plot=False, **kwargs))

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