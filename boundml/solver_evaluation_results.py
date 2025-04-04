import latextable
import matplotlib
import numpy as np
from matplotlib import pyplot as plt
from texttable import Texttable

from boundml.utils import shifted_geometric_mean


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

    def performance_profile(self, metric: str = "nnodes", ratios=np.arange(0, 1.00, .01), filename=None, plot=True):

        if filename:
            backend = matplotlib.get_backend()
            matplotlib.use('pgf')

        metric_index = self.metrics.index(metric)
        n_instances = self.data.shape[0]

        frequency = np.zeros((len(self.solvers), len(ratios)))

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
            res.append(np.sum(ys * (xs[1] - xs[0])) / max)
            if plot:
                plt.plot(xs, ys, label=label)

        if plot:
            plt.legend()
            plt.xlabel(metric)
            plt.ylabel("frequency")
            plt.title(f"Performance profile w.r.t. {metric}")
            # plt.yscale("log")
            plt.xscale("log")

            if filename:
                plt.savefig(filename)
                matplotlib.use(backend)

            else:
                plt.show()

        return np.array(res)

    def report(self, *aggregations: tuple[str, callable], latex_output=False):
        table = Texttable()

        table.add_row(["Solver"] + [aggregation[0] for aggregation in aggregations])

        data = np.zeros((len(aggregations), len(self.solvers)))
        for i, aggregation in enumerate(aggregations):
            data[i, :] = aggregation[1](self)

        for i, solver_name in enumerate(self.solvers):
            row = [solver_name]
            row.extend(data[:, i])
            table.add_row(row)

        if latex_output:
            return latextable.draw_latex(table, caption="Experimental results", label="table")
        else:
            return table.draw()

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
    def auc_score(metric):
        return ("AUC", lambda evaluationResults: evaluationResults.performance_profile(metric, plot=False))
