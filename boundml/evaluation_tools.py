import pickle
import sys
import tempfile

import latextable
import numpy as np
from texttable import Texttable

from boundml.output_control import OutputControl
from boundml.solvers import Solver
import matplotlib
import matplotlib.pyplot as plt

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


def evaluate_solvers(solvers: [Solver], instances, n_instances, metrics):
    print(f"{'Instance':<15}" + "".join([f"{str(solver):<{15 * len(metrics)}}" for solver in solvers]))
    print(f"{'':<15}" + len(solvers) * ("".join([f"{metric:<15}" for metric in metrics])))
    output_control = OutputControl()

    data = np.zeros((n_instances, len(solvers), len(metrics)))

    prob_file = tempfile.NamedTemporaryFile(suffix=".lp")

    crashed_instance_indexes = []

    for i, instance in zip(range(n_instances), instances):
        output_control.mute()
        instance.as_pyscipopt().writeProblem(prob_file.name)
        output_control.unmute()

        print(f"{i:<15}", end="")
        for j, solver in enumerate(solvers):
            solver.solve(prob_file.name)
            line = [solver[metric] for metric in metrics]
            for k, d in enumerate(line):
                data[i, j, k] = d
            print("".join([f"{d:{'<15.3f' if type(d) == float else '<15'}}" for d in line]), end="", flush=True)

        data = np.delete(data, crashed_instance_indexes, axis=0)

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

    prob_file.close()
    return res


if __name__ == "__main__":
    data: SolverEvaluationResults = pickle.load(open("../data", "rb"))

    r = data.report(
        SolverEvaluationResults.sg_metric("nnodes", 10),
        SolverEvaluationResults.sg_metric("time", 1),
        SolverEvaluationResults.nwins("nnodes"),
        SolverEvaluationResults.nsolved(),
        SolverEvaluationResults.auc_score("time"))

    print(r)

    data.performance_profile(metric="time")
    data.performance_profile(metric="nnodes")
