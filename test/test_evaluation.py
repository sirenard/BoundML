import unittest
from types import SimpleNamespace

from utils import setup_fake_environment


setup_fake_environment()

import numpy as np

from boundml.evaluation.evaluation_tools import evaluate_solvers
from boundml.evaluation.solver_evaluation_results import SolverEvaluationResults
from boundml.instances.instances import Instances
from boundml.solvers.solvers import Solver


class DummyProblem:
    def __init__(self):
        self.written = False

    def writeProblem(self, filename, verbose=False):
        with open(filename, "w", encoding="utf-8") as handle:
            handle.write("test\n")
        self.written = True


class DummyInstances(Instances):
    def __init__(self, count):
        self.count = count
        self.generated = 0

    def __next__(self):
        if self.generated >= self.count:
            raise StopIteration
        self.generated += 1
        return DummyProblem()


class DummySolver(Solver):
    def __init__(self, name, metrics):
        self.name = name
        self.metrics = metrics
        self.calls = 0

    def build_model(self):
        pass

    def solve(self, instance):
        self.last_instance = instance
        self.calls += 1

    def __getitem__(self, item):
        return self.metrics[item]

    def __str__(self):
        return self.name


class EvaluationTests(unittest.TestCase):
    def test_evaluate_solvers_collects_metrics(self):
        metrics = ["time", "nnodes"]
        solvers = [
            DummySolver("A", {"time": 1.0, "nnodes": 5}),
            DummySolver("B", {"time": 2.0, "nnodes": 3}),
        ]
        instances = DummyInstances(count=2)

        results = evaluate_solvers(solvers, instances, 2, metrics, n_cpu=1)

        self.assertEqual(results.data.shape, (2, 2, 2))
        self.assertEqual(results.solvers, ["A", "B"])
        self.assertEqual(results.metrics, metrics)
        self.assertTrue(all(solver.calls == 2 for solver in solvers))

    def test_solver_evaluation_results_utilities(self):
        data = np.array(
            [
                [[1.0, 6.0, 0.0], [2.0, 4.0, 0.0]],
                [[3.0, 5.0, 0.5], [1.5, 6.0, 0.0]],
            ]
        )

        results = SolverEvaluationResults(data, ["A", "B"], ["time", "nnodes", "gap"])

        self.assertEqual(results.metric_index, {"time": 0, "nnodes": 1, "gap": 2})

        aggregated = results.aggregate("time", np.mean)
        self.assertTrue(np.allclose(aggregated, np.array([2.0, 1.75])))

        positives, negatives = results.split_instances_over("gap", lambda row: row <= 0)
        self.assertEqual(positives.data.shape[0], 1)
        self.assertEqual(negatives.data.shape[0], 1)

        perf = positives.performance_profile(metric="time", plot=False, logx=False)
        self.assertEqual(perf.shape, (2,))

        results.remove_solver("A")
        self.assertEqual(results.solvers, ["B"])
        self.assertEqual(results.data.shape[1], 1)

        report = positives.compute_report(
            SolverEvaluationResults.sg_metric("time", 1),
            SolverEvaluationResults.nwins("nnodes"),
            SolverEvaluationResults.nsolved(),
            SolverEvaluationResults.auc_score("time", logx=False),
        )
        self.assertIn("solver", report.df.columns.get_level_values(-1))


if __name__ == "__main__":
    unittest.main()
