import unittest

from utils import setup_fake_environment


setup_fake_environment()

import pyscipopt
from pyscipopt import SCIP_RESULT

from boundml.solvers import DefaultScipSolver, ModularSolver
from boundml.components import BranchingComponent


class RecordingBranchComponent(BranchingComponent):
    def __init__(self, name, result):
        super().__init__()
        self.name = name
        self.result = result
        self.reset_calls = 0
        self.done_calls = 0
        self.passive_flags = []

    def reset(self, model):
        self.reset_calls += 1

    def callback(self, model, passive=True):
        self.passive_flags.append(passive)
        return self.result

    def done(self, model):
        self.done_calls += 1

    def __str__(self):
        return self.name


class SolverTests(unittest.TestCase):
    def test_default_scip_solver_builds_model(self):
        solver = DefaultScipSolver("relpscost", scip_params={"limits/time": 5})
        solver.build_model()

        self.assertIsInstance(solver.model, pyscipopt.Model)
        self.assertEqual(solver.model.params["limits/time"], 5)
        self.assertEqual(
            solver.model.params["branching/relpscost/priority"],
            9_999_999,
        )

    def test_default_scip_solver_accessors(self):
        solver = DefaultScipSolver("pscost")
        solver.solve("dummy_instance.lp")

        solver.model.build_dummy_statistics(
            time=1.5,
            nnodes=12,
            obj=3.14,
            gap=0.0,
            tree_est=25.0,
            best_sol="solution",
        )

        self.assertEqual(solver["time"], 1.5)
        self.assertEqual(solver["nnodes"], 12)
        self.assertEqual(solver["obj"], 3.14)
        self.assertEqual(solver["gap"], 0.0)
        self.assertEqual(solver["sol"], "solution")
        self.assertEqual(solver["estimate_nnodes"], 12)

        solver.model.build_dummy_statistics(gap=0.5, tree_est=40.0)
        self.assertEqual(solver["estimate_nnodes"], 40.0)

    def test_modular_solver_installs_branchrule_and_calls_components(self):
        active = RecordingBranchComponent("active", SCIP_RESULT.BRANCHED)
        passive = RecordingBranchComponent("passive", SCIP_RESULT.DIDNOTRUN)
        solver = ModularSolver(active, passive)

        solver.solve("instance.lp")

        self.assertEqual(active.reset_calls, 1)
        self.assertEqual(passive.reset_calls, 1)
        self.assertEqual(active.done_calls, 1)
        self.assertEqual(passive.done_calls, 1)

        # Branchrule should have been added to the model.
        self.assertEqual(len(solver.model.branch_rules), 1)

        # The active component receives passive=False, the next sees passive=True.
        self.assertEqual(active.passive_flags, [False])
        self.assertEqual(passive.passive_flags, [True])

        # String representation concatenates components.
        self.assertIn("active", str(solver))
        self.assertIn("passive", str(solver))


if __name__ == "__main__":
    unittest.main()
