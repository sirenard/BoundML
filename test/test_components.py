import unittest

from utils import setup_fake_environment


setup_fake_environment()

import numpy as np

import pyscipopt
from pyscipopt import SCIP_RESULT

from boundml.components import (
    BranchingComponent,
    ScoringBranchingStrategy,
    Pseudocosts,
    StrongBranching,
    AccuracyBranching,
    ConditionalBranchingComponent,
)
from boundml.components.components import Component, ComponentList
from boundml.core.branchrule import BoundmlBranchrule


class DummyComponent(Component):
    def __init__(self):
        super().__init__()
        self.reset_calls = 0
        self.done_calls = 0

    def reset(self, model):
        self.reset_calls += 1

    def callback(self, model, passive=True):
        return None

    def done(self, model):
        self.done_calls += 1


class IdentityStrategy(ScoringBranchingStrategy):
    def compute_scores(self, model):
        return super().compute_scores(model)


class TrackingBranchingComponent(BranchingComponent):
    def __init__(self, result, name):
        super().__init__()
        self.result = result
        self.name = name
        self.calls = []

    def callback(self, model, passive=True):
        self.calls.append(passive)
        return self.result

    def __str__(self):
        return self.name


class ComponentsTests(unittest.TestCase):
    def setUp(self):
        self.model = pyscipopt.Model()
        self.vars = [
            pyscipopt.Variable("x0", obj=1.0, lpsol=0.25, lp_pos=0),
            pyscipopt.Variable("x1", obj=2.0, lpsol=0.75, lp_pos=1),
            pyscipopt.Variable("x2", obj=3.0, lpsol=0.50, lp_pos=2),
        ]
        sols = [0.25, 0.75, 0.5]
        fracs = [0.1, 0.2, 0.3]
        self.model.set_lp_branch_cands(self.vars, sols=sols, fracs=fracs)

    def test_component_list_forwards_calls(self):
        components = [DummyComponent(), DummyComponent()]
        component_list = ComponentList(components)

        component_list.reset(self.model)
        component_list.done(self.model)

        self.assertEqual([c.reset_calls for c in components], [1, 1])
        self.assertEqual([c.done_calls for c in components], [1, 1])

    def test_scoring_branching_strategy_initialises_scores(self):
        strategy = IdentityStrategy()
        scores = strategy.compute_scores(self.model)
        self.assertEqual(scores.shape, (3,))
        self.assertTrue(np.all(scores == -self.model.infinity()))

    def test_scoring_branching_strategy_performs_branching(self):
        strategy = IdentityStrategy()
        result = strategy.callback(self.model, passive=False)
        self.assertEqual(result, SCIP_RESULT.BRANCHED)
        self.assertEqual(len(self.model.branch_calls), 1)

    def test_pseudocosts_uses_model_scores(self):
        strategy = Pseudocosts()
        for index, variable in enumerate(self.vars):
            self.model.set_pseudocost(variable, score=10 - index)

        result = strategy.callback(self.model, passive=False)
        self.assertEqual(result, SCIP_RESULT.BRANCHED)
        branched_var, _ = self.model.branch_calls[-1]
        self.assertIs(branched_var, self.vars[0])

    def test_strong_branching_computes_scores(self):
        strategy = StrongBranching()
        self.model.setLPObjVal(5.0)
        for variable in self.vars:
            data = (
                12.0,
                14.0,
                True,
                True,
                False,
                False,
                False,
                False,
                False,
            )
            self.model.set_strongbranch_data(variable, data)

        result = strategy.callback(self.model, passive=True)
        self.assertEqual(result, SCIP_RESULT.DIDNOTRUN)
        scores = strategy.scores
        self.assertTrue(np.all(scores == 16.0))
        self.assertFalse(self.model.strongbranch_started)

    def test_accuracy_branching_tracks_positions(self):
        class FixedStrategy(ScoringBranchingStrategy):
            def __init__(self, scores):
                super().__init__()
                self.fixed_scores = np.array(scores, dtype=float)

            def compute_scores(self, model):
                return self.fixed_scores.copy()

        oracle = FixedStrategy([0.9, 0.3, 0.1])
        imitation = FixedStrategy([0.2, 0.8, 0.1])
        strategy = AccuracyBranching(oracle, imitation)

        strategy.callback(self.model, passive=True)
        positions = strategy.get_observations()
        self.assertTrue(np.array_equal(positions, np.array([2])))
        self.assertAlmostEqual(strategy.get_accuracy(2), 1.0)

    def test_conditional_component_selects_first_true_condition(self):
        coalesced = ConditionalBranchingComponent(
            (TrackingBranchingComponent(SCIP_RESULT.DIDNOTRUN, "first"), lambda _: False),
            (TrackingBranchingComponent(SCIP_RESULT.BRANCHED, "second"), lambda _: True),
        )

        result = coalesced.callback(self.model, passive=False)
        self.assertEqual(result, SCIP_RESULT.BRANCHED)
        self.assertEqual(coalesced.get_last_observer_index_used(), 1)

    def test_branchrule_marks_following_components_passive(self):
        components = [
            TrackingBranchingComponent(SCIP_RESULT.BRANCHED, "first"),
            TrackingBranchingComponent(SCIP_RESULT.DIDNOTRUN, "second"),
        ]
        branchrule = BoundmlBranchrule(self.model, components)
        res = branchrule.branchexeclp(False)
        self.assertEqual(res["result"], SCIP_RESULT.BRANCHED)
        self.assertEqual(components[0].calls, [False])
        self.assertEqual(components[1].calls, [True])


if __name__ == "__main__":
    unittest.main()
