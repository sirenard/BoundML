from abc import abstractmethod

import numpy as np
import pyscipopt
from pyscipopt import Model, SCIP_RESULT

from .components import Component


class BranchingComponent(Component):
    """
    A BranchingComponent is a Component called before each branching decision.
    The callback method is called when a branching decision is required
    """

    @abstractmethod
    def callback(self, model: Model, passive: bool=True) -> SCIP_RESULT:
        """
        Callback method called by the solver when a branching decision is required.
        Is responsible to perform the branching as it wants if passive is False.

        Parameters
        ----------
        model : Model
            State of the model
        passive : bool
            Whether the component is allowed to perform a branching or not
        Returns
        -------
        SCIP_RESULT among: SCIP_RESULT.BRANCHED, SCIP_RESULT.DIDNOTRUN, SCIP_RESULT.CUTOFF
        Or None if the component does not aim to perform any action. For exemple, if it collects data.
        """
        raise NotImplementedError("Subclasses must implement this method.")


class ScoringBranchingStrategy(BranchingComponent):
    """
    A ScoringBranchingStrategy is a BranchingComponent that represents a score based branching strategies.
    When called, it computes score for each candidate and branches on the one with the highest score.
    """
    def __init__(self):
        super().__init__()

        self.scores = np.zeros(0)

    @abstractmethod
    def compute_scores(self, model: Model) -> None:
        """
        Compute and update self.scores for each candidate.
        self.scores[i] must contain the score for i th candidate

        Parameters
        ----------
        model : Model
            State of the model
        """
        raise NotImplementedError("Subclasses must implement this method.")


    def callback(self, model: Model, passive: bool=True) -> SCIP_RESULT:
        """
        When called, update the scores and branches on the variable with the highest score if allowed
        Parameters. If one variable has a score of np.nan, then the node is cutoff
        ----------
        model : Model
        passive : bool

        Returns
        -------
        SCIP_RESULT.BRANCHED if passive==False, SCIP_RESULT.DIDNOTRUN otherwise
        """
        candidates, *_ = model.getLPBranchCands()
        self.scores = np.zeros(len(candidates))
        self.scores[:] = -model.infinity()
        self.compute_scores(model)

        if passive:
            return SCIP_RESULT.DIDNOTRUN
        elif np.nan in self.scores:
            return SCIP_RESULT.CUTOFF
        else:
            index = np.argmax(self.scores)
            best_cand = candidates[index]
            model.branchVar(best_cand)

            return SCIP_RESULT.BRANCHED



class Pseudocosts(ScoringBranchingStrategy):
    def compute_scores(self, model: Model) -> None:
        """
        Compute pseudocosts scores for each candidate.
        Parameters
        ----------
        model : Model
        """
        candidates, *_ = model.getLPBranchCands()

        var: pyscipopt.Variable
        for i, var in enumerate(candidates):
            lpsol = var.getLPSol()
            score = model.getVarPseudocostScore(var, lpsol)

            self.scores[i] = score

    def __str__(self):
        return "Pseudocosts"

class StrongBranching(ScoringBranchingStrategy):
    """
    Simple implementation of Strong Branching.
    """
    def __init__(self, priocands: bool = False, all_scores: bool = True):
        """
        Parameters
        ----------
        priocands : bool
            Whether the scoring is only done on priocands
        all_scores : bool
            Whether all the candidates are scored. If True, the scoring is done when it is possible to cut the node
        """
        self.priocands = priocands
        self.all_scores = all_scores

    def compute_scores(self, model: Model) -> None:
        branch_cands, branch_cand_sols, branch_cand_fracs, ncands, npriocands, nimplcands = model.getLPBranchCands()

        n = npriocands if self.priocands else ncands
        # Initialise scores for each variable
        down_bounds = [None for _ in range(npriocands)]
        up_bounds = [None for _ in range(npriocands)]

        # Initialise placeholder values
        num_nodes = model.getNNodes()
        lpobjval = model.getLPObjVal()

        # Start strong branching and iterate over the branching candidates
        model.startStrongbranch()
        for i in range(n):

            # Check the case that the variable has already been strong branched on at this node.
            # This case occurs when events happen in the node that should be handled immediately.
            # When processing the node again (because the event did not remove it), there's no need to duplicate work.
            if model.getVarStrongbranchNode(branch_cands[i]) == num_nodes:
                down, up, downvalid, upvalid, _, lastlpobjval = model.getVarStrongbranchLast(branch_cands[i])
                if downvalid:
                    down_bounds[i] = down
                if upvalid:
                    up_bounds[i] = up
                downgain = max([down - lastlpobjval, 0])
                upgain = max([up - lastlpobjval, 0])
                self.scores[i] = model.getBranchScoreMultiple(branch_cands[i], [downgain, upgain])
                continue

            # Strong branch!
            down, up, downvalid, upvalid, downinf, upinf, downconflict, upconflict, lperror = model.getVarStrongbranch(
                branch_cands[i], 200, idempotent=False)

            # In the case of an LP error handle appropriately (for this example we just break the loop)
            if lperror:
                break

            # In the case of both infeasible sub-problems cutoff the node
            if downinf and upinf:
                self.scores[i] = np.nan
                continue

            # Calculate the gains for each up and down node that strong branching explored
            if not downinf and downvalid:
                down_bounds[i] = down
                downgain = max([down - lpobjval, 0])
            else:
                downgain = 0
            if not upinf and upvalid:
                up_bounds[i] = up
                upgain = max([up - lpobjval, 0])
            else:
                upgain = 0

            # Update the pseudo-costs
            lpsol = branch_cands[i].getLPSol()
            if not downinf and downvalid:
                model.updateVarPseudocost(branch_cands[i], -model.frac(lpsol), downgain, 1)
            if not upinf and upvalid:
                model.updateVarPseudocost(branch_cands[i], 1 - model.frac(lpsol), upgain, 1)

            self.scores[i] = model.getBranchScoreMultiple(branch_cands[i], [downgain, upgain])

    def __str__(self):
        return "StrongBranching"


class AccuracyBranching(ScoringBranchingStrategy):
    """
    AccuracyBranchingComponent is a component that depends on 2 ScoringBranchingStrategy.
    Generally an oracle, and another one that tries to imitate the oracle.
    It outputs the same results as the oracle, but in addition stores for each branching decision at which position the
    second component would have ranked the oracle's decision
    """

    def __init__(self, oracle: ScoringBranchingStrategy, model: ScoringBranchingStrategy):
        super().__init__()
        self.oracle_strategy = oracle
        self.model_strategy = model
        self.observation = []

    def reset(self, model: Model) -> None:
        super().reset(model)
        self.oracle_strategy.reset(model)
        self.model_strategy.reset(model)

    def callback(self, model: Model, passive: bool=True) -> SCIP_RESULT:
        self.oracle_strategy.callback(model, True)
        self.model_strategy.callback(model, False)

        return super().callback(model, passive)

    def compute_scores(self, model: Model) -> None:
        oracle_scores = self.oracle_strategy.scores
        model_scores = self.model_strategy.scores

        self.scores = oracle_scores

        best_index = np.argmax(model_scores)
        oracle_sorted_indexes = np.argsort(-oracle_scores)

        position = np.where(oracle_sorted_indexes == best_index)[0][0] + 1
        self.observation.append(position)

    def done(self, model: Model) -> None:
        self.oracle_strategy.done(model)
        self.model_strategy.done(model)

        super().done(model)

    def get_observations(self):
        return np.array(self.observation)

    def __str__(self):
        return f"Acc {str(self.model_strategy)}"