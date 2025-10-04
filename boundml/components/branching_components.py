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
        self.scores = None

    @abstractmethod
    def compute_scores(self, model: Model) -> np.ndarray:
        """
        Compute scores for each branching candidate.
        scores[i] must contain the score for i th candidate
        If a score is np.nan, the underlying strategy consider that it can cutoff the node

        Parameters
        ----------
        model : Model
            State of the model

        Returns
        ----------
        np.ndarray with a size of the number of branching candidates
        """
        candidates, *_ = model.getLPBranchCands()
        scores = np.zeros(len(candidates))
        scores[:] = -model.infinity()
        return scores


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
        candidates, candidates_sols, *_ = model.getLPBranchCands()
        self.scores = self.compute_scores(model)

        if passive:
            return SCIP_RESULT.DIDNOTRUN
        elif np.nan in self.scores:
            return SCIP_RESULT.CUTOFF
        else:
            index = np.argmax(self.scores)
            model.branchVarVal(candidates[index], candidates_sols[index])

            return SCIP_RESULT.BRANCHED

    def get_last_scores(self):
        """
        Get the last score for each branching candidate used on the last callback
        Returns
        -------
        np.ndarray with a size of the number of branching candidates
        """
        return self.scores



class Pseudocosts(ScoringBranchingStrategy):
    def compute_scores(self, model: Model) -> np.ndarray:
        """
        Compute pseudocosts scores for each candidate.
        Parameters
        ----------
        model : Model
        """
        scores = super().compute_scores(model)
        candidates, *_ = model.getLPBranchCands()

        var: pyscipopt.Variable
        for i, var in enumerate(candidates):
            lpsol = var.getLPSol()
            score = model.getVarPseudocostScore(var, lpsol)

            scores[i] = score

        return scores

    def __str__(self):
        return "Pseudocosts"

class StrongBranching(ScoringBranchingStrategy):
    """
    Simple implementation of Strong Branching.
    """
    def __init__(self, priocands: bool = False, all_scores: bool = True, allow_cutoff: bool = False, idempotent: bool = True):
        """
        Parameters
        ----------
        priocands : bool
            Whether the scoring is only done on priocands
        all_scores : bool
            Whether all the candidates are scored. If True, the scoring is done when it is possible to cut the node
        allow_cutoff : bool
            Whether the cutoff is allowed.
        idempotent: bool
            Whether getVarStrongbranch calls are idempotent.
        """
        super().__init__()
        self.priocands = priocands
        self.all_scores = all_scores
        self.allow_cutoff = allow_cutoff
        self.idempotent = idempotent

    def compute_scores(self, model: Model) -> np.ndarray:
        scores = super().compute_scores(model)

        branch_cands, branch_cand_sols, branch_cand_fracs, ncands, npriocands, nimplcands = model.getLPBranchCands()

        n = npriocands if self.priocands else ncands

        lpobjval = model.getLPObjVal()

        # Start strong branching and iterate over the branching candidates
        model.startStrongbranch()
        for i in range(n):
            # Strong branch!
            down, up, downvalid, upvalid, downinf, upinf, downconflict, upconflict, lperror = model.getVarStrongbranch(
                branch_cands[i], 2147483647, idempotent=self.idempotent)

            down = max(down, lpobjval)
            up = max(up, lpobjval)
            downgain = down - lpobjval
            upgain = up - lpobjval

            scores[i] = model.getBranchScoreMultiple(branch_cands[i], [downgain, upgain])

            # In the case of both infeasible sub-problems cutoff the node
            if not self.all_scores and self.allow_cutoff and downinf and upinf:
                scores[i] = np.nan
                continue

        model.endStrongbranch()
        return scores

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

    def compute_scores(self, model: Model) -> np.ndarray:
        scores = super().compute_scores(model)

        oracle_scores = self.oracle_strategy.compute_scores(model)
        model_scores = self.model_strategy.compute_scores(model)

        scores = oracle_scores

        best_index = np.argmax(model_scores)
        oracle_sorted_indexes = np.argsort(-oracle_scores)

        position = np.where(oracle_sorted_indexes == best_index)[0][0] + 1
        self.observation.append(position)

        return scores

    def done(self, model: Model) -> None:
        self.oracle_strategy.done(model)
        self.model_strategy.done(model)

        super().done(model)

    def get_observations(self):
        return np.array(self.observation)

    def __str__(self):
        return f"Acc {str(self.model_strategy)}"