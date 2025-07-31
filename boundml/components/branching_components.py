from abc import abstractmethod

import numpy as np
import pyscipopt
from pyscipopt import Model, SCIP_RESULT

from boundml.components import Component


class BranchingComponent(Component):
    """
    A BranchingComponent is a Component called before each branching decision.
    The callback method is called when a branching decision is required
    """

    @abstractmethod
    def callback(self, model: Model, passive: bool=True) -> SCIP_RESULT:
        """
        Callback method called by the solver when a branching decision is required

        Parameters
        ----------
        model : Model
            State of the model
        passive : bool
            Whether the component is allowed to perform a branching or not
        Returns
        -------
        SCIP_RESULT among: SCIP_RESULT.BRANCHED, SCIP_RESULT.DIDNOTRUN, SCIP_RESULT.CUTOFF
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
        Parameters
        ----------
        model : Model
        passive : bool

        Returns
        -------
        SCIP_RESULT.BRANCHED if passive==False, SCIP_RESULT.DIDNOTRUN otherwise
        """
        candidates, *_ = model.getLPBranchCands()
        self.scores = np.zeros(len(candidates))
        self.compute_scores(model)

        if passive:
            return SCIP_RESULT.DIDNOTRUN
        else:
            index = np.argmax(self.scores)
            best_cand = candidates[index]
            model.branchVar(best_cand)

            return SCIP_RESULT.BRANCHED



class Pseudocosts(ScoringBranchingStrategy):
    def __init__(self):
        super().__init__()

        # Dict that stores variables' scores of the last decisions
        self.last_scores = {}

    def reset(self, model: Model) -> None:
        self.last_scores = {}

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

