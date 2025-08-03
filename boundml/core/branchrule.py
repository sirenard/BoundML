from typing import Iterable

from pyscipopt import Branchrule, Model, SCIP_RESULT

from boundml.components import BranchingComponent


class BoundmlBranchrule(Branchrule):
    """
    Branchrule used by ModularSolver.
    It consists into calling the corresponding BranchingComponent
    """
    def __init__(self, model: Model, components: Iterable[BranchingComponent]):
        self.model = model
        self.components = components

    def branchexeclp(self, allowaddcons):
        result = SCIP_RESULT.DIDNOTRUN
        passive = False
        for component in self.components:
            r = component.callback(self.model, passive)
            if r is not None and r != SCIP_RESULT.DIDNOTRUN:
                result = r
                passive = True

        return {"result": result}
