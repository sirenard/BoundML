from typing import Iterable

from pyscipopt import Branchrule, Model, SCIP_RESULT

from boundml.components import BranchingComponent, RootNodeBranchingComponent


class BoundmlBranchrule(Branchrule):
    """
    Branchrule used by ModularSolver.
    It consists into calling the corresponding BranchingComponent
    """
    def __init__(self, model: Model, components: Iterable[BranchingComponent]):
        self.model = model
        self.first_branch = False

        self.branching_components = []
        self.root_node_branching_component = []

        for component in components:
            if isinstance(component, RootNodeBranchingComponent):
                self.root_node_branching_component.append(component)
            else:
                self.branching_components.append(component)

    def branchinit(self, *args, **kwargs):
        self.first_branch = True

    def branchexeclp(self, allowaddcons):
        result = SCIP_RESULT.DIDNOTRUN
        passive = False

        if self.first_branch:
            components = self.root_node_branching_component + self.branching_components
            self.first_branch = False
        else:
            components = self.branching_components

        for component in components:
            r = component.callback(self.model, passive)
            if r is not None and r != SCIP_RESULT.DIDNOTRUN:
                result = r
                passive = True

        return {"result": result}

    def branchexecps(self, allowaddcons):
        # Todo Does not run and let SCIP manage this case. Must find a way for branching components to manage this case
        result = SCIP_RESULT.DIDNOTRUN

        return {"result": result}
