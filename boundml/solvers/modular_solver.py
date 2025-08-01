from typing import Callable, Iterable

from pyscipopt import Model

from .solvers import ScipSolver
from ..components import Component, BranchingComponent
from ..components.components import ComponentList
from ..core.branchrule import BoundmlBranchrule


class ModularSolver(ScipSolver):
    """
    A ModularSolver is  ScipSolver that is parametrized with a list of Component
    """

    def __init__(self, *components: Component, scip_params = None, configure: Callable[[Model], None] = None):
        """
        Parameters
        ----------
        components : [Component]
             List of components that parametrized the solver. During a solving process, they are called depending on
             their subtypes (e.g. BranchingComponent are called before making a branching strategy). If several
             components have the same subtypes, they are called in the order they are given. Only the first component
             of each subtype is allowed to perform an action (with passive=False). The other must remain passive.
        scip_params : dict, optional
            Dictionary of parameters to pass to the scip solver.
        configure :  Callable[[Model], None], optional
            Callback function to configure the solver (e.g. add branching strategies, ...)
        """
        super().__init__(scip_params, configure)

        self.components = ComponentList(list(components))
        self.branching_components = []

        for component in components:
            if isinstance(component, BranchingComponent):
                self.branching_components.append(component)



    def build_model(self):
        super().build_model()
        branchrule = BoundmlBranchrule(self.model, self.branching_components)
        self.model.includeBranchrule(
            branchrule,
            "boundml",
            "Custom branching rule for ModularSolver",
            priority=10000000,
            maxdepth=-1,
            maxbounddist=1
        )

    def solve(self, instance: str):
        self.build_model()

        self.model.readProblem(instance)

        self.components.reset(self.model)

        self.model.optimize()

        self.components.done(self.model)

    def __getstate__(self):
        return (self.components, self.scip_params, self.configure)

    def __setstate__(self, state):
        self.__init__(*state[0], scip_params=state[1], configure=state[2])

    def __str__(self):
        return "+".join([str(c) for c in self.components])

