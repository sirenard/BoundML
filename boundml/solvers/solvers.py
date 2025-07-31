from abc import ABC, abstractmethod
from typing import Callable

from pyscipopt import Model

class Solver(ABC):
    @abstractmethod
    def solve(self, instance: str):
        raise NotImplementedError("Subclasses must implement this method.")

    @abstractmethod
    def __getitem__(self, item: str) -> float:
        """
        Get an attribute from the solver after a solve.

        Parameters
        ----------
        item : str
            Name of the attribute to get. Depends on the subtype of the solver.

        Returns
        -------
        Value of the attribute.
        """
        raise NotImplementedError("Subclasses must implement this method.")

class ScipSolver(Solver):
    def __init__(self, scip_params: dict = None, configure: Callable[[Model], None] = None):
        """
        Parameters
        ----------
        scip_params : dict, optional
            Dictionary of parameters to pass to the scip solver.
        configure :  Callable[[Model], None], optional
            Callback function to configure the solver (e.g. add branching strategies, ...)
        """
        self.scip_params = scip_params or {}
        self.configure = configure
        self.model = Model()
        self.set_params(self.scip_params)
        if configure is not None:
            configure(self.model)
        self.model.setParam("display/verblevel", 0)

    def set_params(self, params):
        self.scip_params = params
        self.model.setParams(self.scip_params)

    def __getitem__(self, item: str) -> float:
        """
        Get an attribute from the solver after a solving process.

        Parameters
        ----------
        item : str
            Name of the attribute to get. Either:
                time: time in seconds used by the solver
                nnodes: number of nodes used by the solver
                obj: Final objective value
                sol: Best solution found
                estimate_nnodes: Estimated number of nodes used by the solver to go to optimality

        Returns
        -------
        Value of the attribute.
        """
        match item:
            case "time":
                return self.model.getSolvingTime()
            case "nnodes":
                # return self.model.getNNodes()
                return self.model.getNTotalNodes()
            case "obj":
                return self.model.getObjVal()
            case "gap":
                return self.model.getGap()
            case "sol":
                return self.model.getBestSol()
            case "estimate_nnodes":
                if self.model.getGap() == 0:
                    return self["nnodes"]
                else:
                    return self.model.getTreesizeEstimation()
            case _:
                raise KeyError

    def __getstate__(self):
        return self.scip_params, self.configure

    def __setstate__(self, state):
        self.__init__(*state)

class DefaultScipSolver(ScipSolver):
    """
    Default scip solver.
    Solve the instances based on the given scip parameters.
    """
    def __init__(self, branching_strategy, *args, **kwargs):
        """
        Parameters
        ----------
        branching_strategy : str
            Branching strategy to use. Must be a default SCIP strategy, or a strategy included in the Model using
            the configure callback
        args :
            Arguments to build the parent class ScipSolver
        kwargs :
            Arguments to build the parent class ScipSolver
        """
        super().__init__(*args, **kwargs)
        self.model.setParam(f"branching/{branching_strategy}/priority", 9999999)
        self.branching_strategy = branching_strategy
        self.state = (
            [branching_strategy, *args],
            kwargs
        )

    def solve(self, instance: str):
        self.model.readProblem(instance)
        self.model.optimize()

    def __getstate__(self):
        return self.state

    def __setstate__(self, state):
        self.__init__(*state[0], **state[1])

    def __str__(self):
        return self.branching_strategy