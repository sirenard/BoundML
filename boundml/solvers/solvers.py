import tempfile
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
        self.model = None

    def build_model(self):
        self.model = Model()
        self.set_params(self.scip_params)
        if self.configure is not None:
            self.configure(self.model)
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

    def solve_model(self, model: Model):
        """
        A scip solver can solve directly a pyscipopt model.
        Parameters
        ----------
        model : Model
            model to solve.
        """
        self.build_model() # Must build a new model each time due to some ecole dependencies
        model.setParam("display/verblevel", 0)
        prob_file = tempfile.NamedTemporaryFile(suffix=".lp")
        model.writeProblem(prob_file.name, verbose=False)

        self.solve(prob_file.name)
        prob_file.close()


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
        self.branching_strategy = branching_strategy
        self.state = (
            [branching_strategy, *args],
            kwargs
        )

    def build_model(self):
        super().build_model()
        self.model.setParam(f"branching/{self.branching_strategy}/priority", 9999999)


    def solve(self, instance: str):
        self.build_model()
        self.model.readProblem(instance)
        self.model.optimize()

    def __getstate__(self):
        return self.state

    def __setstate__(self, state):
        self.__init__(*state[0], **state[1])

    def __str__(self):
        return self.branching_strategy