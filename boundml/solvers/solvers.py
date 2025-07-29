import os
import time
from typing import Callable, Any

import ecole
import pyscipopt
from ecole.core import DefaultType, Default

import boundml.utils as utils
from boundml.core.observer import Observer


class Solver:
    """
    A Solver is a wrapper around a SCIP solver.
    It can solve different a MIP and collect different metrics from this solving process.
    """

    def __init__(self, config_function: Callable[[pyscipopt.Model], None] = None):
        """
        Parameters
        ----------
        config_function : Callable[[pyscipopt.Model], None]
            Callback function that is called before starting each solving process.
            Can be useful to finely configure the solver.
        """
        self.model: pyscipopt.Model = None
        self.config_function = config_function

    def solve(self, path: str) -> (int, float):
        """
        Solve the
        Parameters
        ----------
        path : str
            Path to the file to solve.

        Returns
        -------
        Tuple[int, float] that is the number of nodes and the time used to solve the instance
        """
        raise NotImplementedError

    def configure(self):
        if self.config_function is not None:
            self.config_function(self.model)

    def set_params(self, params):
        raise NotImplementedError

    def __getitem__(self, item: str) -> float:
        """
        Get an attribute from the solver after a solve.

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


class EcoleSolver(Solver):
    """
    EcoleSolver is a Solver. The branching decisions are taken by asking an Observer the scores of the candidates
    variables. It branches on the one with the biggest score.
    """

    def __init__(self, score_observer: Observer = None, scip_params: dict = {}, additional_observers: [Observer] = [],
                 before_start_callbacks: [Callable[[str], None]] = [],
                 before_action_callbacks: [Callable[[int, [int], Any], None]] = [], *args, **kwargs):
        """
        Parameters
        ----------
        score_observer : Observer
            Observer that scores the candidates variables.
            If no score_observer is provided, a default Observer is used. It forces the solver to not take any decisions
            and delegate the branching to SCIP. This can be useful to only make observations during the default behavior
            of SCIP using additional_observers.
        scip_params : dict
            Dictionary of parameters to pass to the SCIP solver.
        additional_observers : [Observer]
            List of additional observers to perform any desired actions at each decision process.
            Can be used to collect data for exemple.
        before_start_callbacks : [Callable[[str], None]]
            List of Callable that are called before the start of the solver with the name of the file to solve in
            argument.
        before_action_callbacks : [Callable[[int, [int], Any], None]]
            List of Callable that are called before each branching decision. It takes as argument, the action that will
            be performed, the list of possible action and the list of observations of each observer (score observer and
            additional_observers).
        args :
            Additional arguments to pass to the parent Class Solver.
        kwargs :
            Additional arguments to pass to the parent Class Solver.
        """
        super().__init__(*args, **kwargs)
        if score_observer is None:
            score_observer = Observer()

        self.observer_index = 0
        self.observers = [score_observer, *additional_observers]
        self.env: ecole.environment = ecole.environment.Branching(
            observation_function=(
                score_observer,
                *additional_observers
            ),
            information_function={
                "nb_nodes": ecole.reward.NNodes(),
                "time": ecole.reward.SolvingTime(),
            },
            scip_params=scip_params,
        )

        self.before_start_callbacks = before_start_callbacks
        self.before_action_callbacks = before_action_callbacks

        self.state = (
            [score_observer, scip_params, additional_observers, before_start_callbacks, before_action_callbacks, *args],
            kwargs
        )

    def set_before_action_callbacks(self, c):
        self.before_action_callbacks = c

    def set_before_start_callbacks(self, c):
        self.before_start_callbacks = c

    def solve(self, path: str) -> (int, float):
        """
        Start the solving process.
        At each step (branching decisions), perform the action based on the score_observer.

        Parameters
        ----------
        path : str
            Path to the file to solve.

        Returns
        -------
        Tuple[int, float] that is the number of nodes and the time used to solve the instance
        """
        pid = os.getpid()
        m = pyscipopt.Model()
        m.setParam("display/verblevel", 0)
        m.readProblem(path)
        instance: ecole.scip.Model = ecole.scip.Model.from_pyscipopt(m)
        for observer in self.observers:
            if isinstance(observer, Observer):
                observer.reset(path)
        self.before_start(path)
        self.env.seed(0)

        t = time.time()
        observations, action_set, _, done, info = self.env.reset(instance)
        self.model = instance.as_pyscipopt()
        self.configure()

        nb_nodes = info["nb_nodes"]
        while not done:
            # Ugly and bad, but do the work for the moment when dataset generation is used with fork version of LB
            new_pid = os.getpid()
            if pid != new_pid:
                pid = new_pid
                self.before_action_callbacks = []
                # self.env.observation_function = self.observer.observers[0]

            for i, observer in enumerate(self.observers):
                if isinstance(observer, Observer) and observer.is_principal_observer():
                    self.observer_index = i

            action = self.get_action(action_set, observations)
            self.before_action(action, action_set, observations)

            observations, action_set, _, done, info = self.env.step(action)
            self.model = self.env.model.as_pyscipopt()
            nb_nodes += info["nb_nodes"]
            if not done:
                done = self.after_action()

        t = time.time() - t

        self.model = self.env.model.as_pyscipopt()

        self.observers[self.observer_index].done(self.model)

        return nb_nodes, t

    def before_start(self, path):
        return [f(path) for f in self.before_start_callbacks]

    def before_action(self, action, action_set, observation):
        return [f(action, action_set, observation) for f in self.before_action_callbacks]

    def after_action(self):
        return False

    def get_action(self, action_set, observation) -> int | DefaultType:
        scores = observation[self.observer_index]

        # If no scores is given, use the default action (let scip branch how it wants)
        if scores is None:
            return Default

        action_index = scores[action_set].argmax()
        # action_index = np.random.choice(np.flatnonzero(scores[action_set] == scores[action_set].max())) # chose at random an index with max value
        action = action_set[action_index]
        return action

    def set_params(self, params: dict) -> None:
        """
        Set the parameters of the solver.
        Parameters
        ----------
        params : dict
            Parameters to set that respect SCIP parameters
        """
        self.env.scip_params = params
        self.state[0][1] = params

    def __str__(self):
        return str(self.observers[0])

    def __getstate__(self):
        return self.state

    def __setstate__(self, state):
        self.__init__(*state[0], **state[1])


class ClassicSolver(Solver):
    """
    ClassicSolver is a Solver that is a wrapper arround a SCIP solver. It solves the instances using default SCIP.
    """
    def __init__(self, branching_policy: str, scip_params: dict={}, *args, **kwargs):
        """
        Parameters
        ----------
        branching_policy : str
            Name of the branching policy to use. It must be a valid SCIP policy.
        scip_params : dict
            Parameters to pass to the SCIP solver.
        args :
        kwargs :
        """
        super().__init__(*args, **kwargs)
        self.model = pyscipopt.Model()
        self.branching_policy = branching_policy
        self.configure()
        self.model.setParams(scip_params)

        self.state = (
            [branching_policy, scip_params, *args],
            kwargs
        )

    def configure(self):
        super().configure()
        utils.configure(self.model, branching_policy=self.branching_policy)

    def solve(self, path: str):
        self.model.readProblem(path)
        self.model.optimize()

    def set_params(self, params):
        self.model.setParams(params)

    def __str__(self):
        return self.branching_policy

    def __getstate__(self):
        return self.state

    def __setstate__(self, state):
        self.__init__(*state[0], **state[1])
