import os
import time

import ecole
import numpy as np
import pyscipopt

import boundml.utils as utils
from boundml.environments import HistoryBranchingEnvironment
from boundml.observers import Observer


class Solver:
    def __init__(self, config_function=None):
        self.model: pyscipopt.Model = None
        self.config_function = config_function

    def solve(self, path: str):
        raise NotImplementedError

    def configure(self):
        if self.config_function is not None:
            self.config_function(self.model)

    def set_params(self, params):
        raise NotImplementedError

    def __getitem__(self, item):
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

    def __init__(self, score_observer: Observer, scip_params={}, additional_observers=[], before_start_callbacks=[],
                 before_action_callbacks=[], *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.observer_index = 0
        self.observers = [score_observer, *additional_observers]
        self.env: ecole.environment = HistoryBranchingEnvironment(
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

    def solve(self, path: str):
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

    def get_action(self, action_set, observation) -> int:
        scores = observation[self.observer_index]
        action_index = scores[action_set].argmax()
        # action_index = np.random.choice(np.flatnonzero(scores[action_set] == scores[action_set].max())) # chose at random an index with max value
        action = action_set[action_index]
        return action

    def set_params(self, params):
        self.env.scip_params = params
        self.state[0][1] = params

    def __str__(self):
        return str(self.observers[0])

    def __getstate__(self):
        return self.state

    def __setstate__(self, state):
        self.__init__(*state[0], **state[1])


class ClassicSolver(Solver):
    def __init__(self, branching_policy, scip_params={}, *args, **kwargs):
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
