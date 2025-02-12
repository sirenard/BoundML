import ecole
import pyscipopt
from pyscipopt.scip import PY_SCIP_PARAMSETTING

import boundml.utils as utils


class SimpleBranchingDynamics(ecole.dynamics.BranchingDynamics):
    def reset_dynamics(self, model, solution_path=None):
        # keeping a reference to the model
        # Fix huge problems
        self.pyscipopt_model: pyscipopt.Model = model.as_pyscipopt()

        utils.configure(self.pyscipopt_model)

        if solution_path is not None:
            sol = self.pyscipopt_model.readSolFile(solution_path)
            self.pyscipopt_model.addSol(sol)

            self.pyscipopt_model.setPresolve(PY_SCIP_PARAMSETTING.OFF)
            self.pyscipopt_model.setSeparating(PY_SCIP_PARAMSETTING.OFF)
            self.pyscipopt_model.setHeuristics(PY_SCIP_PARAMSETTING.OFF)

        # h = CountingHandler(self.pyscipopt_model)
        # self.pyscipopt_model.includeEventhdlr(h, "Counter", "python event handler to count solutions")
        # self.handler = h

        # Let the parent class get the model to the root node and return
        # the done flag / action_set
        return super().reset_dynamics(model)


class HistoryBranchingEnvironment(ecole.environment.Environment):
    __Dynamics__ = SimpleBranchingDynamics
    #__DefaultObservationFunction__ = ecole.observation.NodeBipartite



class SimpleConfiguringDynamics(ecole.dynamics.ConfiguringDynamics):
    def reset_dynamics(self, model):
        # Share memory with Ecole model
        self.pyscipopt_model: pyscipopt.Model = model.as_pyscipopt()

        utils.configure(self.pyscipopt_model)

        # Let the parent class get the model to the root node and return
        # the done flag / action_set
        return super().reset_dynamics(model)


class SimpleBranching(ecole.environment.Environment):
    __Dynamics__ = SimpleBranchingDynamics

    def reset(self, *dynamics_args, **dynamics_kwargs):
        res = super().reset(*dynamics_args, **dynamics_kwargs)
        done = res[3]
        if done:
            model = self.model.as_pyscipopt()
            self.observation_function.finish(model)

        return res

    def step(self, *dynamics_args, **dynamics_kwargs):
        res = super().step(*dynamics_args, **dynamics_kwargs)
        done = res[3]
        if done:
            model = self.model.as_pyscipopt()
            self.observation_function.finish(model)

        return res

class SimpleConfiguring(ecole.environment.Configuring):
    __Dynamics__ = SimpleConfiguringDynamics