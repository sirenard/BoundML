import ecole
import numpy as np
import pyscipopt
import torch
from boundml.model import load_policy, get_device


class Observer:
    def __init__(self, seed=None, principal_observer=False):
        self.seed = seed
        self.instance_path = None
        self.principal_observer = principal_observer

    def before_reset(self, model):
        return

    def extract(self, model, done):
        return

    def reset(self, instance_path, seed=None):
        self.instance_path = instance_path
        self.seed = seed

    def done(self, model):
        return

    def set_principal_observer(self, val):
        self.principal_observer = val

    def is_principal_observer(self):
        return self.principal_observer

    def __str__(self):
        return "default"


class RandomObserver(Observer):
    def __init__(self, seed=None):
        super().__init__(seed)
        self.rng = np.random.default_rng()

    def before_reset(self, model):
        return

    def extract(self, model, done):
        m: pyscipopt.Model = model.as_pyscipopt()
        candidates, *_ = m.getLPBranchCands()
        n_vars = int(m.getNVars())

        prob_indexes = sorted([var.getCol().getLPPos() for var in candidates])

        scores = self.rng.random(len(prob_indexes))
        res = np.zeros(n_vars)
        res[:] = np.nan
        res[prob_indexes] = scores
        return res

    def seed(self, seed):
        self.rng = np.random.default_rng(seed)

    def __str__(self):
        return "random"

class StrongBranching(Observer):
    def __init__(self):
        super().__init__()
        self.observer = ecole.observation.StrongBranchingScores()

    def before_reset(self, model):
        self.observer.before_reset(model)

    def extract(self, model, done):
        return self.observer.extract(model, done)

    def __getstate__(self):
        return []

    def __setstate__(self, _):
        self.__init__()

    def __str__(self):
        return "SB"


class PseudoCost(Observer):
    def __init__(self):
        super().__init__()
        self.observer = ecole.observation.Pseudocosts()

    def before_reset(self, model):
        self.observer.before_reset(model)

    def extract(self, model, done):
        return self.observer.extract(model, done)

    def __getstate__(self):
        return []

    def __setstate__(self, _):
        self.__init__()

    def __str__(self):
        return "PC"


class GnnObserver(Observer):
    def __init__(self, policy_path: str, feature_observer=ecole.observation.NodeBipartite(), try_use_gpu=False,
                 **kwargs):
        super().__init__()
        self.policy_path = policy_path
        self.policy = load_policy(policy_path, try_use_gpu, **kwargs)
        self.feature_observer = feature_observer
        self.policy_path = policy_path
        self.try_use_gpu = try_use_gpu
        self.has_tree_features = "n_tree_features" in kwargs

    def before_reset(self, model):
        self.feature_observer.before_reset(model)

    def extract(self, model, done):
        # print(f"{os.getpid()}: Extract")
        device = get_device(self.try_use_gpu)
        observation = self.feature_observer.extract(model, done)
        observation = (
            torch.from_numpy(observation.row_features.astype(np.float32)).to(device),
            torch.from_numpy(observation.edge_features.indices.astype(np.int64)).to(device),
            torch.from_numpy(observation.edge_features.values.astype(np.float32)).view(-1, 1).to(device),
            torch.from_numpy(observation.variable_features.astype(np.float32)).to(device),
            observation.variable_features.shape[0],
            torch.from_numpy(observation.tree_features.astype(np.float32)).to(
                device) if self.has_tree_features else np.array([]),
        )

        logits = self.policy(*observation)
        return logits.cpu().detach().numpy()

    def __getstate__(self):
        return self.policy_path

    def __setstate__(self, state):
        self.__init__(state)

    def __str__(self):
        return f"GNN({self.policy_path})"

class AccuracyObserver(Observer):
    """
    Given 2 observers, this observer extracts the placement of the second observer choice in the scoring of the first one
    It stores all its observations
    """
    def __init__(self, oracle: Observer, model_observer: Observer):
        super().__init__()
        self.oracle: Observer = oracle
        self.model_observer: Observer = model_observer
        self.observations = []

    def before_reset(self, model):
        self.oracle.before_reset(model)
        self.model_observer.before_reset(model)

    def extract(self, model, done):
        m: pyscipopt.Model = model.as_pyscipopt()
        candidates, *_ = m.getLPBranchCands()
        prob_indexes = [var.getCol().getLPPos() for var in candidates]

        oracle_scores = self.oracle.extract(model, done)

        if self.oracle.is_principal_observer():
            self.set_principal_observer(True)
            return oracle_scores

        oracle_scores = oracle_scores[prob_indexes]

        model_scores = self.model_observer.extract(model, done)
        model_scores = model_scores[prob_indexes]

        best_index = np.argmax(model_scores)
        oracle_sorted_indexes = np.argsort(-oracle_scores)

        position = np.where(oracle_sorted_indexes == best_index)[0][0] + 1
        self.observations.append(position)
        return position

    def get_observations(self):
        return np.array(self.observations)

    def done(self, model):
        self.oracle.done(model)
        self.model_observer.done(model)

    def reset(self, instance_path, seed=None):
        super().reset(instance_path, seed)
        self.oracle.reset(instance_path, seed)
        self.model_observer.reset(instance_path, seed)

    def __str__(self):
        return f"Acc {str(self.model_observer)}"

class ConditionalObservers(Observer):
    def __init__(self, observers, conditions):
        super().__init__()
        self.observers = observers
        self.conditions = conditions
        self.last_observer_index_used = -1

    def before_reset(self, model):
        for observer in self.observers:
            if observer is not None:
                observer.before_reset(model)

    def extract(self, model, done):
        m: pyscipopt.Model = model.as_pyscipopt()
        for i, (observer, condition) in enumerate(zip(self.observers, self.conditions)):
            if condition(m):
                self.last_observer_index_used = i
                if observer is None:
                    return
                else:
                    return observer.extract(model, done)

    def reset(self, instance_path, seed=None):
        super().reset(instance_path, seed)
        for observer in self.observers:
            if isinstance(observer, Observer):
                observer.reset(instance_path, seed)

    def get_last_observer_index_used(self):
        return self.last_observer_index_used

    def done(self, model: pyscipopt.Model):
        for observer in self.observers:
            if isinstance(observer, Observer):
                observer.done(model)

    def __str__(self):
        s = ", ".join([str(observer) for observer in self.observers])
        return f"Cond({s})"
