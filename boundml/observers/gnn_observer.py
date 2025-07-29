import ecole
import numpy as np
import torch

from boundml.ml.model import load_policy, get_device
from boundml.core.observer import Observer


class GnnObserver(Observer):
    """
    Observer that uses a Graph Neural Network to compute the scores.
    """
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
