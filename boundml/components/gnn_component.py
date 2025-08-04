import numpy as np
import pyscipopt
import torch
from pyscipopt import Model

from boundml.ml.model import load_policy, get_device

from .branching_components import ScoringBranchingStrategy, BranchingComponent


class GnnBranching(ScoringBranchingStrategy):
    """
    Use a Graph Neural Network to compute the scores.
    """
    def __init__(self, policy_path: str, feature_component: BranchingComponent, try_use_gpu=False, **kwargs):
        """
        Parameters
        ----------
        policy_path : str
            Path to the torch model file to load
        feature_component : BranchingComponent
            Component used to get the input of the model
        try_use_gpu : bool
            Whether to try to use GPU or not
        kwargs :
            Additional arguments passed to load_policy
        """
        super().__init__()
        self.policy_path = policy_path
        self.policy = load_policy(policy_path, try_use_gpu, **kwargs)
        self.feature_component = feature_component
        self.policy_path = policy_path
        self.try_use_gpu = try_use_gpu

        self.has_tree_features = "n_tree_features" in kwargs

    def reset(self, model: Model) -> None:
        super().reset(model)
        self.feature_component.reset(model)

    def compute_scores(self, model: Model) -> None:
        self.feature_component.callback(model, True)
        observation = self.feature_component.observation

        device = get_device(self.try_use_gpu)
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
        scores = logits.cpu().detach().numpy()

        candidates, *_ = model.getLPBranchCands()
        var: pyscipopt.Variable
        for i, var in enumerate(candidates):
            prob_index = var.getCol().getLPPos()
            self.scores[i] = scores[prob_index]


    def done(self, model: Model) -> None:
        super().done(model)
        self.feature_component.done(model)

    def __str__(self):
        return f"GNN({self.policy_path})"