import gzip
import pickle
from pathlib import Path

import numpy as np
from pyscipopt import Model

from boundml.components import ScoringBranchingStrategy, Pseudocosts, Component, BranchingComponent, \
    ConditionalBranchingComponent
from boundml.instances import Instances
from boundml.solvers import ModularSolver


class BranchingDatasetGenerator:
    """
    Class to generate a dataset from branching decisions. Useful to train later a model to imitate a strategy.
    """

    def __init__(self, instances: Instances, expert_strategy: ScoringBranchingStrategy, state_component_observer: Component,
                 exploration_strategy: ScoringBranchingStrategy = Pseudocosts(), expert_probability: float = 0.1,
                 seed=None, sample_counter: int = 0, episode_counter: int = 0, **kwargs):
        self.rng = np.random.default_rng(seed)

        strategy = ConditionalBranchingComponent(
            (expert_strategy, lambda _: self.rng.random() < expert_probability), (exploration_strategy, lambda _: True))

        self.storer = DatasetStorer(expert_strategy, strategy, state_component_observer, sample_counter)

        self.solver = ModularSolver(self.storer, **kwargs)

        self.instances = instances
        self.episode_counter = episode_counter

        # Skip the first episode_counter instances if not 0
        for _ in range(self.episode_counter):
            next(self.instances)

    def generate(self, folder_name: str, max_samples: int = -1, max_instances: int = -1, sample_prefix: str = ""):
        """
        Generate the dataset
        Parameters
        ----------
        folder_name : str
            Folder name to store the samples
        max_samples : int
            Maximum number of samples to generate
        max_instances : int
            Maximum number of instances used for the generation
        sample_prefix : str
            Prefix name of the generated samples. Useful if generations done in parallel
        """
        Path(folder_name).mkdir(parents=True, exist_ok=True)
        assert max_samples > 0 or max_instances > 0, "One of these parameters must be > 0"
        self.storer.setup(folder_name, max_samples, sample_prefix)
        sample_counter = self.storer.sample_counter
        count = 0
        while (max_samples < 0 or sample_counter < max_samples) and (max_instances < 0 or count < max_instances):
            instance = next(self.instances)
            self.episode_counter += 1
            count += 1
            self.solver.solve_model(instance)

            sample_counter = self.storer.sample_counter
            print(f"Episode {self.episode_counter}, {sample_counter} samples collected so far")


class DatasetStorer(BranchingComponent):
    def __init__(self, expert_strategy: ScoringBranchingStrategy, conditional_strategy: ConditionalBranchingComponent,
                 state_component_observer: Component, sample_counter):
        super().__init__()

        self.expert_strategy = expert_strategy
        self.conditional_strategy = conditional_strategy
        self.state_component_observer = state_component_observer
        self.folder_name = ""
        self.max_samples = -1
        self.sample_prefix = ""
        self.sample_counter = sample_counter

    def setup(self, folder_name: str, max_samples: int, sample_prefix: str):
        self.folder_name = folder_name
        self.max_samples = max_samples
        self.sample_prefix = sample_prefix

    def reset(self, model: Model) -> None:
        self.conditional_strategy.reset(model)
        self.state_component_observer.reset(model)

    def callback(self, model: Model, passive: bool = True):
        res = self.conditional_strategy.callback(model, passive=passive)
        scores_are_expert = self.conditional_strategy.get_last_observer_index_used() == 0
        if scores_are_expert and (self.max_samples < 0 or self.sample_counter < self.max_samples):
            self.sample_counter += 1

            expert_scores = self.expert_strategy.scores

            candidates, *_ = model.getLPBranchCands()
            action_set = [var.getCol().getLPPos() for var in candidates]
            action = action_set[np.argmax(expert_scores)]

            scores = np.zeros(model.getNVars())
            scores[:] = np.nan
            scores[action_set] = expert_scores

            self.state_component_observer.callback(model, True)
            node_state = self.state_component_observer.observation



            data = [node_state, action, action_set, scores]
            filename = f"{self.folder_name}/sample{self.sample_prefix}_{self.sample_counter}.pkl"

            with gzip.open(filename, "wb") as f:
                pickle.dump(data, f)

        return res

    def done(self, model: Model) -> None:
        self.conditional_strategy.done(model)
        self.state_component_observer.done(model)
