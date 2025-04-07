import gzip
import pickle
import tempfile
from pathlib import Path
import numpy as np
import ecole.observation

from boundml.observers import ConditionalObservers
from boundml.solvers import EcoleSolver


class DatasetGenerator:
    def __init__(self, instances, expert_observer=ecole.observation.StrongBranchingScores, exploration_observer=ecole.observation.Pseudocosts(), expert_probability=0.1, state_observer=ecole.observation.NodeBipartite(), seed=None, sample_counter=0, episode_counter=0, score_observer=None, **kwargs):
        self.rng = np.random.default_rng(seed)
        self.expert_observer = expert_observer

        self.observer = ConditionalObservers(*list(zip(*[
            (
                expert_observer,
                lambda _: self.rng.random() < expert_probability
            ),
            (
                exploration_observer,
                lambda _: True
            )
        ])))

        self.copy_branching_observer = score_observer is None
        if self.copy_branching_observer:
            obs = [state_observer]
        else:
            obs = [state_observer, score_observer]

        self.solver = EcoleSolver(self.observer, additional_observers=obs, **kwargs)
        self.instances = instances
        self.sample_counter = sample_counter
        self.episode_counter = episode_counter

        for _ in range(self.episode_counter):
            next(self.instances)

    def generate(self, folder_name, max_samples=-1, max_instances=-1, sample_prefix=""):
        Path(folder_name).mkdir(parents=True, exist_ok=True)
        assert max_samples > 0 or max_instances > 0, "One of these parameters must be > 0"

        def before_action(action, action_set, observation):
            scores, node_observation, *_ = observation

            if self.copy_branching_observer:
                save_scores = scores
            else:
                save_scores = observation[2]

            scores_are_expert = self.observer.get_last_observer_index_used() == 0

            if scores_are_expert and (max_samples<0 or self.sample_counter < max_samples):
                self.sample_counter += 1
                data = [node_observation, action, action_set, save_scores]
                filename = f"{folder_name}/sample{sample_prefix}_{self.sample_counter}.pkl"

                with gzip.open(filename, "wb") as f:
                    pickle.dump(data, f)

        self.solver.set_before_action_callbacks([before_action])

        prob_file = tempfile.NamedTemporaryFile(suffix=".lp")

        count=0
        while ( max_samples<0 or self.sample_counter < max_samples) and (max_instances<0 or count<max_instances):
            instance = next(self.instances)
            self.episode_counter += 1
            count += 1
            m = instance.as_pyscipopt()
            m.setParam("display/verblevel", 0)
            m.writeProblem(prob_file.name, verbose=False)
            self.solver.solve(prob_file.name)

            print(f"Episode {self.episode_counter}, {self.sample_counter} samples collected so far")

        prob_file.close()
