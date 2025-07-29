import gzip
import pickle
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torch_geometric
from torch.optim.lr_scheduler import ReduceLROnPlateau


def get_device(try_use_gpu=True):
    return torch.device("cuda" if try_use_gpu and torch.cuda.is_available() else "cpu")


class BipartiteNodeData(torch_geometric.data.Data):
    """
    This class encode a node bipartite graph observation as returned by the `ecole.observation.NodeBipartite`
    observation function in a format understood by the pytorch geometric data handlers.
    """

    def __init__(
            self,
            constraint_features,
            edge_indices,
            edge_features,
            variable_features,
            tree_features,
            candidates,
            nb_candidates,
            candidate_choice,
            candidate_scores,
    ):
        super().__init__()
        self.constraint_features = constraint_features
        self.edge_index = edge_indices
        self.edge_attr = edge_features
        self.variable_features = variable_features
        self.tree_features = tree_features
        self.candidates = candidates
        if variable_features is not None:
            self.n_nodes = variable_features.size()[0]
        else:
            self.n_nodes = 0
        self.nb_candidates = nb_candidates
        self.candidate_choices = candidate_choice
        self.candidate_scores = candidate_scores

    def __inc__(self, key, value, store, *args, **kwargs):
        """
        We overload the pytorch geometric method that tells how to increment indices when concatenating graphs
        for those entries (edge index, candidates) for which this is not obvious.
        """
        if key == "edge_index":
            return torch.tensor(
                [[self.constraint_features.size(0)], [self.variable_features.size(0)]]
            )
        elif key == "candidates":
            return self.variable_features.size(0)
        else:
            return super().__inc__(key, value, *args, **kwargs)


class GraphDataset(torch_geometric.data.Dataset):
    """
    This class encodes a collection of graphs, as well as a method to load such graphs from the disk.
    It can be used in turn by the data loaders provided by pytorch geometric.
    """

    def __init__(self, sample_files=None, sample=[]):
        super().__init__(root=None, transform=None, pre_transform=None)
        self.sample_files = sample_files
        self.sample = sample

    def len(self):
        if self.sample_files is not None:
            return len(self.sample_files)
        else:
            return len(self.sample)

    def get(self, index):
        """
        This method loads a node bipartite graph observation as saved on the disk during data collection.
        """
        if self.sample_files is not None:
            with gzip.open(self.sample_files[index], "rb") as f:
                # try:
                # print(self.sample_files[index])
                sample = pickle.load(f)
                # except:
                #    print(self.sample_files[index])
                #    raise f"Error with file {self.sample_files[index]}"
        else:
            sample = self.sample[index]

        sample_observation, sample_action, sample_action_set, sample_scores = sample

        constraint_features = sample_observation.row_features
        edge_indices = sample_observation.edge_features.indices.astype(np.int32)
        edge_features = np.expand_dims(sample_observation.edge_features.values, axis=-1)
        variable_features = sample_observation.variable_features
        if hasattr(sample_observation, 'tree_features'):
            tree_features = sample_observation.tree_features
        else:
            tree_features = np.array([])

        # We note on which variables we were allowed to branch, the scores as well as the choice
        # taken by strong branching (relative to the candidates)
        candidates = np.array(sample_action_set, dtype=np.int32)
        if type(sample_scores) != float:
            candidate_scores = np.array([sample_scores[j] for j in candidates])
            candidate_choice = np.where(candidates == sample_action)[0][0]

            graph = BipartiteNodeData(
                torch.FloatTensor(constraint_features),
                torch.LongTensor(edge_indices),
                torch.FloatTensor(edge_features),
                torch.FloatTensor(variable_features),
                torch.FloatTensor(tree_features),
                torch.LongTensor(candidates),
                len(candidates),
                torch.LongTensor([candidate_choice]),
                torch.FloatTensor(candidate_scores)
            )
        else:
            graph = BipartiteNodeData(
                torch.FloatTensor(constraint_features),
                torch.LongTensor(edge_indices),
                torch.FloatTensor(edge_features),
                torch.FloatTensor(variable_features),
                torch.FloatTensor(tree_features),
                torch.LongTensor(candidates),
                len(candidates),
                torch.LongTensor([]),
                torch.FloatTensor(np.array([sample_scores]))
            )

        # We must tell pytorch geometric how many nodes there are, for indexing purposes
        graph.num_nodes = constraint_features.shape[0] + variable_features.shape[0]

        return graph


class GNNPolicy(torch.nn.Module):
    def __init__(self, n_tree_features=0):
        super().__init__()
        emb_size = 64
        cons_nfeats = 5
        edge_nfeats = 1
        var_nfeats = 19
        self.n_tree_features = n_tree_features

        # CONSTRAINT EMBEDDING
        self.cons_embedding = torch.nn.Sequential(
            torch.nn.LayerNorm(cons_nfeats),
            torch.nn.Linear(cons_nfeats, emb_size),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_size, emb_size),
            torch.nn.ReLU(),
        )

        # EDGE EMBEDDING
        self.edge_embedding = torch.nn.Sequential(
            torch.nn.LayerNorm(edge_nfeats),
        )

        # VARIABLE EMBEDDING
        self.var_embedding = torch.nn.Sequential(
            torch.nn.LayerNorm(var_nfeats),
            torch.nn.Linear(var_nfeats, emb_size),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_size, emb_size),
            torch.nn.ReLU(),
        )

        self.conv_v_to_c = BipartiteGraphConvolution()
        self.conv_c_to_v = BipartiteGraphConvolution()

        self.output_module = torch.nn.Sequential(
            torch.nn.Linear(emb_size + n_tree_features, emb_size + n_tree_features),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_size + n_tree_features, 1, bias=False),
        )

    def forward(
            self, constraint_features, edge_indices, edge_features, variable_features, n_var, tree_features=np.array([])
    ):
        reversed_edge_indices = torch.stack([edge_indices[1], edge_indices[0]], dim=0)

        # First step: linear embedding layers to a common dimension (64)
        constraint_features = self.cons_embedding(constraint_features)
        edge_features = self.edge_embedding(edge_features)
        variable_features = self.var_embedding(variable_features)

        # Two half convolutions
        constraint_features = self.conv_v_to_c(
            variable_features, reversed_edge_indices, edge_features, constraint_features
        )
        variable_features = self.conv_c_to_v(
            constraint_features, edge_indices, edge_features, variable_features
        )

        # A final MLP on the variable features
        x = variable_features
        if self.n_tree_features != 0:
            n_row = int(tree_features.size()[0] / self.n_tree_features)
            tree_features = tree_features.reshape(n_row, self.n_tree_features)
            col = torch.repeat_interleave(tree_features, n_var, dim=0)
            x = torch.column_stack((x, col))
        output = self.output_module(x).squeeze(-1)
        return output


class BipartiteGraphConvolution(torch_geometric.nn.MessagePassing):
    """
    The bipartite graph convolution is already provided by pytorch geometric and we merely need
    to provide the exact form of the messages being passed.
    """

    def __init__(self):
        super().__init__("add")
        emb_size = 64

        self.feature_module_left = torch.nn.Sequential(
            torch.nn.Linear(emb_size, emb_size)
        )
        self.feature_module_edge = torch.nn.Sequential(
            torch.nn.Linear(1, emb_size, bias=False)
        )
        self.feature_module_right = torch.nn.Sequential(
            torch.nn.Linear(emb_size, emb_size, bias=False)
        )
        self.feature_module_final = torch.nn.Sequential(
            torch.nn.LayerNorm(emb_size),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_size, emb_size),
        )

        self.post_conv_module = torch.nn.Sequential(torch.nn.LayerNorm(emb_size))

        # output_layers
        self.output_module = torch.nn.Sequential(
            torch.nn.Linear(2 * emb_size, emb_size),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_size, emb_size),
        )

    def forward(self, left_features, edge_indices, edge_features, right_features):
        """
        This method sends the messages, computed in the message method.
        """
        output = self.propagate(
            edge_indices,
            size=(left_features.shape[0], right_features.shape[0]),
            node_features=(left_features, right_features),
            edge_features=edge_features,
        )
        return self.output_module(
            torch.cat([self.post_conv_module(output), right_features], dim=-1)
        )

    def message(self, node_features_i, node_features_j, edge_features):
        output = self.feature_module_final(
            self.feature_module_left(node_features_i)
            + self.feature_module_edge(edge_features)
            + self.feature_module_right(node_features_j)
        )
        return output


def process(policy, data_loader, optimizer=None):
    """
    This function will process a whole epoch of training or validation, depending on whether an optimizer is provided.
    """
    mean_loss = 0
    mean_acc = 0

    n_samples_processed = 0
    with torch.set_grad_enabled(optimizer is not None):
        for batch in data_loader:
            batch = batch.to(get_device())
            # Compute the logits (i.e. pre-softmax activations) according to the policy on the concatenated graphs
            logits = policy(
                batch.constraint_features,
                batch.edge_index,
                batch.edge_attr,
                batch.variable_features,
                batch.n_nodes,
                batch.tree_features,
            )
            # Index the results by the candidates, and split and pad them
            logits = pad_tensor(logits[batch.candidates], batch.nb_candidates)
            # Compute the usual cross-entropy classification loss
            loss = F.cross_entropy(logits, batch.candidate_choices)

            if optimizer is not None:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            true_scores = pad_tensor(batch.candidate_scores, batch.nb_candidates)
            true_bestscore = true_scores.max(dim=-1, keepdims=True).values

            predicted_bestindex = logits.max(dim=-1, keepdims=True).indices
            accuracy = (
                (true_scores.gather(-1, predicted_bestindex) == true_bestscore)
                .float()
                .mean()
                .item()
            )

            mean_loss += loss.item() * batch.num_graphs
            mean_acc += accuracy * batch.num_graphs
            n_samples_processed += batch.num_graphs

    mean_loss /= n_samples_processed
    mean_acc /= n_samples_processed
    return mean_loss, mean_acc


def pad_tensor(input_, pad_sizes, pad_value=-1e8):
    """
    This utility function splits a tensor and pads each split to make them all the same size, then stacks them.
    """
    max_pad_size = pad_sizes.max()
    output = input_.split(pad_sizes.cpu().numpy().tolist())
    output = torch.stack(
        [
            F.pad(slice_, (0, max_pad_size - slice_.size(0)), "constant", pad_value)
            for slice_ in output
        ],
        dim=0,
    )
    return output


def train(sample_folder: str, learning_rate: float, n_epochs: int, output="agent.pkl", policy=None,
          dataset=GraphDataset, **kwargs):
    sample_files = get_sample_files(sample_folder)
    train_files = sample_files[: int(0.8 * len(sample_files))]
    valid_files = sample_files[int(0.8 * len(sample_files)):]

    train_data = dataset(train_files)
    train_loader = torch_geometric.loader.DataLoader(train_data, batch_size=32, shuffle=True)
    valid_data = dataset(valid_files)
    valid_loader = torch_geometric.loader.DataLoader(valid_data, batch_size=32, shuffle=False)

    if policy is None:
        policy = GNNPolicy(**kwargs)
    policy = policy.to(get_device())

    # observation = train_data[0].to(DEVICE)
    #
    # logits = policy(
    #     observation.constraint_features,
    #     observation.edge_index,
    #     observation.edge_attr,
    #     observation.variable_features,
    #     observation.n_nodes,
    #     observation.tree_features,
    # )
    # action_distribution = F.softmax(logits[observation.candidates], dim=-1)

    # print(action_distribution)

    optimizer = torch.optim.Adam(policy.parameters(), lr=learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.2, patience=5, threshold=1e-2)
    for epoch in range(n_epochs):
        lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch + 1}, lr {lr}")

        train_loss, train_acc = process(policy, train_loader, optimizer)
        print(f"Train loss: {train_loss:0.3f}, accuracy {train_acc:0.3f}")

        valid_loss, valid_acc = process(policy, valid_loader, None)
        print(f"Valid loss: {valid_loss:0.3f}, accuracy {valid_acc:0.3f}")

        scheduler.step(valid_loss)

        torch.save(policy.state_dict(), output)

    return policy


def load_policy(filename, try_use_gpu=False, **kwargs):
    device = torch.device("cuda" if try_use_gpu and torch.cuda.is_available() else "cpu")
    policy = GNNPolicy(**kwargs).to(device)
    if device.type == "cpu":
        map_location = torch.device('cpu')
    else:
        map_location = None
    policy.load_state_dict(torch.load(filename, map_location=map_location))
    return policy


def get_sample_files(folder: str):
    return [str(path) for path in Path("{}/".format(folder)).glob("sample*.pkl")]
