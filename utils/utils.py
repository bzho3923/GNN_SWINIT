import numpy as np
import torch
from utils.data_processing import get_data
from torch_geometric.data import  TemporalData

class MergeLayer(torch.nn.Module):
  def __init__(self, dim1, dim2, dim3, dim4):
    super().__init__()
    self.fc1 = torch.nn.Linear(dim1 + dim2, dim3)
    self.fc2 = torch.nn.Linear(dim3, dim4)
    self.act = torch.nn.ReLU()

    torch.nn.init.xavier_normal_(self.fc1.weight)
    torch.nn.init.xavier_normal_(self.fc2.weight)

  def forward(self, x1, x2):
    x = torch.cat([x1, x2], dim=1)
    h = self.act(self.fc1(x))
    return self.fc2(h)


class MLP(torch.nn.Module):
  def __init__(self, dim, drop=0.3):
    super().__init__()
    self.fc_1 = torch.nn.Linear(dim, 80)
    self.fc_2 = torch.nn.Linear(80, 10)
    self.fc_3 = torch.nn.Linear(10, 1)
    self.act = torch.nn.ReLU()
    self.dropout = torch.nn.Dropout(p=drop, inplace=False)

  def forward(self, x):
    x = self.act(self.fc_1(x))
    x = self.dropout(x)
    x = self.act(self.fc_2(x))
    x = self.dropout(x)
    return self.fc_3(x).squeeze(dim=1)


class EarlyStopMonitor(object):
    def __init__(self, max_round=3, higher_better=True, tolerance=1e-10, outer_max_round=2, num_epoch_per_round=40):
        self.max_round = max_round
        self.num_outer_round = 0
        self.num_round = 0
        self.num_epoch_per_round = num_epoch_per_round
        self.outer_max_round = outer_max_round
        self.epoch_count = 0
        self.best_epoch = [0 for i in range(outer_max_round)]

        self.last_best = [None for i in range(outer_max_round)]
        self.higher_better = higher_better
        self.tolerance = tolerance

    def early_stop_check(self, curr_val):
        if not self.higher_better:
          curr_val *= -1
        if self.last_best[self.num_outer_round] is None:
          self.last_best[self.num_outer_round] = curr_val
        elif (curr_val - self.last_best[self.num_outer_round]) / np.abs(self.last_best[self.num_outer_round]) > self.tolerance:
          self.last_best[self.num_outer_round] = curr_val
          self.num_round = 0
          self.best_epoch[self.num_outer_round] = self.epoch_count
        else:
          self.num_round += 1

        self.epoch_count += 1
        if self.num_round >= self.max_round: #or self.epoch_count % self.num_epoch_per_round == 0:
            self.num_outer_round += 1
            self.num_round = 0
            return True
        else:
            return False
    
    def final_stop_check(self):
        return self.num_outer_round >= self.outer_max_round


class RandEdgeSampler(object):
  def __init__(self, src_list, dst_list, seed=None):
    self.seed = None
    self.src_list = np.unique(src_list)
    self.dst_list = np.unique(dst_list)

    if seed is not None:
      self.seed = seed
      self.random_state = np.random.RandomState(self.seed)

  def sample(self, size):
    if self.seed is None:
      src_index = np.random.randint(0, len(self.src_list), size)
      dst_index = np.random.randint(0, len(self.dst_list), size)
    else:

      src_index = self.random_state.randint(0, len(self.src_list), size)
      dst_index = self.random_state.randint(0, len(self.dst_list), size)
    return self.src_list[src_index], self.dst_list[dst_index]

  def reset_random_state(self):
    self.random_state = np.random.RandomState(self.seed)


def get_neighbor_finder(data, uniform, max_node_idx=None):
  max_node_idx = max(data.sources.max(), data.destinations.max()) if max_node_idx is None else max_node_idx
  adj_list = [[] for _ in range(max_node_idx + 1)]
  for source, destination, edge_idx, timestamp in zip(data.sources, data.destinations,
                                                      data.edge_idxs,
                                                      data.timestamps):
    adj_list[source].append((destination, edge_idx, timestamp))
    adj_list[destination].append((source, edge_idx, timestamp))

  return NeighborFinder(adj_list, uniform=uniform)


class NeighborFinder:
  def __init__(self, adj_list, uniform=False, seed=None):
    self.node_to_neighbors = []
    self.node_to_edge_idxs = []
    self.node_to_edge_timestamps = []

    for neighbors in adj_list:
      # Neighbors is a list of tuples (neighbor, edge_idx, timestamp)
      # We sort the list based on timestamp
      sorted_neighhbors = sorted(neighbors, key=lambda x: x[2])
      self.node_to_neighbors.append(np.array([x[0] for x in sorted_neighhbors]))
      self.node_to_edge_idxs.append(np.array([x[1] for x in sorted_neighhbors]))
      self.node_to_edge_timestamps.append(np.array([x[2] for x in sorted_neighhbors]))

    self.uniform = uniform

    if seed is not None:
      self.seed = seed
      self.random_state = np.random.RandomState(self.seed)

  def find_before(self, src_idx, cut_time):
    """
    Extracts all the interactions happening before cut_time for user src_idx in the overall interaction graph. The returned interactions are sorted by time.

    Returns 3 lists: neighbors, edge_idxs, timestamps

    """
    i = np.searchsorted(self.node_to_edge_timestamps[src_idx], cut_time)

    return self.node_to_neighbors[src_idx][:i], self.node_to_edge_idxs[src_idx][:i], self.node_to_edge_timestamps[src_idx][:i]

  def get_temporal_neighbor(self, source_nodes, timestamps, n_neighbors=20):
    """
    Given a list of users ids and relative cut times, extracts a sampled temporal neighborhood of each user in the list.

    Params
    ------
    src_idx_l: List[int]
    cut_time_l: List[float],
    num_neighbors: int
    """
    assert (len(source_nodes) == len(timestamps))

    tmp_n_neighbors = n_neighbors if n_neighbors > 0 else 1
    # NB! All interactions described in these matrices are sorted in each row by time
    neighbors = np.zeros((len(source_nodes), tmp_n_neighbors)).astype(
      np.int32)  # each entry in position (i,j) represent the id of the item targeted by user src_idx_l[i] with an interaction happening before cut_time_l[i]
    edge_times = np.zeros((len(source_nodes), tmp_n_neighbors)).astype(
      np.float32)  # each entry in position (i,j) represent the timestamp of an interaction between user src_idx_l[i] and item neighbors[i,j] happening before cut_time_l[i]
    edge_idxs = np.zeros((len(source_nodes), tmp_n_neighbors)).astype(
      np.int32)  # each entry in position (i,j) represent the interaction index of an interaction between user src_idx_l[i] and item neighbors[i,j] happening before cut_time_l[i]

    for i, (source_node, timestamp) in enumerate(zip(source_nodes, timestamps)):
      source_neighbors, source_edge_idxs, source_edge_times = self.find_before(source_node,
                                                   timestamp)  # extracts all neighbors, interactions indexes and timestamps of all interactions of user source_node happening before cut_time

      if len(source_neighbors) > 0 and n_neighbors > 0:
        if self.uniform:  # if we are applying uniform sampling, shuffles the data above before sampling
          sampled_idx = np.random.randint(0, len(source_neighbors), n_neighbors)

          neighbors[i, :] = source_neighbors[sampled_idx]
          edge_times[i, :] = source_edge_times[sampled_idx]
          edge_idxs[i, :] = source_edge_idxs[sampled_idx]

          # re-sort based on time
          pos = edge_times[i, :].argsort()
          neighbors[i, :] = neighbors[i, :][pos]
          edge_times[i, :] = edge_times[i, :][pos]
          edge_idxs[i, :] = edge_idxs[i, :][pos]
        else:
          # Take most recent interactions
          source_edge_times = source_edge_times[-n_neighbors:]
          source_neighbors = source_neighbors[-n_neighbors:]
          source_edge_idxs = source_edge_idxs[-n_neighbors:]

          assert (len(source_neighbors) <= n_neighbors)
          assert (len(source_edge_times) <= n_neighbors)
          assert (len(source_edge_idxs) <= n_neighbors)

          neighbors[i, n_neighbors - len(source_neighbors):] = source_neighbors
          edge_times[i, n_neighbors - len(source_edge_times):] = source_edge_times
          edge_idxs[i, n_neighbors - len(source_edge_idxs):] = source_edge_idxs

    return neighbors, edge_idxs, edge_times

def get_sampler(data_name='wikipedia'):   
    _, _, full_data, train_data, val_data, test_data, new_node_val_data, \
    new_node_test_data = get_data(data_name)
    train_rand_sampler = RandEdgeSampler(train_data.sources, train_data.destinations, seed=11)
    val_rand_sampler = RandEdgeSampler(full_data.sources, full_data.destinations, seed=0)

    test_rand_sampler = RandEdgeSampler(full_data.sources, full_data.destinations, seed=2)
    nn_test_rand_sampler = RandEdgeSampler(new_node_test_data.sources,
                                           new_node_test_data.destinations,
                                           seed=3)
    return train_rand_sampler, val_rand_sampler, test_rand_sampler, nn_test_rand_sampler

def get_sampler_newdata(data_name='wikipedia'):   
    node_features, edge_features, full_data, train_data, val_data, test_data, new_node_val_data, \
    new_node_test_data = get_data(data_name)
    train_rand_sampler = RandEdgeSampler(train_data.sources, train_data.destinations, seed=11)
    val_rand_sampler = RandEdgeSampler(full_data.sources, full_data.destinations, seed=0)

    test_rand_sampler = RandEdgeSampler(full_data.sources, full_data.destinations, seed=2)
    nn_test_rand_sampler = RandEdgeSampler(new_node_test_data.sources,
                                           new_node_test_data.destinations,
                                           seed=3)
    new_val_data = TemporalData(src=torch.tensor(new_node_val_data.sources)-1, dst=torch.tensor(new_node_val_data.destinations)-1, 
                            t=torch.tensor(new_node_val_data.timestamps, dtype=torch.long), msg=torch.tensor(edge_features[new_node_val_data.edge_idxs], dtype=torch.float), y=torch.tensor(new_node_val_data.labels))
    
    new_test_data = TemporalData(src=torch.tensor(new_node_test_data.sources)-1, dst=torch.tensor(new_node_test_data.destinations)-1, 
                            t=torch.tensor(new_node_test_data.timestamps, dtype=torch.long), msg=torch.tensor(edge_features[new_node_test_data.edge_idxs], dtype=torch.float), y=torch.tensor(new_node_test_data.labels))

    return train_rand_sampler, val_rand_sampler, test_rand_sampler, nn_test_rand_sampler, new_val_data, new_test_data

def get_sampler_4inductive(data_name='wikipedia'):   
    node_features, edge_features, full_data, train_data, val_data, test_data, new_node_val_data, \
    new_node_test_data = get_data(data_name)
    train_rand_sampler = RandEdgeSampler(train_data.sources, train_data.destinations, seed=11)
    val_rand_sampler = RandEdgeSampler(full_data.sources, full_data.destinations, seed=0)
    nn_val_rand_sampler = RandEdgeSampler(new_node_val_data.sources, new_node_val_data.destinations,
                                      seed=1)

    test_rand_sampler = RandEdgeSampler(full_data.sources, full_data.destinations, seed=2)
    nn_test_rand_sampler = RandEdgeSampler(new_node_test_data.sources,
                                           new_node_test_data.destinations,
                                           seed=3)
    train_data_1 = TemporalData(src=torch.tensor(train_data.sources)-1, dst=torch.tensor(train_data.destinations)-1, 
                            t=torch.tensor(train_data.timestamps, dtype=torch.long), msg=torch.tensor(edge_features[train_data.edge_idxs], dtype=torch.float), y=torch.tensor(train_data.labels))
    
    val_data_1 = TemporalData(src=torch.tensor(val_data.sources)-1, dst=torch.tensor(val_data.destinations)-1, 
                            t=torch.tensor(val_data.timestamps, dtype=torch.long), msg=torch.tensor(edge_features[val_data.edge_idxs], dtype=torch.float), y=torch.tensor(val_data.labels))
    
    test_data_1 = TemporalData(src=torch.tensor(test_data.sources)-1, dst=torch.tensor(test_data.destinations)-1, 
                            t=torch.tensor(test_data.timestamps, dtype=torch.long), msg=torch.tensor(edge_features[test_data.edge_idxs], dtype=torch.float), y=torch.tensor(test_data.labels))
    
    
    new_val_data = TemporalData(src=torch.tensor(new_node_val_data.sources)-1, dst=torch.tensor(new_node_val_data.destinations)-1, 
                            t=torch.tensor(new_node_val_data.timestamps, dtype=torch.long), msg=torch.tensor(edge_features[new_node_val_data.edge_idxs], dtype=torch.float), y=torch.tensor(new_node_val_data.labels))
    
    new_test_data = TemporalData(src=torch.tensor(new_node_test_data.sources)-1, dst=torch.tensor(new_node_test_data.destinations)-1, 
                            t=torch.tensor(new_node_test_data.timestamps, dtype=torch.long), msg=torch.tensor(edge_features[new_node_test_data.edge_idxs], dtype=torch.float), y=torch.tensor(new_node_test_data.labels))

    return train_rand_sampler, val_rand_sampler, test_rand_sampler, nn_val_rand_sampler, nn_test_rand_sampler, train_data_1, val_data_1, test_data_1, new_val_data, new_test_data 