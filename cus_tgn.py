import copy
from typing import Callable, Tuple

import torch
from torch import Tensor, nn
from torch.nn import Linear, GRUCell
from torch_scatter import scatter, scatter_max

from torch_geometric.nn.inits import zeros

from pytorch_wavelets import DWT1DForward # or simply DWT1D, IDWT1D

class FeedForward(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
            nn.LayerNorm(out_dim)
        )
    def forward(self, x):
        return self.net(x)
    def reset_parameters(self):
        for layer in self.children():
            for n, l in layer.named_modules():
                if hasattr(l, 'reset_parameters'):
                    print(f'Reset trainable parameters of layer = {l}')
                    l.reset_parameters()
                    

class TGNMemory(torch.nn.Module):
    r"""The Temporal Graph Network (TGN) memory model from the
    `"Temporal Graph Networks for Deep Learning on Dynamic Graphs"
    <https://arxiv.org/abs/2006.10637>`_ paper.
    .. note::
        For an example of using TGN, see `examples/tgn.py
        <https://github.com/rusty1s/pytorch_geometric/blob/master/examples/
        tgn.py>`_.
    Args:
        num_nodes (int): The number of nodes to save memories for.
        raw_msg_dim (int): The raw message dimensionality.
        memory_dim (int): The hidden memory dimensionality.
        time_dim (int): The time encoding dimensionality.
        message_module (torch.nn.Module): The message function which
            combines source and destination node memory embeddings, the raw
            message and the time encoding.
        aggregator_module (torch.nn.Module): The message aggregator function
            which aggregates messages to the same destination into a single
            representation.
    """
    def __init__(self, num_nodes: int, raw_msg_dim: int, memory_dim: int,
                 time_dim: int, message_module: Callable,
                 aggregator_module: Callable):
        super(TGNMemory, self).__init__()

        self.num_nodes = num_nodes
        self.raw_msg_dim = raw_msg_dim
        self.memory_dim = memory_dim
        self.time_dim = time_dim

        self.msg_s_module = message_module
        self.msg_d_module = copy.deepcopy(message_module)
        self.aggr_module = aggregator_module
        self.time_enc = TimeEncoder(time_dim)
        
        # add custom wavelet 
        self.dwt = DWT1DForward(wave='db1', J=2)
        self.lin = FeedForward(20, 10, 3, dropout = 0.5)
        self.gru = GRUCell(3*raw_msg_dim+memory_dim, memory_dim, bias=False)
        self.ln = torch.nn.LayerNorm(3*raw_msg_dim+memory_dim)
        
        self.register_buffer('memory', torch.empty(num_nodes, memory_dim))
        last_update = torch.empty(self.num_nodes, dtype=torch.long)
        self.register_buffer('last_update', last_update)
        self.register_buffer('__assoc__',
                             torch.empty(num_nodes, dtype=torch.long))
        
        self.msg_s_store = {}
        self.msg_d_store = {}
        self.wave_s_store = []
        self.wave_d_store = []
#         self.wave_s_store_val = []
#         self.wave_d_store_test = []
#         self.wave_s_store_val = []
#         self.wave_d_store_test = []
        self.wave_store_exist = False
#         self.wave_store_val_exist = False
#         self.wave_store_test_exist = False
        self.t_batch = 0
        
        self.reset_parameters()
  

        
    def reset_parameters(self):
        if hasattr(self.msg_s_module, 'reset_parameters'):
            self.msg_s_module.reset_parameters()
        if hasattr(self.msg_d_module, 'reset_parameters'):
            self.msg_d_module.reset_parameters()
        if hasattr(self.aggr_module, 'reset_parameters'):
            self.aggr_module.reset_parameters()
        self.time_enc.reset_parameters()
        self.gru.reset_parameters()
        self.lin.reset_parameters()
        self.ln.reset_parameters()
        self.reset_state()

    def reset_state(self):
        """Resets the memory to its initial state."""
        zeros(self.memory)
        zeros(self.last_update)
        self.__reset_message_store__()
        self.t_batch = 0

    def detach(self):
        """Detachs the memory from gradient computation."""
        self.memory.detach_()

    def forward(self, n_id: Tensor) -> Tuple[Tensor, Tensor]:
        """Returns, for all nodes :obj:`n_id`, their current memory and their
        last updated timestamp."""
        if self.training:
            memory = self.__get_updated_memory__(n_id)
        else:
            memory = self.memory[n_id]

        return memory

    def update_state(self, src, dst, t, raw_msg):
        """Updates the memory with newly encountered interactions
        :obj:`(src, dst, t, raw_msg)`."""
        n_id = torch.cat([src, dst]).unique()

        if self.training:
            self.__update_memory__(n_id)
            self.__update_msg_store__(src, dst, t, raw_msg, self.msg_s_store, self.wave_s_store)
            self.__update_msg_store__(dst, src, t, raw_msg, self.msg_d_store, self.wave_d_store)
        else:
            self.__update_msg_store__(src, dst, t, raw_msg, self.msg_s_store, self.wave_s_store)
            self.__update_msg_store__(dst, src, t, raw_msg, self.msg_d_store, self.wave_d_store)
            self.__update_memory__(n_id)
        self.t_batch += 1

    def __reset_message_store__(self):

        # Message store format: (src, dst, t, msg)       
        i = self.memory.new_empty((0, ), dtype=torch.long)
        msg = self.memory.new_empty((0, ))
        # Message store format: (src, dst, t, msg)
        self.msg_s_store = {j: (i, msg) for j in range(self.num_nodes)}
        self.msg_d_store = {j: (i, msg) for j in range(self.num_nodes)}

    def __update_memory__(self, n_id):
        memory = self.__get_updated_memory__(n_id)
        self.memory[n_id] = memory
#         self.last_update[n_id] = last_update

    def __get_updated_memory__(self, n_id):
        self.__assoc__[n_id] = torch.arange(n_id.size(0), device=n_id.device)
        
#         self.lin1(torch.cat(self.msg_s_store[i], dim=-1)).view(1,-1)
        # Compute messages (src -> dst).
#         msg_s, t_s, src_s, dst_s = self.__compute_msg__(
#             n_id, self.msg_s_store, self.msg_s_module)
        data = [self.msg_s_store[i] for i in n_id.tolist()]
        src_s, raw_msg_s = list(zip(*data))
        src_s = torch.cat(src_s, dim=0)
        msg_s = torch.cat(raw_msg_s, dim=0)      
        msg_s_compress = self.lin(msg_s).view(msg_s.size(0), -1) if msg_s.size(0)!= 0 else msg_s
        msg_s_compress = torch.cat((self.memory[src_s], msg_s_compress), dim=-1)
        # Compute messages (dst -> src).
#         msg_d, t_d, src_d, dst_d = self.__compute_msg__(
#             n_id, self.msg_d_store, self.msg_d_module)
        data = [self.msg_d_store[i] for i in n_id.tolist()]
        src_d, raw_msg_d = list(zip(*data))
        src_d = torch.cat(src_d, dim=0)
        msg_d = torch.cat(raw_msg_d, dim=0)      
        msg_d_compress = self.lin(msg_d).view(msg_d.size(0), -1) if msg_d.size(0)!= 0 else msg_d
        msg_d_compress = torch.cat((self.memory[src_d], msg_d_compress), dim=-1)
        # Aggregate messages.
        idx = torch.cat([src_s, src_d], dim=0)
        msg = torch.cat([msg_s_compress, msg_d_compress], dim=0)
        if msg.size(0)!=0:
            msg_perm = msg.new_zeros((n_id.size(0), msg.size(-1)))
            msg_perm[self.__assoc__[idx]] = msg
            # Get local copy of updated memory.
            memory = self.gru(self.ln(msg_perm), self.memory[n_id])
        else:
            memory = self.memory[n_id]
        


        return memory

    def __update_msg_store__(self, src, dst, t, raw_msg, msg_store, wave_store):
        # wavelet raw and also as msg
        if not self.wave_store_exist: 
#         if self.training:
            wave_store_t = {}
            t_inte = torch.linspace(min(t), max(t), 20).to(self.memory.device)
            t_inte_tmp = t_inte[:, None]

            n_id, perm = src.sort()
            n_id, count = n_id.unique_consecutive(return_counts=True)

            for i, idx in zip(n_id.tolist(), perm.split(count.tolist())):
                dist = abs(t[None, idx] - t_inte_tmp)
                raw_msg_inte = raw_msg[idx[dist.argmin(1)]]
                yl, yh = self.dwt((raw_msg_inte.T)[None,:])
                tmp = torch.cat((yl, *yh), dim=-1)
                msg_store[i] = (torch.tensor([i], dtype=torch.long, device=self.memory.device), tmp)
                wave_store_t[i] = msg_store[i]
            wave_store.append(wave_store_t)
#             else:
#                 t_inte = torch.linspace(min(t), max(t), 20).to(self.memory.device)
#                 t_inte_tmp = t_inte[:, None]

#                 n_id, perm = src.sort()
#                 n_id, count = n_id.unique_consecutive(return_counts=True)
                
#                 for i, idx in zip(n_id.tolist(), perm.split(count.tolist())):
#                     dist = abs(t[None, idx] - t_inte_tmp)
#                     raw_msg_inte = raw_msg[idx[dist.argmin(1)]]
#                     yl, yh = self.dwt((raw_msg_inte.T)[None,:])
#                     tmp = torch.cat((yl, *yh), dim=-1)
#                     msg_store[i] = (torch.tensor([i], dtype=torch.long, device=self.memory.device), tmp)


        else:
            wave_store_t = wave_store[self.t_batch]
            n_id, perm = src.sort()
            n_id, count = n_id.unique_consecutive(return_counts=True)
            
            for i, idx in zip(n_id.tolist(), perm.split(count.tolist())):
                msg_store[i] = wave_store_t[i]
                
    
#         n_id, perm = src.sort()
#         n_id, count = n_id.unique_consecutive(return_counts=True)
#         for i, idx in zip(n_id.tolist(), perm.split(count.tolist())):
#             msg_store[i] = (src[idx], dst[idx], t[idx], raw_msg[idx])

    def __compute_msg__(self, n_id, msg_store, msg_module):
        data = [msg_store[i] for i in n_id.tolist()]
        src, dst, t, raw_msg = list(zip(*data))
        src = torch.cat(src, dim=0)
        dst = torch.cat(dst, dim=0)
        t = torch.cat(t, dim=0)
        raw_msg = torch.cat(raw_msg, dim=0)
        t_rel = t - self.last_update[src]
        t_enc = self.time_enc(t_rel.to(raw_msg.dtype))

        msg = msg_module(self.memory[src], self.memory[dst], raw_msg, t_enc)

        return msg, t, src, dst

    def train(self, mode: bool = True):
        """Sets the module in training mode."""
        if self.training and not mode:
            # Flush message store to memory in case we just entered eval mode.
            self.__update_memory__(
                torch.arange(self.num_nodes, device=self.memory.device))
            self.__reset_message_store__()
        super(TGNMemory, self).train(mode)


class IdentityMessage(torch.nn.Module):
    def __init__(self, raw_msg_dim: int, memory_dim: int, time_dim: int):
        super(IdentityMessage, self).__init__()
        self.out_channels = raw_msg_dim + 2 * memory_dim + time_dim

    def forward(self, z_src, z_dst, raw_msg, t_enc):
        return torch.cat([z_src, z_dst, raw_msg, t_enc], dim=-1)


class LastAggregator(torch.nn.Module):
    def forward(self, msg, index, t, dim_size):
        _, argmax = scatter_max(t, index, dim=0, dim_size=dim_size)
        out = msg.new_zeros((dim_size, msg.size(-1)))
        mask = argmax < msg.size(0)  # Filter items with at least one entry.
        out[mask] = msg[argmax[mask]]
        return out


class MeanAggregator(torch.nn.Module):
    def forward(self, msg, index, t, dim_size):
        return scatter(msg, index, dim=0, dim_size=dim_size, reduce='mean')


class TimeEncoder(torch.nn.Module):
    def __init__(self, out_channels):
        super(TimeEncoder, self).__init__()
        self.out_channels = out_channels
        self.lin = Linear(1, out_channels)

    def reset_parameters(self):
        self.lin.reset_parameters()

    def forward(self, t):
        return self.lin(t.view(-1, 1)).cos()


class LastNeighborLoader(object):
    def __init__(self, num_nodes: int, size: int, device=None):
        self.size = size

        self.neighbors = torch.empty((num_nodes, size), dtype=torch.long,
                                     device=device)
        self.e_id = torch.empty((num_nodes, size), dtype=torch.long,
                                device=device)
        self.__assoc__ = torch.empty(num_nodes, dtype=torch.long,
                                     device=device)

        self.reset_state()

    def __call__(self, n_id):
        neighbors = self.neighbors[n_id]
        nodes = n_id.view(-1, 1).repeat(1, self.size)
        e_id = self.e_id[n_id]

        # Filter invalid neighbors (identified by `e_id < 0`).
        mask = e_id >= 0
        neighbors, nodes, e_id = neighbors[mask], nodes[mask], e_id[mask]

        # Relabel node indices.
        n_id = torch.cat([n_id, neighbors]).unique()
#         self.__assoc__[n_id] = torch.arange(n_id.size(0), device=n_id.device)
#         neighbors, nodes = self.__assoc__[neighbors], self.__assoc__[nodes]

        return n_id, torch.stack([neighbors, nodes]), e_id

    def insert(self, src, dst):
        # Inserts newly encountered interactions into an ever growing
        # (undirected) temporal graph.

        # Collect central nodes, their neighbors and the current event ids.
        neighbors = torch.cat([src, dst], dim=0)
        nodes = torch.cat([dst, src], dim=0)
        e_id = torch.arange(self.cur_e_id, self.cur_e_id + src.size(0),
                            device=src.device).repeat(2)
        self.cur_e_id += src.numel()

        # Convert newly encountered interaction ids so that they point to
        # locations of a "dense" format of shape [num_nodes, size].
        nodes, perm = nodes.sort()
        neighbors, e_id = neighbors[perm], e_id[perm]

        n_id = nodes.unique()
        self.__assoc__[n_id] = torch.arange(n_id.numel(), device=n_id.device)

        dense_id = torch.arange(nodes.size(0), device=nodes.device) % self.size
        dense_id += self.__assoc__[nodes].mul_(self.size)

        dense_e_id = e_id.new_full((n_id.numel() * self.size, ), -1)
        dense_e_id[dense_id] = e_id
        dense_e_id = dense_e_id.view(-1, self.size)

        dense_neighbors = e_id.new_empty(n_id.numel() * self.size)
        dense_neighbors[dense_id] = neighbors
        dense_neighbors = dense_neighbors.view(-1, self.size)

        # Collect new and old interactions...
        e_id = torch.cat([self.e_id[n_id, :self.size], dense_e_id], dim=-1)
        neighbors = torch.cat(
            [self.neighbors[n_id, :self.size], dense_neighbors], dim=-1)

        # And sort them based on `e_id`.
        e_id, perm = e_id.topk(self.size, dim=-1)
        self.e_id[n_id] = e_id
        self.neighbors[n_id] = torch.gather(neighbors, 1, perm)

    def reset_state(self):
        self.cur_e_id = 0
        self.e_id.fill_(-1)