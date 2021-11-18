import numpy as np
from scipy import sparse
from scipy.sparse.linalg import lobpcg
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_sparse import spmm
from torch_geometric.utils import get_laplacian
import torch_geometric.transforms as T
# from ogb.nodeproppred import PygNodePropPredDataset, Evaluator
# from logger import Logger
# import argparse
import math

# from torch_geometric.nn import GATConv
# from torch_geometric.utils import to_undirected


# function for pre-processing
# @torch.no_grad()
# def scipy_to_torch_sparse(A):
#     A = sparse.coo_matrix(A)
#     row = torch.tensor(A.row)
#     col = torch.tensor(A.col)
#     index = torch.stack((row, col), dim=0)
#     value = torch.Tensor(A.data)

#     return torch.sparse_coo_tensor(index, value, A.shape)

@torch.no_grad()
def scipy_to_torch_sparse(A, device):
    A = sparse.coo_matrix(A)
    row = torch.tensor(A.row, dtype=torch.int64)
    col = torch.tensor(A.col, dtype=torch.int64)
    index = torch.stack((row, col), dim=0)
    value = torch.Tensor(A.data)

    return (index.to(device), value.to(device))


# function for pre-processing
def ChebyshevApprox(f, n):  # Assuming f : [0, pi] -> R.
    quad_points = 500
    c = [None] * n
    a = math.pi / 2
    for k in range(1, n + 1):
        Integrand = lambda x: np.cos((k - 1) * x) * f(a * (np.cos(x) + 1))
        x = np.linspace(0, np.pi, quad_points)
        y = Integrand(x)
        c[k - 1] = 2 / np.pi * np.trapz(y, x)

    return c

# function for pre-processing
def get_operator(L, DFilters, n, s, J, Lev):
    r = len(DFilters)
    c = [None] * r
    for j in range(r):
        c[j] = ChebyshevApprox(DFilters[j], n)
    a = np.pi / 2  # consider the domain of masks as [0, pi]
    # Fast Tight Frame Decomposition (FTFD)
    FD1 = sparse.identity(L.shape[0])
    d = dict()
    for l in range(1, Lev + 1):
        for j in range(r):
            T0F = FD1
            T1F = ((s ** (-J + l - 1) / a) * L) @ T0F - T0F
            d[j, l - 1] = (1 / 2) * c[j][0] * T0F + c[j][1] * T1F
            for k in range(2, n):
                TkF = ((2 / a * s ** (-J + l - 1)) * L) @ T1F - 2 * T1F - T0F
                T0F = T1F
                T1F = TkF
                d[j, l - 1] += c[j][k] * TkF
        FD1 = d[0, l - 1]

    return d

# function for pre-processing get d_list
def get_d_list(edge_index, num_nodes):
    L = get_laplacian(edge_index, num_nodes=num_nodes, normalization='sym')
    L = sparse.coo_matrix((L[1].cpu().numpy(), (L[0][0, :].cpu().numpy(), L[0][1, :].cpu().numpy())), shape=(num_nodes, num_nodes))


    ## FrameType = 'Haar'
    FrameType = 'Haar'
    if FrameType == 'Haar':
        D1 = lambda x: np.cos(x / 2)
        D2 = lambda x: np.sin(x / 2)
        DFilters = [D1, D2]
        RFilters = [D1, D2]
    elif FrameType == 'Linear':
        D1 = lambda x: np.square(np.cos(x / 2))
        D2 = lambda x: np.sin(x) / np.sqrt(2)
        D3 = lambda x: np.square(np.sin(x / 2))
        DFilters = [D1, D2, D3]
        RFilters = [D1, D2, D3]
    elif FrameType == 'Quadratic':  # not accurate so far
        D1 = lambda x: np.cos(x / 2) ** 3
        D2 = lambda x: np.multiply((np.sqrt(3) * np.sin(x / 2)), np.cos(x / 2) ** 2)
        D3 = lambda x: np.multiply((np.sqrt(3) * np.sin(x / 2) ** 2), np.cos(x / 2))
        D4 = lambda x: np.sin(x / 2) ** 3
        DFilters = [D1, D2, D3, D4]
        RFilters = [D1, D2, D3, D4]
    else:
        raise Exception('Invalid FrameType')  

    Lev = 2  # level of transform
    dilation = 2  # dilation scale
    n = 2  # n - 1 = Degree of Chebyshev Polynomial Approximation
    lambda_max = 4
    J = np.log(lambda_max / np.pi) / np.log(dilation) + Lev - 1  # dilation level to start the decomposition
    r = len(DFilters)
    # perform the Chebyshev Approximation
    c = [None] * r
    for j in range(r):
        c[j] = ChebyshevApprox(DFilters[j], n)

    # get matrix operators
    d = get_operator(L, DFilters, n, dilation, J, Lev)
    # enhance sparseness of the matrix operators (optional)
    # d[np.abs(d) < 0.001] = 0.0
    # store the matrix operators (torch sparse format) into a list: row-by-row
    d_list = list()
    for i in range(r):
        for l in range(Lev):
            d_list.append(scipy_to_torch_sparse(d[i, l], edge_index.device))
    return d_list
                

class UFGConv(nn.Module):
    def __init__(self, in_features, out_features, r, Lev, num_nodes, shrinkage=None, threshold=1e-4, bias=False):
        super(UFGConv, self).__init__()
        self.Lev = Lev
        self.shrinkage = shrinkage
        self.threshold = threshold
        self.crop_len = (Lev - 1) * num_nodes
        self.num_nodes = num_nodes
        self.weight = nn.Parameter(torch.Tensor(in_features, out_features))
        self.filter1 = nn.Parameter(torch.Tensor(num_nodes, 1))
        self.filter2 = nn.Parameter(torch.Tensor(num_nodes, 1))                             
        self.filter3 = nn.Parameter(torch.Tensor(num_nodes, 1))                             
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
         
        self.reset_parameters()


    def reset_parameters(self):
        nn.init.uniform_(self.filter1, 0.9, 1.1)
        nn.init.uniform_(self.filter2, 0.9, 1.1)
        nn.init.uniform_(self.filter3, 0.9, 1.1)
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x, d_list):
        # d_list is a list of matrix operators (torch sparse format), row-by-row
        # x is a torch dense tensor
        x = torch.matmul(x, self.weight)

        # Fast Tight Frame Decomposition
#         x0 = spmm(d_list[0][0], d_list[0][1], self.num_nodes, self.num_nodes, x)
        x1 = spmm(d_list[1][0], d_list[1][1], self.num_nodes, self.num_nodes, x)
        x2 = spmm(d_list[2][0], d_list[2][1], self.num_nodes, self.num_nodes, x)
        x3 = spmm(d_list[3][0], d_list[3][1], self.num_nodes, self.num_nodes, x)
#         x = torch.cat((x1, x2, x3), dim=0)
#         x = torch.sparse.mm(torch.cat(d_list, dim=0), x)
        # the output x has shape [r * Lev * num_nodes, #Features]

        # perform wavelet shrinkage (optional)
        if self.shrinkage is not None:
            if self.shrinkage == 'soft':
                x = torch.mul(torch.sign(x), (((torch.abs(x) - self.threshold) + torch.abs(torch.abs(x) - self.threshold)) / 2))
            elif self.shrinkage == 'hard':
                x = torch.mul(x, (torch.abs(x) > self.threshold))
            else:
                raise Exception('Shrinkage type is invalid')

        # Hadamard product in spectral domain
        x1 = self.filter1 * x1
        x2 = self.filter2 * x2
        x3 = self.filter3 * x3
        # filter has shape [r * Lev * num_nodes, 1]
        # the output x has shape [r * Lev * num_nodes, #Features]

        # Fast Tight Frame Reconstruction
#         x = torch.sparse.mm(torch.cat(d_list[self.Lev - 1:], dim=1), x[self.crop_len:, :])
        x1 = spmm(d_list[1][0], d_list[1][1], self.num_nodes, self.num_nodes, x1)
        x2 = spmm(d_list[2][0], d_list[2][1], self.num_nodes, self.num_nodes, x2)
        x3 = spmm(d_list[3][0], d_list[3][1], self.num_nodes, self.num_nodes, x3)                            
        
        x = x1 + x2 + x3
        if self.bias is not None:
            x += self.bias
        return x


class Net(nn.Module):
    def __init__(self, num_features, nhid, num_classes, r, Lev, num_nodes, shrinkage=None, threshold=1e-4,
                 dropout_prob=0.5, num_layers=2, batch_norm=True):
        super(Net, self).__init__()

        self.batch_norm = batch_norm
        self.convs = torch.nn.ModuleList()
        self.convs.append(UFGConv(num_features, nhid, r, Lev, num_nodes, shrinkage=shrinkage, threshold=threshold))
        if self.batch_norm:
            self.bns = torch.nn.ModuleList()
            self.bns.append(torch.nn.LayerNorm(nhid))
        for _ in range(num_layers - 2):
            self.convs.append(UFGConv(nhid, nhid, r, Lev, num_nodes, shrinkage=shrinkage, threshold=threshold))
            if self.batch_norm:
                self.bns.append(torch.nn.BatchNorm1d(nhid))
        self.convs.append(UFGConv(nhid, num_classes, r, Lev, num_nodes, shrinkage=shrinkage, threshold=threshold))

        self.dropout_prob = dropout_prob

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        if self.batch_norm:
            for bn in self.bns:
                bn.reset_parameters()

    def forward(self, x, d_list, n_id):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, d_list)
#             num_nodes = x.shape[0]
#             memory_dim = x.shape[1]
#             x = x[n_id, :].clone()
            if self.batch_norm:
                x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout_prob, training=self.training)
            
#             i = torch.stack((n_id.view(-1,1).repeat(1,memory_dim).view(-1), torch.arange(memory_dim).repeat(n_id.size(0)).to(device)), dim=0)
#             x = torch.sparse_coo_tensor(i, x.view(-1), (num_nodes, memory_dim)).to_dense()
        x = self.convs[-1](x, d_list)
        return x