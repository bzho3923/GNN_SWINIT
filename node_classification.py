import math
import logging
import time
import sys
import argparse
import torch
import numpy as np
from pathlib import Path


from ufg_global import *
from utils.data_processing import findMode
from utils.utils import get_sampler

import os.path as osp
from torch.nn import Linear
from sklearn.metrics import average_precision_score, roc_auc_score

from torch_geometric.datasets import JODIEDataset
from torch_geometric.nn.models.tgn import (IdentityMessage,
                                           LastAggregator)
from cus_tgn import LastNeighborLoader


class GraphAttentionEmbedding(torch.nn.Module):
    def __init__(self, in_channels, out_channels, msg_dim, time_enc, num_nodes, batch_norm=True, drop_out=0.5):
        super(GraphAttentionEmbedding, self).__init__()
        self.time_enc = time_enc
        edge_dim = msg_dim + time_enc.out_channels
        self.conv = Net(in_channels, out_channels, out_channels, r=2, Lev=2, num_nodes=num_nodes, shrinkage=None,
                threshold=1e-3, dropout_prob=drop_out, num_layers=2, batch_norm=batch_norm)

    def forward(self, x, d_list, n_id):
        return self.conv(x, d_list, n_id)


class LinkPredictor(torch.nn.Module):
    def __init__(self, in_channels):
        super(LinkPredictor, self).__init__()
        self.lin_src = Linear(in_channels, in_channels)
        self.lin_dst = Linear(in_channels, in_channels)
        self.lin_final = Linear(in_channels, 1)

    def forward(self, z_src, z_dst):
        h = self.lin_src(z_src) + self.lin_dst(z_dst)
        h = h.relu()
        return self.lin_final(h)




def pre_train():
    memory.train()
    gnn.train()
    link_pred.train()

    memory.reset_state()  # Start with a fresh memory.
    neighbor_loader.reset_state()  # Start with an empty graph.

    total_loss, d_list_batch = 0, []
    for batch in train_data.seq_batches(batch_size=args.batch_size):

        optimizer.zero_grad()
        
        if data_process_type == 'svd':
            src, pos_dst, t, msg = batch.src, batch.dst, batch.t, batch.msg @ mode
        elif data_process_type == 'raw':
            src, pos_dst, t, msg = batch.src, batch.dst, batch.t, batch.msg
        elif data_process_type == 'raw+svd':
            src, pos_dst, t, msg = batch.src, batch.dst, batch.t, torch.cat((batch.msg, batch.msg @ mode), dim=-1)        
        else: raise NameError('wrong data preprocessing type')
        

        # Sample negative destination nodes.
        neg_dst = torch.randint(min_dst_idx, max_dst_idx + 1, (src.size(0), ),
                                dtype=torch.long, device=device)

        n_id = torch.cat([src, pos_dst, neg_dst]).unique()
        n_id, edge_index, e_id = neighbor_loader(n_id)

        # Get updated memory of all nodes involved in the computation.
        z, last_update = memory(n_id)
        
        num_nodes = data.num_nodes
        memory_dim = memory.memory_dim
        
        d_list = get_d_list(edge_index, num_nodes)
                
        d_list_batch.append(d_list)
        

        s = torch.zeros(data.num_nodes, memory_dim, device=device, dtype=torch.float32)
        s[n_id] = z
        z_full = gnn(s, d_list, n_id)

        pos_out = link_pred(z_full[src], z_full[pos_dst])
        neg_out = link_pred(z_full[src], z_full[neg_dst])

        loss = criterion(pos_out, torch.ones_like(pos_out))
        loss += criterion(neg_out, torch.zeros_like(neg_out))

        # Update memory and neighbor loader with ground-truth state.
        memory.update_state(src, pos_dst, t, msg)
        neighbor_loader.insert(src, pos_dst)

        loss.backward()
        optimizer.step()
        memory.detach()
        total_loss += float(loss) * batch.num_events

    return total_loss / train_data.num_events, d_list_batch


def train(d_list_batch):
    memory.train()
    gnn.train()
    link_pred.train()

    memory.reset_state()  # Start with a fresh memory.
    neighbor_loader.reset_state()  # Start with an empty graph.

    total_loss, t_batch = 0, 0
    for batch in train_data.seq_batches(batch_size=args.batch_size):
        optimizer.zero_grad()

        if data_process_type == 'svd':
            src, pos_dst, t, msg = batch.src, batch.dst, batch.t, batch.msg @ mode
        elif data_process_type == 'raw':
            src, pos_dst, t, msg = batch.src, batch.dst, batch.t, batch.msg
        elif data_process_type == 'raw+svd':
            src, pos_dst, t, msg = batch.src, batch.dst, batch.t, torch.cat((batch.msg, batch.msg @ mode), dim=-1)        
        else: raise NameError('wrong data preprocessing type')

        # Sample negative destination nodes.
        neg_dst = torch.randint(min_dst_idx, max_dst_idx + 1, (src.size(0), ),
                                dtype=torch.long, device=device)

        n_id = torch.cat([src, pos_dst, neg_dst]).unique()
        n_id, edge_index, e_id = neighbor_loader(n_id)

        # Get updated memory of all nodes involved in the computation.
        z, last_update = memory(n_id)
        
        num_nodes = data.num_nodes
        memory_dim = memory.memory_dim
        
        d_list = d_list_batch[t_batch]
        t_batch += 1
        
        s = torch.zeros(data.num_nodes, memory_dim, device=device, dtype=torch.float32)
        s[n_id] = z
        z_full = gnn(s, d_list, n_id)

        pos_out = link_pred(z_full[src], z_full[pos_dst])
        neg_out = link_pred(z_full[src], z_full[neg_dst])

        loss = criterion(pos_out, torch.ones_like(pos_out))
        loss += criterion(neg_out, torch.zeros_like(neg_out))

        # Update memory and neighbor loader with ground-truth state.
        memory.update_state(src, pos_dst, t, msg)
        neighbor_loader.insert(src, pos_dst)

        loss.backward()
        optimizer.step()
        memory.detach()
        total_loss += float(loss) * batch.num_events

    return total_loss / train_data.num_events

@torch.no_grad()
def pre_test(inference_data):
    memory.eval()
    gnn.eval()
    link_pred.eval()


    torch.manual_seed(12345)  # Ensure deterministic sampling across epochs.

    aps, aucs, d_list_batch_test = [], [], []
    for batch in inference_data.seq_batches(batch_size=args.batch_size):
        
        if data_process_type == 'svd':
            src, pos_dst, t, msg = batch.src, batch.dst, batch.t, batch.msg @ mode
        elif data_process_type == 'raw':
            src, pos_dst, t, msg = batch.src, batch.dst, batch.t, batch.msg
        elif data_process_type == 'raw+svd':
            src, pos_dst, t, msg = batch.src, batch.dst, batch.t, torch.cat((batch.msg, batch.msg @ mode), dim=-1)        
        else: raise NameError('wrong data preprocessing type')

        neg_dst = torch.randint(min_dst_idx, max_dst_idx + 1, (src.size(0), ),
                                dtype=torch.long, device=device)

        n_id = torch.cat([src, pos_dst, neg_dst]).unique()
        n_id, edge_index, e_id = neighbor_loader(n_id)
        assoc[n_id] = torch.arange(n_id.size(0), device=device)

        z, last_update = memory(n_id)
        num_nodes = data.num_nodes
        memory_dim = memory.memory_dim
        
        d_list = get_d_list(edge_index, num_nodes)
        d_list_batch_test.append(d_list)
        
        s = torch.zeros(data.num_nodes, memory_dim, device=device, dtype=torch.float32)
        s[n_id] = z
        z_full = gnn(s, d_list, n_id)

        pos_out = link_pred(z_full[src], z_full[pos_dst])
        neg_out = link_pred(z_full[src], z_full[neg_dst])

        y_pred = torch.cat([pos_out, neg_out], dim=0).sigmoid().cpu()
        y_true = torch.cat(
            [torch.ones(pos_out.size(0)),
             torch.zeros(neg_out.size(0))], dim=0)

        aps.append(average_precision_score(y_true, y_pred))
        aucs.append(roc_auc_score(y_true, y_pred))

        memory.update_state(src, pos_dst, t, msg)
        neighbor_loader.insert(src, pos_dst)

    return float(torch.tensor(aps).mean()), float(torch.tensor(aucs).mean()), d_list_batch_test

@torch.no_grad()
def test(inference_data, d_list_batch):
    memory.eval()
    gnn.eval()
    link_pred.eval()
    
    torch.manual_seed(12345)  # Ensure deterministic sampling across epochs.

    aps, aucs, t_batch = [], [], 0
    for batch in inference_data.seq_batches(batch_size=args.batch_size):
        
        if data_process_type == 'svd':
            src, pos_dst, t, msg = batch.src, batch.dst, batch.t, batch.msg @ mode
        elif data_process_type == 'raw':
            src, pos_dst, t, msg = batch.src, batch.dst, batch.t, batch.msg
        elif data_process_type == 'raw+svd':
            src, pos_dst, t, msg = batch.src, batch.dst, batch.t, torch.cat((batch.msg, batch.msg @ mode), dim=-1)        
        else: raise NameError('wrong data preprocessing type')
            
        neg_dst = torch.randint(min_dst_idx, max_dst_idx + 1, (src.size(0), ),
                                dtype=torch.long, device=device)

        n_id = torch.cat([src, pos_dst, neg_dst]).unique()
        n_id, edge_index, e_id = neighbor_loader(n_id)
        assoc[n_id] = torch.arange(n_id.size(0), device=device)

        z, last_update = memory(n_id)
        num_nodes = data.num_nodes
        memory_dim = memory.memory_dim
        d_list = d_list_batch[t_batch]
        t_batch += 1
        
        s = torch.zeros(data.num_nodes, memory_dim, device=device, dtype=torch.float32)
        s[n_id] = z
        z_full = gnn(s, d_list, n_id)

        pos_out = link_pred(z_full[src], z_full[pos_dst])
        neg_out = link_pred(z_full[src], z_full[neg_dst])

        y_pred = torch.cat([pos_out, neg_out], dim=0).sigmoid().cpu()
        y_true = torch.cat(
            [torch.ones(pos_out.size(0)),
             torch.zeros(neg_out.size(0))], dim=0)

        aps.append(average_precision_score(y_true, y_pred))
        aucs.append(roc_auc_score(y_true, y_pred))

        memory.update_state(src, pos_dst, t, msg)
        neighbor_loader.insert(src, pos_dst)


    return float(torch.tensor(aps).mean()), float(torch.tensor(aucs).mean())

def train_node_class(d_list_batch):   
    memory.eval()
    gnn.eval()

    memory.reset_state()  # Start with a fresh memory.
    neighbor_loader.reset_state()  # Start with an empty graph.
    decoder.train()
    total_loss, t_batch = 0, 0

    for batch in train_data.seq_batches(batch_size=args.batch_size):
        if data_process_type == 'svd':
            src, pos_dst, t, msg, labels = batch.src, batch.dst, batch.t, batch.msg @ mode, batch.y
        elif data_process_type == 'raw':
            src, pos_dst, t, msg, labels = batch.src, batch.dst, batch.t, batch.msg, batch.y
        elif data_process_type == 'raw+svd':
            src, pos_dst, t, msg, labels = batch.src, batch.dst, batch.t, torch.cat((batch.msg, batch.msg @ mode), dim=-1), batch.y        
        else: raise NameError('wrong data preprocessing type')
            
        decoder_optimizer.zero_grad()

        n_id = torch.cat([src, pos_dst]).unique()
        n_id, edge_index, e_id = neighbor_loader(n_id)
        assoc[n_id] = torch.arange(n_id.size(0), device=device)

        # Get updated memory of all nodes involved in the computation.
        with torch.no_grad():
            z, last_update = memory(n_id)
            num_nodes = data.num_nodes
            memory_dim = memory.memory_dim
            d_list = d_list_batch[t_batch]
            t_batch += 1

            s = torch.zeros(data.num_nodes, memory_dim, device=device, dtype=torch.float32)
            s[n_id] = z
            z_full = gnn(s, d_list, n_id)

        labels_batch_torch = labels.float().to(device)
        pred = decoder(z[assoc[src]]).sigmoid()
        decoder_loss = decoder_loss_criterion(pred, labels_batch_torch)
        
        memory.update_state(src, pos_dst, t, msg)
        neighbor_loader.insert(src, pos_dst)
        memory.detach()
        
        decoder_loss.backward()
        decoder_optimizer.step()

        total_loss += float(decoder_loss) * batch.num_events

    return total_loss / train_data.num_events

@torch.no_grad()
def test_node_class(inference_data, d_list_batch):
    memory.eval()
    gnn.eval()
    decoder.eval()

   
    pred_list, t_batch = [], 0
    for batch in inference_data.seq_batches(batch_size=args.batch_size):
        if data_process_type == 'svd':
            src, pos_dst, t, msg, labels = batch.src, batch.dst, batch.t, batch.msg @ mode, batch.y
        elif data_process_type == 'raw':
            src, pos_dst, t, msg, labels = batch.src, batch.dst, batch.t, batch.msg, batch.y
        elif data_process_type == 'raw+svd':
            src, pos_dst, t, msg, labels = batch.src, batch.dst, batch.t, torch.cat((batch.msg, batch.msg @ mode), dim=-1), batch.y        
        else: raise NameError('wrong data preprocessing type')

        n_id = torch.cat([src, pos_dst]).unique()
        n_id, edge_index, e_id = neighbor_loader(n_id)
        assoc[n_id] = torch.arange(n_id.size(0), device=device)

        z, last_update = memory(n_id)
        num_nodes = data.num_nodes
        memory_dim = memory.memory_dim
        d_list = d_list_batch[t_batch]
        t_batch += 1
        
        s = torch.zeros(data.num_nodes, memory_dim, device=device, dtype=torch.float32)
        s[n_id] = z
        z_full = gnn(s, d_list, n_id)

        pred = decoder(z[assoc[src]]).sigmoid().cpu()


        pred_list.append(pred)
        memory.update_state(src, pos_dst, t, msg)
        neighbor_loader.insert(src, pos_dst)

    return pred_list 


### Argument and global variables
parser = argparse.ArgumentParser()
parser.add_argument('-d', '--data', type=str, help='Dataset name (eg. wikipedia or reddit)',
                    default='wikipedia')
# parser.add_argument('--bs', type=int, default=200, help='Batch_size')
parser.add_argument('--prefix', type=str, default='', help='Prefix to name the checkpoints')
parser.add_argument('--drop_out', type=float, default=0.5, help='dropout rate')
parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
parser.add_argument('--memory_type', type=str, default='gru', help='The type of memory function')
parser.add_argument('--data_process_type', type=str, default='svd', help='Process the data')
parser.add_argument('--num_modes', type=int, default=100, help='Number of modes of svd')
parser.add_argument('--n_runs', type=int, default=1, help='Number of runs')
parser.add_argument('--early_stopper', type=int, default=8, help='Number of waits for early stopper')
parser.add_argument('--num_epoch', type=int, default=20, help='Number of epoch')
parser.add_argument('--batch_size', type=int, default=200, help='batch size')
parser.add_argument('--gpu', type=int, default=0, help='Idx for the gpu to use')
parser.add_argument('--embedding_dim', type=int, default=100, help='Dimensions of the node embedding')
parser.add_argument('--time_dim', type=int, default=100, help='Dimensions of the time embedding')
parser.add_argument('--memory_dim', type=int, default=100, help='Dimensions of the time embedding')
parser.add_argument('--batch_norm', dest='batch_norm', action='store_true')
parser.set_defaults(batch_norm=True)
parser.add_argument('--no-batch_norm', dest='batch_norm', action='store_false')



try:
  args = parser.parse_args() 

except:
  parser.print_help()
  sys.exit(0)
    
    
Path("./saved_models/").mkdir(parents=True, exist_ok=True)
Path("./saved_checkpoints/").mkdir(parents=True, exist_ok=True)
MODEL_SAVE_PATH = f'./saved_models/{args.data}-{args.data_process_type}-{args.num_modes}-{args.memory_type}-{args.memory_dim}.pth'
get_checkpoint_path = lambda \
    epoch: f'./saved_checkpoints/{args.data_process_type}-{args.prefix}-{args.data}-epoch{epoch}-run{reps}.pth'

### set up logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
Path("log/").mkdir(parents=True, exist_ok=True)
fh = logging.FileHandler('log/{}.log'.format(str(time.time())))
fh.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.WARN)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
ch.setFormatter(formatter)
logger.addHandler(fh)
logger.addHandler(ch)
logger.info(args)
if args.memory_type == 'gru':
    from torch_geometric.nn import  TGNMemory
elif args.memory_type == 'mlp':
    from tgn_mlp import TGNMemory
else: raise NameError('wrong memory function type, must be mlp or gru')


device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
path = osp.join(osp.abspath(''), 'data')
dataset = JODIEDataset(path, name=args.data)
data = dataset[0].to(device)

# Ensure to only sample actual destination nodes as negatives.
min_dst_idx, max_dst_idx = int(data.dst.min()), int(data.dst.max())

train_data, val_data, test_data = data.train_val_test_split(
    val_ratio=0.15, test_ratio=0.15)

# raw+svd
data_process_type = args.data_process_type

if data_process_type == 'svd':
    num_modes = args.num_modes
    mode = findMode(train_data, num_modes)
    msg_dim = num_modes
elif data_process_type == 'raw':
    msg_dim = data.msg.size(-1)
elif data_process_type == 'raw+svd':
    num_modes = args.num_modes
    mode = findMode(train_data, num_modes)
    msg_dim = num_modes + data.msg.size(-1)
else: raise NameError('wrong data preprocessing type')
    
# Helper vector to map global node indices to local ones.
assoc = torch.empty(data.num_nodes, dtype=torch.long, device=device)
    
memory_dim, time_dim, embedding_dim = args.memory_dim, args.time_dim, args.embedding_dim


memory = TGNMemory(
    data.num_nodes,
    msg_dim,
    memory_dim,
    time_dim,
    message_module=IdentityMessage(msg_dim, memory_dim, time_dim),
    aggregator_module=LastAggregator(),
).to(device)

gnn = GraphAttentionEmbedding(
    in_channels=memory_dim,
    out_channels=embedding_dim,
    msg_dim=msg_dim,
    time_enc=memory.time_enc,
    num_nodes=data.num_nodes,
).to(device)

neighbor_loader = LastNeighborLoader(data.num_nodes, size=10, device=device)
link_pred = LinkPredictor(in_channels=embedding_dim).to(device)
optimizer = torch.optim.AdamW(
    set(memory.parameters()) | set(gnn.parameters())
    | set(link_pred.parameters()), lr=0.0001)
criterion = torch.nn.BCEWithLogitsLoss()



# pretrain and pretest to generate d_list_batch
loss_rec = []
loss, d_list_batch = pre_train()
loss_rec.append(loss)
val_ap, val_auc, d_list_batch_val = pre_test(val_data)
test_ap, test_auc, d_list_batch_test = pre_test(test_data)


memory.load_state_dict(torch.load(MODEL_SAVE_PATH)['memory_state_dict'])
gnn.load_state_dict(torch.load(MODEL_SAVE_PATH)['gnn_state_dict'])

from utils.utils import MLP

# create results array
reps = args.n_runs
test_auc_best = np.zeros(reps)

for r in range(reps):
    max_ap, arg_max_epoch = 0.0, 0
    
    decoder = MLP(memory_dim, drop=0.1).to(device)
    decoder_optimizer = torch.optim.AdamW(decoder.parameters(), lr=args.lr)
    decoder_loss_criterion = torch.nn.BCELoss()


    loss_rec = []
    for epoch in range(1, args.num_epoch + 1):
        loss = train_node_class(d_list_batch)
        logger.info(f'  Epoch: {epoch:02d}, Loss: {loss:.4f}')
        loss_rec.append(loss)
        pred_list = test_node_class(val_data, d_list_batch_val)
        preds = torch.cat(pred_list)
        logger.info(f'  val auc: {roc_auc_score(val_data.y.cpu(), preds):.4f}')
        pred_list = test_node_class(test_data, d_list_batch_test)
        preds = torch.cat(pred_list)
        test_auc = roc_auc_score(test_data.y.cpu(), preds)
        logger.info(f'  test auc: {test_auc:.4f}')
        test_auc_best[r] = test_auc
    logger.info(f' === Best final auc: {test_auc_best[r]:.4f} at run: {r+1:02d}')

logger.info(f'Old nodes: average auc: {np.mean(test_auc_best):.4f} after {reps} runs')
logger.info(f'Old nodes: std of auc: {np.std(test_auc_best):.4f} after {reps} runs')





    
