# SWINIT: Spectral Transform Forms Scalable Transformer [[arXiv](https://arxiv.org/abs/2111.07602)] 


![](figures/fig1.png)	




## Introduction

Many real-world relational systems, such as social networks and biological systems, contain dynamic interactions. When learning dynamic graph representation, it is essential to employ sequential temporal information and geometric structure. Mainstream work achieves topological embedding via message passing networks (e.g., GCN, GAT). The temporal evolution, on the other hand, is conventionally expressed via memory units (e.g., LSTM or GRU) that possess convenient information filtration in a gate mechanism. Though, such a design prevents large-scale input sequence due to the over-complicated encoding. This work learns from the philosophy of self-attention and proposes an efficient spectral-based neural unit that employs informative long-range temporal interaction. The developed spectral window unit (SWINIT) model predicts scalable dynamic graphs with assured efficiency. The architecture is assembled with a few simple effective computational blocks that constitute randomized SVD, MLP, and graph Framelet convolution. The SVD plus MLP module encodes the long-short-term feature evolution of the dynamic graph events. A fast framelet graph transform in the framelet convolution embeds the structural dynamics. Both strategies enhance the model ability on scalable analysis. In particular, the iterative SVD approximation shrinks the computational complexity of attention to $\mathcal{O}(Nd\log(d))$ for the dynamic graph with $N$ edges and $d$ edge features, and the multiscale transform of framelet convolution allows sufficient scalability in the network training. Our SWINIT achieves state-of-the-art performance on a variety of online continuous-time dynamic graph learning tasks, while compared to baseline methods, the number of its learnable parameters reduces by up to seven times.


#### Paper link: [Spectral Transform Forms Scalable Transformer](https://arxiv.org/abs/2111.07602)

## Requirements

To install requirements:

```
pip install -r requirements.txt
```
## Dataset and Preprocessing

### Download the public data
Download the sample datasets (eg. wikipedia and reddit) from
[here](http://snap.stanford.edu/jodie/) and store their csv files in a folder named
```data/```.

### Preprocess the data
We use the dense `npy` format to save the features in binary format. If edge features or nodes 
features are absent, they will be replaced by a vector of zeros. 
```{bash}
python utils/preprocess_data.py --data wikipedia --bipartite
python utils/preprocess_data.py --data reddit --bipartite
```



## Running the experiments


### Self-supervised learning using the link prediction task:
```{bash}
# Transductive learning on the wikipedia dataset
python  link_prediction_transductive.py --data wikipedia --drop_out 0.3 --num_modes 70 --memory_dim 150 --n_runs 1 

# Inductive learning on the wikipedia dataset
python link_prediction_inductive.py  --data wikipedia --drop_out 0.3  --memory_dim 150 --gpu 0 --early_stopper 10 --n_runs 1 --num_modes 70 
```

### Supervised learning on dynamic node classification 
(this requires a trained model from the self-supervised task, by eg. running the commands above):
```{bash}
# Node classification
python node_classification.py --drop_out 0.3 --num_modes 70 --memory_dim 150 --n_runs 1 

```





