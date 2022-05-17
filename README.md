# Spectral Transformers for Dynamic Graph Learning
<!-- [[arXiv](https://arxiv.org/abs/2111.07602)]  -->


![](figures/fig1.png)	




## Introduction

This work establishes a fully-spectral framework to capture informative long-range temporal interactions in a dynamic system. We propose spectral transformer for predicting continuous-time dynamic graphs. Our model uses power method SVD and global graph framelet convolution to encode time-depending features and graph structure. The SVD serves as a high-order linear self-attention with determined propagation rules. The spectral transformer thus gains high expressivity in linear complexity by amending unbalanced energy of graph latent representations in the network. The framelet convolution in the second stage of the model establishes scalable and transferable geometric characterization for prediction. We examine the efficiency of the proposed model on a variety of online learning tasks. It achieves top performance with a reduced number of learnable parameters and faster propagation speed.


<!-- #### Paper link: [Spectral Transform Forms Scalable Transformer](https://arxiv.org/abs/2111.07602) -->

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





