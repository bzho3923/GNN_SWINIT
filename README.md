# SWINIT: Spectral Transform Forms Scalable Transformer [[arXiv](https://arxiv.org/abs/2111.07602)] 


![](figures/fig1.png)	




## Introduction

Despite the plethora of different models for deep learning on graphs, few approaches have been proposed thus far for dealing with graphs that present some sort of dynamic nature (e.g. evolving features or connectivity over time).
 
In this paper, we present Temporal Graph Networks (TGNs), a generic, efficient framework for deep learning on dynamic graphs represented as sequences of timed events. Thanks to a novel combination of memory modules and graph-based operators, TGNs are able to significantly outperform previous approaches being at the same time more computationally efficient. 

We furthermore show that several previous models for learning on dynamic graphs can be cast as specific instances of our framework. We perform a detailed ablation study of different components of our framework and devise the best configuration that achieves state-of-the-art performance on several transductive and inductive prediction tasks for dynamic graphs.


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





