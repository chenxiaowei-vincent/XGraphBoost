# XGNNBOOST

XGBOOST is an algorithm combining GNN and XGBOOST, which can introduce the machine learning algorithm XGBOOST under the existing GNN network architecture to improve the algorithm capability.The GNN used in this paper includes DMPNN, GGNN and GCN

## Installation

Create a conda environment :
```shell script
conda create env -f environment.yaml
```


```shell script
conda activate my-rdkit-env
```

## train--dmpnn_xgboost
```shell script
cd /home/xgraphboost/dmpnn_xgboost
python train.py --protein delaney_ESOL --dataset_type regression --save_dir delaney_ESOL_r --epoch 200
```
## train--gcn_xgboost
```shell script
cd /home/xgraphboost/gcn_ggnn_xgboost
python gcn_xgb_morgan.py --folder benchmark_molnet/molnet_random_bbbp_c --type classification
```

## train--ggnn_xgboost
```shell script
cd /home/xgraphboost/gcn_ggnn_xgboost
python xgb_ggnn.py --folder benchmark_molnet/molnet_random_bbbp_c --type classification```






