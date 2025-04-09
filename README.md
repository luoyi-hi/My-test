# FaST: 

## 1. Experimental Details

### 1.1 Dataset Description

The CA dataset we use is from the Performance Measurement System (PeMS) maintained by the California Department of Transportation (CalTrans). For specific details, refer to the literature [1]. The San Diego (SD) and Greater Los Angeles (GLA) areas are two representative subregions selected from the CA dataset, containing 716 and 3834 sensors, respectively. 

The dataset can be downloaded from the following link: https://www.kaggle.com/datasets/liuxu77/largest. The link contains seven files. To reproduce our experiment results, you need to download the following three files: “ca_his_raw_2019.h5”, “ca_meta.csv”, “ca_rn_adj.npy”.

Install environment dependencies using the following command:

```shell
pip install -r requirements.txt
```

Unzip the downloaded data into the "row_dataset" directory. Then, sequentially use the following commands to generate the traffic data required for model training:

```shell
python DataPipeline/generate_data.py

python DataPipeline/generate_data_for_training.py --dataset ca --years 2019
python DataPipeline/generate_data_for_training.py --dataset gla --years 2019
python DataPipeline/generate_data_for_training.py --dataset sd --years 2019

python DataPipeline/process_adj.py

python DataPipeline/generate_idx.py
```
The statistics of the dataset are summarized in Table 1:

#### Table 1: **Dataset statistics**

| Data | #nodes | Time interval | Time range           | Std    | Mean   | Features     |
| ---- | ------ | ------------- | -------------------- | ------ | ------ | ------------ |
| SD   | 716    | 15 minute     | [1/1/2019, 1/1/2020) | 184.02 | 244.31 | traffic flow |
| GLA  | 3,834  | 15 minute     | [1/1/2019, 1/1/2020) | 187.77 | 276.82 | traffic flow |
| CA   | 8,600  | 15 minute     | [1/1/2019, 1/1/2020) | 177.12 | 237.39 | traffic flow |

For more dataset details, refer to literature [1].

**Reference**

[1] Xu Liu, Yutong Xia, Yuxuan Liang, Junfeng Hu, Yiwei Wang, Lei Bai, Chao Huang, Zhenguang Liu, Bryan Hooi, and Roger Zimmermann. 2023. LargeST: A Benchmark Dataset for Large-Scale Traffic Forecasting. In The Annual Conference on Neural Information Processing Systems. New Orleans, LA, USA.

### 1.2 Data Generation for Model Training
We use the 2019 SD, GLA, and CA datasets. First, we obtain all samples through a sliding window, then split the samples into training, validation, and test sets in a 6:2:2 ratio.
The generated data will be stored in the “BasicTS-master/datasets” directory. In each data directory, the “his.npz” file contains the raw traffic flow features, as well as the corresponding daily and weekly features. The “adj_mx.pkl” file contains the adjacency matrix for the data, and “desc.json” stores the data information. Other folders, such as “{input_len}_{output_len}”, store the sample indices for the training, validation, and test sets for the corresponding forecast length.


### 1.3 Experimental Setup

Our model is implemented based on the "BasicTS" framework. The FaST model uses the Adam optimizer with an initial learning rate of 0.002, and a weight decay parameter of 0.0001 for regularization. During the FaST training process, the learning rate scheduling strategy uses `MultiStepLR`, which decays the learning rate by a factor of 0.5 at the 10th, 20th, 30th, 40th, and 50th epochs for multi-stage progressive optimization, helping the model converge more stably. The maximum training epochs for all methods are set to 100, with early stopping on the validation set to determine the best parameters. The performance is evaluated using MAE, RMSE, and MAPE. All experiments are conducted in an environment with an AMD EPYC 7532 @2.40GHz, NVIDIA RTX A6000 GPU (48GB), 128GB RAM, and Ubuntu 20.04. The default deep learning library is PyTorch 2.2.1, and the Python version is 3.11.

### 1.4 Training FaST model
Go to the “BasicTS-master” directory and use the following commands to run our model:

```shell
# FaST on SD dataset
python experiments/train_seed.py -c FaST/sd_96_48.py -g 0
python experiments/train_seed.py -c FaST/sd_96_96.py -g 0
python experiments/train_seed.py -c FaST/sd_96_192.py -g 0
python experiments/train_seed.py -c FaST/sd_96_672.py -g 0

# FaST on GLA dataset
python experiments/train_seed.py -c FaST/gla_96_48.py -g 0
python experiments/train_seed.py -c FaST/gla_96_96.py -g 0
python experiments/train_seed.py -c FaST/gla_96_192.py -g 0
python experiments/train_seed.py -c FaST/gla_96_672.py -g 0

# FaST on CA dataset
python experiments/train_seed.py -c FaST/ca_96_48.py -g 0
python experiments/train_seed.py -c FaST/ca_96_96.py -g 0
python experiments/train_seed.py -c FaST/ca_96_192.py -g 0
python experiments/train_seed.py -c FaST/ca_96_672.py -g 0
```

### 1.5 FaST Model Reproduction: Reproducing FaST's experiment results using our trained parameters

Owing to storage constraints, we currently provide only the model parameters trained on the SD dataset, which are sufficient for reproducing the corresponding results reported in this paper.

The trained parameters for other datasets will be uploaded to a public cloud drive upon the acceptance of this paper. To reproduce the SD results, execute the following command in the “BasicTS-master” directory:

``` shell
python experiments/evaluate.py -cfg  baselines/FaST/sd_96_48.py -ckpt Parameters_FaST/sd/96_48/FaST_best_val_MAE.pt -g 0
```

### 1.6 Baseline Reproduction

Use the following commands to reproduce baseline models:

```shell
# STID
# STID on SD dataset
python experiments/train_seed.py -c baselines/STID/sd_96_48.py -g 0
python experiments/train_seed.py -c baselines/STID/sd_96_96.py -g 0
python experiments/train_seed.py -c baselines/STID/sd_96_192.py -g 0
python experiments/train_seed.py -c baselines/STID/sd_96_672.py -g 0
# STID on GLA dataset
python experiments/train_seed.py -c baselines/STID/gla_96_48.py -g 0
python experiments/train_seed.py -c baselines/STID/gla_96_96.py -g 0
python experiments/train_seed.py -c baselines/STID/gla_96_192.py -g 0
python experiments/train_seed.py -c baselines/STID/gla_96_672.py -g 0
# STID on CA dataset
python experiments/train_seed.py -c baselines/STID/ca_96_48.py -g 0
python experiments/train_seed.py -c baselines/STID/ca_96_96.py -g 0
python experiments/train_seed.py -c baselines/STID/ca_96_192.py -g 0
python experiments/train_seed.py -c baselines/STID/ca_96_672.py -g 0

# DLinear
# DLinear on SD dataset
python experiments/train_seed.py -c baselines/DLinear/sd_96_48.py -g 0
python experiments/train_seed.py -c baselines/DLinear/sd_96_96.py -g 0
python experiments/train_seed.py -c baselines/DLinear/sd_96_192.py -g 0
python experiments/train_seed.py -c baselines/DLinear/sd_96_672.py -g 0
# DLinear on GLA dataset
python experiments/train_seed.py -c baselines/DLinear/gla_96_48.py -g 0
python experiments/train_seed.py -c baselines/DLinear/gla_96_96.py -g 0
python experiments/train_seed.py -c baselines/DLinear/gla_96_192.py -g 0
python experiments/train_seed.py -c baselines/DLinear/gla_96_672.py -g 0
# DLinear on CA dataset
python experiments/train_seed.py -c baselines/DLinear/ca_96_48.py -g 0
python experiments/train_seed.py -c baselines/DLinear/ca_96_96.py -g 0
python experiments/train_seed.py -c baselines/DLinear/ca_96_192.py -g 0
python experiments/train_seed.py -c baselines/DLinear/ca_96_672.py -g 0

# NHITS
# NHITS on SD dataset
python experiments/train_seed.py -c baselines/NHiTS/sd_96_48.py -g 0
python experiments/train_seed.py -c baselines/NHiTS/sd_96_96.py -g 0
python experiments/train_seed.py -c baselines/NHiTS/sd_96_192.py -g 0
python experiments/train_seed.py -c baselines/NHiTS/sd_96_672.py -g 0
# NHITS on GLA dataset
python experiments/train_seed.py -c baselines/NHiTS/gla_96_48.py -g 0
python experiments/train_seed.py -c baselines/NHiTS/gla_96_96.py -g 0
python experiments/train_seed.py -c baselines/NHiTS/gla_96_192.py -g 0
python experiments/train_seed.py -c baselines/NHiTS/gla_96_672.py -g 0
# NHITS on CA dataset
python experiments/train_seed.py -c baselines/NHiTS/ca_96_48.py -g 0
python experiments/train_seed.py -c baselines/NHiTS/ca_96_96.py -g 0
python experiments/train_seed.py -c baselines/NHiTS/ca_96_192.py -g 0
python experiments/train_seed.py -c baselines/NHiTS/ca_96_672.py -g 0

# CycleNet
# CycleNet on SD dataset
python experiments/train_seed.py -c baselines/CycleNet/sd_96_48.py -g 0
python experiments/train_seed.py -c baselines/CycleNet/sd_96_96.py -g 0
python experiments/train_seed.py -c baselines/CycleNet/sd_96_192.py -g 0
python experiments/train_seed.py -c baselines/CycleNet/sd_96_672.py -g 0
# CycleNet on GLA dataset
python experiments/train_seed.py -c baselines/CycleNet/gla_96_48.py -g 0
python experiments/train_seed.py -c baselines/CycleNet/gla_96_96.py -g 0
python experiments/train_seed.py -c baselines/CycleNet/gla_96_192.py -g 0
python experiments/train_seed.py -c baselines/CycleNet/gla_96_672.py -g 0
# CycleNet on CA dataset
python experiments/train_seed.py -c baselines/CycleNet/ca_96_48.py -g 0
python experiments/train_seed.py -c baselines/CycleNet/ca_96_96.py -g 0
python experiments/train_seed.py -c baselines/CycleNet/ca_96_192.py -g 0
python experiments/train_seed.py -c baselines/CycleNet/ca_96_672.py -g 0

# DCRNN
# DCRNN on SD dataset
python experiments/train_seed.py -c baselines/DCRNN/sd_96_48.py -g 0
python experiments/train_seed.py -c baselines/DCRNN/sd_96_96.py -g 0

# BigST
# BigST on SD dataset
python experiments/train_seed.py -c baselines/BigST/sd_96_48.py -g 0
python experiments/train_seed.py -c baselines/BigST/sd_96_48_2.py -g 0
python experiments/train_seed.py -c baselines/BigST/sd_96_96.py -g 0
python experiments/train_seed.py -c baselines/BigST/sd_96_96_2.py -g 0
python experiments/train_seed.py -c baselines/BigST/sd_96_192.py -g 0
python experiments/train_seed.py -c baselines/BigST/sd_96_192_2.py -g 0
python experiments/train_seed.py -c baselines/BigST/sd_96_672.py -g 0
python experiments/train_seed.py -c baselines/BigST/sd_96_672_2.py -g 0

# STGCN
# STGCN on SD dataset
python experiments/train_seed.py -c baselines/STGCN/sd_96_48.py -g 0
python experiments/train_seed.py -c baselines/STGCN/sd_96_96.py -g 0
python experiments/train_seed.py -c baselines/STGCN/sd_96_192.py -g 0
python experiments/train_seed.py -c baselines/STGCN/sd_96_672.py -g 0

# GWNet
# GWNet on SD dataset
python experiments/train_seed.py -c baselines/GWNet/sd_96_48.py -g 0
python experiments/train_seed.py -c baselines/GWNet/sd_96_96.py -g 0
python experiments/train_seed.py -c baselines/GWNet/sd_96_192.py -g 0
python experiments/train_seed.py -c baselines/GWNet/sd_96_672.py -g 0


# SGP
# Please refer to: ‘https://github.com/Graph-Machine-Learning-Group/sgp’ to configure the relevant environment
# Switch to the sgp directory
# Copy data
cd BasicTS-master/baselines/sgp-main
# SGP on SD dataset
python experiments/run_traffic_sgps_sd_96_48.py 
python experiments/run_traffic_sgps_sd_96_96.py 
python experiments/run_traffic_sgps_sd_96_192.py 
python experiments/run_traffic_sgps_sd_96_672.py 

# RPMixer
# Please refer to: ‘https://sites.google.com/view/rpmixer’ to configure the relevant environment
# Switch to the RPMixer directory
cd BasicTS-master/baselines/RPMixer
# RPMixer on SD dataset
python sd_96_48.py
python sd_96_96.py
python sd_96_192.py
python sd_96_672.py
# RPMixer on GLA dataset
python gla_96_48.py
python gla_96_96.py
# RPMixer on CA dataset
python ca_96_48.py

```



