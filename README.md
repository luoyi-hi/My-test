# FaST: 

## 1.实验细节

### 1.1数据集描述

我们使用的 CA 数据集来源于由加利福尼亚交通局（CalTrans）维护的性能测量系统（PeMS），具体来源于参考文献 [1]。圣地亚哥（SD）和大洛杉矶地区（GLA）是从 CA 数据集中选取的两个具有代表性的子区域，分别包含 716 个和 3834 个传感器。所有传感器的元数据信息汇总如表 1 所示:

#### Table 1: Sensor Metadata Description

| Attribute | Description                                   | Possible Range of Values              |
| --------- | --------------------------------------------- | ------------------------------------- |
| ID        | The identifier of a sensor in PeMS            | 6 to 9 digits number                  |
| Lat       | The latitude of a sensor                      | Real number                           |
| Lng       | The longitude of a sensor                     | Real number                           |
| District  | The district of a sensor in PeMS              | 3, 4, 5, 6, 7, 8, 10, 11, 12          |
| County    | The county of a sensor in California          | String                                |
| Fwy       | The highway where a sensor is located         | String starts with 'I', 'US', or 'SR' |
| Lane      | The number of lanes where a sensor is located | 1, 2, 3, 4, 5, 6, 7, 8                |
| Type      | The type of a sensor                          | Mainline                              |
| Direction | The direction of the highway                  | N, S, E, W                            |

我们所使用的是2019年的CA数据集，先通过滑动窗口得到所有样本，然后样本按6:2:2的比例进行划分，从而得到训练集，验证集和测试集，其中数据集的统计可以参考表2：

#### Table 2: **Dataset statistics**

| Data | #nodes | Time interval | Time range           | 标准差 | 均值   | 使用特征     |
| ---- | ------ | ------------- | -------------------- | ------ | ------ | ------------ |
| SD   | 716    | 15 minute     | [1/1/2019, 1/1/2020) | 184.02 | 244.31 | traffic flow |
| GLA  | 3,834  | 15 minute     | [1/1/2019, 1/1/2020) | 187.77 | 276.82 | traffic flow |
| CA   | 8,600  | 15 minute     | [1/1/2019, 1/1/2020) | 177.12 | 237.39 | traffic flow |

想要得到更多的数据集细节信息，请参考文献[1]。

**Reference**

[1] Xu Liu, Yutong Xia, Yuxuan Liang, Junfeng Hu, Yiwei Wang, Lei Bai, Chao Huang, Zhenguang Liu, Bryan Hooi, and Roger Zimmermann. 2023. LargeST: A Benchmark Dataset for Large-Scale Traffic Forecasting. In The Annual Conference on Neural Information Processing Systems. New Orleans, LA, USA.

### 1.2数据准备

数据集可以从该链接下载：https://www.kaggle.com/datasets/liuxu77/largest，该链接一共有7个文件，为了复现我们的实验结果，您只需下载：“ca_his_raw_2019.h5”、“ca_meta.csv”、“ca_rn_adj.npy”这三个文件

将下载好的数据解压放到row_dataset目录下，然后按顺序使用下面的命令生成模型训练所需的流量数据。

```sh
python row_dataset/generate_data.py

python row_dataset/generate_data_for_training.py --dataset ca --years 2019
python row_dataset/generate_data_for_training.py --dataset gla --years 2019
python row_dataset/generate_data_for_training.py --dataset sd --years 2019

python row_dataset/process_adj.py

python row_dataset/generate_idx.py
```

### 1.3数据介绍

生成的数据会存放在BasicTS-master/datasets目录下，其中每个数据目录下的“his.npz”存放了数据的原始流量特征，以及对应的日特征和周特征，“adj_mx.pkl”是数据对应的邻接矩阵，desc.json存储了数据的信息。其它形如“{input_len}_{output_len}”的文件夹，里面存放了其对应预测长度的训练集，验证集和测试集的样本索引，其中每个预测步数对应的训练集，验证集，测试集的样本数参考表3：

#### Table 3: The number of training, validation, and test samples for each forecast horizon

| Forecast Horizon | Number of Training Samples | **Number of Validation Samples** | Number of Test Samples |
| ---------------- | -------------------------- | -------------------------------- | ---------------------- |
| 48               | 20938                      | 6979                             | 6980                   |
| 96               | 20909                      | 6969                             | 6971                   |
| 192              | 20851                      | 6950                             | 6952                   |
| 672              | 20563                      | 6854                             | 6856                   |

### 1.4实验运行

我们的代码基于 BasicTS 实现，FaST模型采用了Adam优化器，初始学习率设为0.002，并添加了权重衰减参数0.0001以增强正则化效果。在FaST训练过程中，学习率调度策略使用了 `MultiStepLR`，在第10、20、30、40与50轮时进行衰减，每次将当前学习率乘以0.5，从而实现多阶段的渐进式优化，有助于模型更稳定地收敛。所有方法的最大训练轮次为100，在验证集上使用早停策略确定最佳参数。所有实验使用MAE、RMSE和MAPE评估模型性能。所有实验在AMD EPYC 7532 @2.40GHz，NVIDIA RTX A6000 GPU（48GB），128GB RAM和Ubuntu 20.04的环境下，进行了实验。我们采用 PyTorch 2.2.1 作为默认的深度学习库，使用的python版本是3.11。

通过以下命令安装其它依赖：

```shell
pip install -r requirements.txt
```

请转到“BasicTS-master”目录下，然后可以使用以下命令来运行我们的模型：

```shell
# SD
python experiments/train_seed.py -c baselines/FaST/sd_96_48.py -g 0
python experiments/train_seed.py -c baselines/FaST/sd_96_96.py -g 0
python experiments/train_seed.py -c baselines/FaST/sd_96_192.py -g 0
python experiments/train_seed.py -c baselines/FaST/sd_96_672.py -g 0

# GLA
python experiments/train_seed.py -c baselines/FaST/gla_96_48.py -g 0
python experiments/train_seed.py -c baselines/FaST/gla_96_96.py -g 0
python experiments/train_seed.py -c baselines/FaST/gla_96_192.py -g 0
python experiments/train_seed.py -c baselines/FaST/gla_96_672.py -g 0

# CA
python experiments/train_seed.py -c baselines/FaST/ca_96_48.py -g 0
python experiments/train_seed.py -c baselines/FaST/ca_96_96.py -g 0
python experiments/train_seed.py -c baselines/FaST/ca_96_192.py -g 0
python experiments/train_seed.py -c baselines/FaST/ca_96_672.py -g 0
```

### 1.5FaST模型复现

由于空间限制，这里我们仅提供了在SD数据集上96预测48的模型参数，您可以使用这个模型参数从而复现我们论文中报告的结果。在“BasicTS-master”目录下执行以下命令：

``` shell
python experiments/evaluate.py -cfg  baselines/FaST/sd_96_48.py -ckpt Parameters_FaST/sd/96_48/FaST_best_val_MAE.pt -g 0
```

### 1.6基线复现

可以使用以下命令来复现基线模型：

```shell
# STID
# SD
python experiments/train_seed.py -c baselines/STID/sd_96_48.py -g 0
python experiments/train_seed.py -c baselines/STID/sd_96_96.py -g 0
python experiments/train_seed.py -c baselines/STID/sd_96_192.py -g 0
python experiments/train_seed.py -c baselines/STID/sd_96_672.py -g 0
# GLA
python experiments/train_seed.py -c baselines/STID/gla_96_48.py -g 0
python experiments/train_seed.py -c baselines/STID/gla_96_96.py -g 0
python experiments/train_seed.py -c baselines/STID/gla_96_192.py -g 0
python experiments/train_seed.py -c baselines/STID/gla_96_672.py -g 0
# CA
python experiments/train_seed.py -c baselines/STID/ca_96_48.py -g 0
python experiments/train_seed.py -c baselines/STID/ca_96_96.py -g 0
python experiments/train_seed.py -c baselines/STID/ca_96_192.py -g 0
python experiments/train_seed.py -c baselines/STID/ca_96_672.py -g 0

# DLinear
# SD
python experiments/train_seed.py -c baselines/DLinear/sd_96_48.py -g 0
python experiments/train_seed.py -c baselines/DLinear/sd_96_96.py -g 0
python experiments/train_seed.py -c baselines/DLinear/sd_96_192.py -g 0
python experiments/train_seed.py -c baselines/DLinear/sd_96_672.py -g 0
# GLA
python experiments/train_seed.py -c baselines/DLinear/gla_96_48.py -g 0
python experiments/train_seed.py -c baselines/DLinear/gla_96_96.py -g 0
python experiments/train_seed.py -c baselines/DLinear/gla_96_192.py -g 0
python experiments/train_seed.py -c baselines/DLinear/gla_96_672.py -g 0
# CA
python experiments/train_seed.py -c baselines/DLinear/ca_96_48.py -g 0
python experiments/train_seed.py -c baselines/DLinear/ca_96_96.py -g 0
python experiments/train_seed.py -c baselines/DLinear/ca_96_192.py -g 0
python experiments/train_seed.py -c baselines/DLinear/ca_96_672.py -g 0

# NHITS
# SD
python experiments/train_seed.py -c baselines/NHITS/sd_96_48.py -g 0
python experiments/train_seed.py -c baselines/NHITS/sd_96_96.py -g 0
python experiments/train_seed.py -c baselines/NHITS/sd_96_192.py -g 0
python experiments/train_seed.py -c baselines/NHITS/sd_96_672.py -g 0
# GLA
python experiments/train_seed.py -c baselines/NHITS/gla_96_48.py -g 0
python experiments/train_seed.py -c baselines/NHITS/gla_96_96.py -g 0
python experiments/train_seed.py -c baselines/NHITS/gla_96_192.py -g 0
python experiments/train_seed.py -c baselines/NHITS/gla_96_672.py -g 0
# CA
python experiments/train_seed.py -c baselines/NHITS/ca_96_48.py -g 0
python experiments/train_seed.py -c baselines/NHITS/ca_96_96.py -g 0
python experiments/train_seed.py -c baselines/NHITS/ca_96_192.py -g 0
python experiments/train_seed.py -c baselines/NHITS/ca_96_672.py -g 0

# CycleNet
# SD
python experiments/train_seed.py -c baselines/CycleNet/sd_96_48.py -g 0
python experiments/train_seed.py -c baselines/CycleNet/sd_96_96.py -g 0
python experiments/train_seed.py -c baselines/CycleNet/sd_96_192.py -g 0
python experiments/train_seed.py -c baselines/CycleNet/sd_96_672.py -g 0
# GLA
python experiments/train_seed.py -c baselines/CycleNet/gla_96_48.py -g 0
python experiments/train_seed.py -c baselines/CycleNet/gla_96_96.py -g 0
python experiments/train_seed.py -c baselines/CycleNet/gla_96_192.py -g 0
python experiments/train_seed.py -c baselines/CycleNet/gla_96_672.py -g 0
# CA
python experiments/train_seed.py -c baselines/CycleNet/ca_96_48.py -g 0
python experiments/train_seed.py -c baselines/CycleNet/ca_96_96.py -g 0
python experiments/train_seed.py -c baselines/CycleNet/ca_96_192.py -g 0
python experiments/train_seed.py -c baselines/CycleNet/ca_96_672.py -g 0

# DCRNN
# SD
python experiments/train_seed.py -c baselines/DCRNN/sd_96_48.py -g 0
python experiments/train_seed.py -c baselines/DCRNN/sd_96_96.py -g 0

# BigST
# SD
python experiments/train_seed.py -c baselines/BigST/sd_96_48.py -g 0
python experiments/train_seed.py -c baselines/BigST/sd_96_48_2.py -g 0
python experiments/train_seed.py -c baselines/BigST/sd_96_96.py -g 0
python experiments/train_seed.py -c baselines/BigST/sd_96_96_2.py -g 0
python experiments/train_seed.py -c baselines/BigST/sd_96_192.py -g 0
python experiments/train_seed.py -c baselines/BigST/sd_96_192_2.py -g 0
python experiments/train_seed.py -c baselines/BigST/sd_96_672.py -g 0
python experiments/train_seed.py -c baselines/BigST/sd_96_672_2.py -g 0

# STGCN
# SD
python experiments/train_seed.py -c baselines/STGCN/sd_96_48.py -g 0
python experiments/train_seed.py -c baselines/STGCN/sd_96_96.py -g 0
python experiments/train_seed.py -c baselines/STGCN/sd_96_192.py -g 0
python experiments/train_seed.py -c baselines/STGCN/sd_96_672.py -g 0

# GWNet
# SD
python experiments/train_seed.py -c baselines/GWNet/sd_96_48.py -g 0
python experiments/train_seed.py -c baselines/GWNet/sd_96_96.py -g 0
python experiments/train_seed.py -c baselines/GWNet/sd_96_192.py -g 0
python experiments/train_seed.py -c baselines/GWNet/sd_96_672.py -g 0


# SGP
# 请参考："https://github.com/Graph-Machine-Learning-Group/sgp"配置相关环境
# 切换到sgp目录
# 复制数据
python experiments/run_traffic_sgps_sd_96_48.py 
python experiments/run_traffic_sgps_sd_96_96.py 
python experiments/run_traffic_sgps_sd_96_192.py 
python experiments/run_traffic_sgps_sd_96_672.py 

# RPMixer
# 请参考："https://sites.google.com/view/rpmixer"配置相关环境
# 切换到RPMixer目录
python sd_96_48.py
python sd_96_96.py
python sd_96_192.py
python sd_96_672.py
# GLA
python gla_96_48.py
python gla_96_96.py
# CA
python ca_96_48.py

```



