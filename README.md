![Python >=3.5](https://img.shields.io/badge/Python->=3.5-blue.svg)
![PyTorch >=1.0](https://img.shields.io/badge/PyTorch->=1.0-yellow.svg)

# [Unsupervised Person Re-identification with Stochastic Training Strategy](https://arxiv.org/pdf/2108.06938.pdf)
This repository contains the implementation of [Unsupervised Person Re-identification with Stochastic Training Strategy](https://arxiv.org/pdf/2108.06938.pdf). Our code is based on [SPCL](https://github.com/yxgeee/SpCL).
 
## Prepare Datasets

```shell
cd examples/data
```
Download the datasets [DukeMTMC-reID](https://arxiv.org/abs/1609.01775), [Market-1501](https://drive.google.com/file/d/0B8-rUzbwVRk0c054eEozWG9COHM/view), [MSMT17](https://arxiv.org/abs/1711.08565).
Then unzip them under the directory like
```
/examples/data
├── dukemtmc
│   └── DukeMTMC-reID
├── market1501
│   └── Market-1501-v15.09.15
├── msmt17
│   └── MSMT17_V1
```
### Prepare Pre-trained Models
ImageNet-pretrained models for **ResNet-50** will be automatically downloaded in the python script.

## Training and test unsupervised model for person re-ID
We utilize 4 GTX-1080TI GPUs for training. **Note that**

+ use `--eps 0.7` (default) for MSMT17, and `--eps 0.5` for DukeMTMC-reID, Market-1501;

```shell
# DukeMTMC-reID
CUDA_VISIBLE_DEVICES=0,1,2,3 python examples/spcl_train_usl.py -d dukemtmc --eps 0.5 --logs-dir logs/duke_resnet50

# Market-1501
CUDA_VISIBLE_DEVICES=0,1,2,3 python examples/spcl_train_usl.py -d market1501 --eps 0.5 --logs-dir logs/market_resnet50

# MSMT17_V1
CUDA_VISIBLE_DEVICES=0,1,2,3 python examples/spcl_train_usl.py -d msmt17 --eps 0.7 --logs-dir logs/msmt_resnet50
```
