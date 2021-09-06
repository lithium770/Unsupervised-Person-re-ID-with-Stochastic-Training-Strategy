![Python >=3.5](https://img.shields.io/badge/Python->=3.5-blue.svg)
![PyTorch >=1.0](https://img.shields.io/badge/PyTorch->=1.0-yellow.svg)

# Unsupervised Person Re-identification with Stochastic Training Strategy(https://arxiv.org/pdf/2108.06938.pdf)

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
## Training and test unsupervised model for person re-ID
*Example #1:* DukeMTMC-reID
```shell
CUDA_VISIBLE_DEVICES=0,1,2,3 python examples/spcl_train_usl.py -d dukemtmc --eps 0.5 --logs-dir logs/duke_resnet50
```
*Example #2:* Market-1501
```shell
CUDA_VISIBLE_DEVICES=0,1,2,3 python examples/spcl_train_usl.py -d market1501 --eps 0.5 --logs-dir logs/market_resnet50
```

*Example #3:* MSMT17_V1
```shell
CUDA_VISIBLE_DEVICES=0,1,2,3 python examples/spcl_train_usl.py -d msmt17 --eps 0.7 --logs-dir logs/market_resnet50
```
