# Code for "Supervision Adaptation Balances In-Distribution Generalization and Out-of-Distribution Detection"

## requirement
* Python 3.8
* Pytorch 1.12
* scikit-learn
* tqdm
* pandas
* scipy

## Datasets
### In-distribution Datasets
* [CIFAR10](https://www.cs.toronto.edu/~kriz/cifar.html)
* [CIFAR100](https://www.cs.toronto.edu/~kriz/cifar.html)

Our codes will download the two in-distribution datasets automatically.

### Out-of-Distribtion Datasets
The following four out-of-distribution datasets are provided by [ODIN](https://github.com/ShiyuLiang/odin-pytorch)
* [TinyImageNet](https://www.dropbox.com/s/kp3my3412u5k9rl/Imagenet_resize.tar.gz)
* [LSUN](https://www.dropbox.com/s/moqh2wh8696c3yl/LSUN_resize.tar.gz)
* [iSUN](https://www.dropbox.com/s/ssz7qxfqae0cca5/iSUN.tar.gz)

Each out-of-distribution dataset should be put in the corresponding subdir in [./data](./data)
