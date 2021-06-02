# Multi Clustered Federated Learning


## Setup
```
# make directory
$ mkdir data
$ mkdir logs

# install packages
$ pip install -r requirements.txt
```

## Download Dataset

### MNIST
```
$ mkdir data/mnist
$ cd data/mnist
$ wget http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz
$ wget http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz
$ wget http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz
$ wget http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz
$ cd ../../
```

#### CIFAR-10
```
$ mkdir data/cifar10
$ cd data/cifar10
$ wget https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
$ tar -zxvf cifar-10-python.tar.gz
$ rm cifar-10-python.tar.gz
$ cd ../../
```

## Training

### Option
- seed : (int) random seed.
- device: (str) cuda | cpu (e.g. cpu, cuda, cuda:0, ...)
- num_workers: (int) the number of process for dataloader. default: 0.
- n_clients: (int) the number of clients.
- n_noise: (int) the number of adversarial clients.
- n_rounds: (int) total number of training rounds.
- batch_size: (int) local update's batch size.
- local_epoch: (int) local update's epoch size.
- lr: (float) learning rate for SGD.
- adv: (str) type of adversarial setting. clean | random | flip | prand.
- thresh: (float) dissimilarity threshold for CFL and MCFL. default: 0.2.
- iid: (bool) data distribution type.
- parallel: (bool) Run local update in parallel. Local updates will be completed faster. Note: Even if the process and random seed are the same, changing the script may result in a loss of reproducibility.
- threads: (int) the number of process for local update in parallel.

For details on the other options, please check the corresponding config for your training script. (e.g. `config/fedavg_mnist.yaml` for `experiments/fedavg_mnist.py`)

### FedAvg
```
cd experiments
# NonIID MNIST on random
python fedavg_mnist.py adv=random iid=False
# IID CIFAR-10 on clean
python fedavg_cifar.py adv=clean iid=True
```

### CFL
```
cd experiments
# NonIID MNIST on flip
python cfl_mnist.py adv=flip iid=False
# IID CIFAR-10 on prand (the number of adversarial clients is 40)
python cfl_cifar.py adv=prand iid=True n_noise=40
```

### MCFL
```
cd experiments
# NonIID MNIST on flip
python mcfl_mnist.py adv=flip iid=False n_rounds=700
# IID CIFAR-10 on prand (the number of adversarial clients is 40)
python mcfl_cifar.py adv=prand iid=True n_noise=40 n_rounds=1200
```

option:
- alpha: (float) regurarization parameter.
- alpha_scheduler: (str) regularization scheduler. cos | static | linear.
- base_n_cls: (int) max clueter size.

### CFL w/ Global Regularization
```
cd experiments
# NonIID MNIST on flip
python gncfl_mnist.py adv=flip iid=False n_rounds=700
```