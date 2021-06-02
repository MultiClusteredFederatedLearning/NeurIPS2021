import os
import os.path as ops
import urllib.request
import gzip
import numpy as np
import pickle


def get_mnist_data(datadir):
    dataroot = 'http://yann.lecun.com/exdb/mnist/'
    key_file = {
        'train_img': 'train-images-idx3-ubyte.gz',
        'train_label': 'train-labels-idx1-ubyte.gz',
        'test_img': 't10k-images-idx3-ubyte.gz',
        'test_label': 't10k-labels-idx1-ubyte.gz'
    }
    os.makedirs(datadir, exist_ok=True)

    for key, filename in key_file.items():
        if ops.exists(ops.join(datadir, filename)):
            print(f"already downloaded : {filename}")
        else:
            urllib.request.urlretrieve(ops.join(dataroot, filename),
                                       ops.join(datadir, filename))

    with gzip.open(ops.join(datadir, key_file["train_img"]), "rb") as f:
        train_img = np.frombuffer(f.read(), np.uint8, offset=16)
    train_img = train_img.reshape(-1, 784)

    with gzip.open(ops.join(datadir, key_file["train_label"]), "rb") as f:
        train_label = np.frombuffer(f.read(), np.uint8, offset=8)

    with gzip.open(ops.join(datadir, key_file["test_img"]), "rb") as f:
        test_img = np.frombuffer(f.read(), np.uint8, offset=16)
    test_img = test_img.reshape(-1, 784)

    with gzip.open(ops.join(datadir, key_file["test_label"]), "rb") as f:
        test_label = np.frombuffer(f.read(), np.uint8, offset=8)

    return train_img, train_label, test_img, test_label

def get_cifar10_data(datadir):
    datadir = os.path.join(datadir, "cifar-10-batches-py")

    # == train ==
    train_img = []
    train_label = []
    for i in range(1, 6):
        path = os.path.join(datadir, f"data_batch_{i}")
        with open(path, 'rb') as f:
            data = pickle.load(f, encoding="latin-1")
        train_img.append(data['data'])
        train_label.append(data['labels'])
    train_img = np.concatenate(train_img, axis=0).reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    train_label = np.concatenate(train_label, axis=0)

    # == test ==
    path = os.path.join(datadir, f"test_batch")
    with open(path, 'rb') as f:
        data = pickle.load(f, encoding="latin-1")
    test_img = np.array(data['data']).reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    test_label = np.array(data['labels'])
    
    return train_img, train_label, test_img, test_label

def get_cifar100_data(datadir):
    datadir = os.path.join(datadir, "cifar-100-python")
    # == train =
    path = os.path.join(datadir, "train")
    with open(path, 'rb') as f:
        data = pickle.load(f, encoding="latin-1")
    train_img = np.array(data['data']).reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    train_label = np.array(data['fine_labels'])

    # == test ==
    path = os.path.join(datadir, "test")
    with open(path, 'rb') as f:
        data = pickle.load(f, encoding="latin-1")
    test_img = np.array(data['data']).reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    test_label = np.array(data['fine_labels'])

    return train_img, train_label, test_img, test_label