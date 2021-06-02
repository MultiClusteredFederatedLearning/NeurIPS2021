import numpy as np
import random

def sorted_split(x, y, n_shards):
    sorted_index = np.argsort(y)
    x = x[sorted_index]
    y = y[sorted_index]
    shard_size = len(x) // n_shards
    init_indice = np.arange(0, len(x), shard_size)
    print(f"the number of shard : {len(init_indice)}")
    x = np.array([x[s:s+shard_size] for s in init_indice])
    y = np.array([y[s:s+shard_size] for s in init_indice])
    return x, y

def random_split(x, y, n_shards, shuffle=True):
    shard_size = len(x) // n_shards
    index = np.arange(len(y))
    if shuffle:
        random.shuffle(index)
        x = x[index]
        y = y[index]
    init_indice = np.arange(0, len(x), shard_size)
    print(f"the number of shard : {len(init_indice)}")
    x = np.array([x[s:s + shard_size] for s in init_indice])
    y = np.array([y[s:s + shard_size] for s in init_indice])
    return x, y