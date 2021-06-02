import os
import argparse
import datetime
from copy import deepcopy
import traceback

import numpy as np
import torch
import torch.nn as nn
import torch.multiprocessing as mp
from torch.optim import SGD
from torch.utils.data import DataLoader
from omegaconf import OmegaConf
import ruamel.yaml
yaml = ruamel.yaml.YAML()
from loguru import logger
import tensorboardX as tbx
from tqdm import tqdm
import itertools

import _init_paths

from clients.base import BaseClient
from servers.cfl import ByzantineClusteredFederatedLearning as CFL
from datasets.cifar import CIFARDataset
from utils.general import seed_everything, get_lr, load_conf
from utils.data import get_cifar10_data
from utils.data_split import sorted_split, random_split

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 5)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d((2, 2))
        self.conv2 = nn.Conv2d(32, 32, 5)
        self.fc1 = nn.Linear(800, 256)
        self.fc2 = nn.Linear(256, 10)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x
        


def setup_dataset(cfg):
    # MNISTデータの読み込み
    train_img, train_label, test_img, test_label = get_cifar10_data("../data/cifar10")
    img_channel = 3

    # データの分割
    n_shards = cfg.n_clients * cfg.shards_per_client
    if cfg.iid:
        train_img, train_label = random_split(train_img, train_label, n_shards=n_shards)
    else:
        train_img, train_label = sorted_split(train_img, train_label, n_shards=n_shards)
    assert n_shards == len(train_img)

    clients_shards_indice = np.split(np.random.choice(n_shards, n_shards, replace=False), cfg.n_clients)
    train_img = [np.concatenate(train_img[idx], axis=0) for idx in clients_shards_indice]
    train_label = [np.concatenate(train_label[idx], axis=0) for idx in clients_shards_indice]
    assert len(train_img) == cfg.n_clients

    return train_img, train_label, test_img, test_label, img_channel


def setup_client(cfg, train_img, train_label, img_channel):
    # -- client setup --
    clients = []
    for images, labels in zip(train_img, train_label):
        model = CNN()
        optimizer = SGD(model.parameters(), lr=cfg.lr)
        criterion = torch.nn.CrossEntropyLoss()
        dataset = CIFARDataset(images, labels, training=True)
        dataloader = DataLoader(
            dataset,
            batch_size=cfg.batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=0,
            drop_last=True
        )
        client = BaseClient(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            dataloader=dataloader,
        )
        clients.append(client)
    return clients

def local_update(args):
    client=args[0]
    epoch=args[1]
    client.local_update(epoch=epoch)

def main(cfg_path):
    logger.debug(f"load config : {cfg_path}")
    cfg = load_conf(cfg_path)
    logger.debug('\n' + OmegaConf.to_yaml(cfg))

    logger.debug(f"set seed : {cfg.seed}")
    seed_everything(cfg.seed)

    exp_name = os.path.splitext(os.path.basename(__file__))[0]
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    logdir = os.path.join("../logs/", exp_name)
    os.makedirs(logdir, exist_ok=True)
    logdir = os.path.join(logdir, timestamp)
    os.makedirs(logdir, exist_ok=True)

    logger.debug(f"make log directory : {logdir}")
    logger.add(os.path.join(logdir, "log.txt"), level="DEBUG")

    writer = tbx.SummaryWriter(logdir)

    with open(os.path.join(logdir, "config.yaml"), 'w') as f:
        yaml.dump(dict(cfg), f)

    logger.debug(f"setup dataset")

    train_img, train_label, val_img, val_label, img_channel = setup_dataset(cfg)
    logger.debug(f"setup client")
    clients = setup_client(cfg, train_img, train_label, img_channel)
    logger.debug(f"num of client : {len(clients)}")

    n_noise = cfg.n_noise
    logger.debug(f"The number of Noise Client is {n_noise}.")
    faild_label_pair = np.array([6, 7, 5, 8, 9, 2, 0, 1, 3, 4]) # ノイズクライアントの誤りペア
    for i in range(n_noise):
        if cfg.adv == 'zero':
            clients[i].dataloader.dataset.labels = np.zeros_like(clients[i].dataloader.dataset.labels)
        elif cfg.adv == 'rand':
            clients[i].dataloader.dataset.labels = np.random.randint(0, 10, clients[i].dataloader.dataset.labels.shape)
        elif cfg.adv == 'flip':
            clients[i].dataloader.dataset.labels = faild_label_pair[clients[i].dataloader.dataset.labels]
        elif cfg.adv == 'prand':
            percentage = np.random.rand() # ノイズの割合
            n_noise_data = int(percentage * len(clients[i].dataloader.dataset.labels))
            noise_index = np.random.choice(np.arange(len(clients[i].dataloader.dataset.labels)), n_noise_data, replace=False)
            clients[i].dataloader.dataset.labels[noise_index] = np.random.randint(0, 10, clients[i].dataloader.dataset.labels[noise_index].shape)
    
    logger.debug(f"setup server")
    # -- server setup --
    val_dataloader = DataLoader(
        CIFARDataset(val_img, val_label, training=False),
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
    )

    server = CFL(
        model=deepcopy(clients[0].model),
        clients=clients,
        criterion=torch.nn.CrossEntropyLoss(),
        dataloader=val_dataloader)

    history = np.zeros((cfg.n_rounds, 5)) # loss, accuracy, xsim, n_bengin, n_adv
    logger.debug("start training")
    logger.debug(f"| {'step':^10} | {'loss':^12} | {'accuracy':^12} | {'lr':^12} | {'xsim':^8} | {'bengin':^6} | {'adv':^6}")

    # analysis
    similarity_matrix = np.zeros((cfg.n_rounds, len(clients), len(clients)))

    if cfg.adv == 'clean':
        server.benign = np.arange(n_noise, 100)
        server.adv = np.arange(n_noise)

    for t in range(cfg.n_rounds):
        samples = [client for client in clients]

        # send parameter from server to client
        for client in samples:
            client.model.load_state_dict(server.model.state_dict())
        
        
        # device: CPU -> GPU
        for client in samples:
            client.to(cfg.device)

        # local update
        if cfg.parallel:
            with mp.Pool(min(cfg.threads, mp.cpu_count())) as p:
                with tqdm(total=len(samples), leave=False) as tq:
                    for _ in p.imap_unordered(local_update, zip(samples,[cfg.local_epoch] * len(samples))):
                        tq.update(1)
        else:
            for client in tqdm(samples, leave=False):
                client.local_update(epoch=cfg.local_epoch)

        # device: GPU -> CPU
        for client in samples:
            client.to("cpu")

        xsim, simmat, cls1, cls2 = server.clustering(thresh=cfg.thresh)

        similarity_matrix[t] = simmat

        n_benign = len(server.benign)
        n_adv = len(server.adv)

        server.aggregate()
        server.to(cfg.device)
        loss, accuracy = server.eval()
        server.to("cpu")

        lr = get_lr(clients[0].optimizer)

        # log_param
        writer.add_scalar('loss', loss, t)
        writer.add_scalar('accuracy', accuracy, t)

        history[t, 0] = loss
        history[t, 1] = accuracy
        history[t, 2] = xsim
        history[t, 3] = n_benign
        history[t, 4] = n_adv

        logger.debug(f"| {t:>10} | {round(loss, 7):>12} | {round(accuracy, 7):>12} | {round(lr, 7):>12} | {round(xsim, 5):>8} | {n_benign:>6} | {n_adv:>6}")

    np.save(os.path.join(logdir, 'history.npy'), history)
    np.save(os.path.join(logdir, 'simmat.npy'), similarity_matrix)

if __name__ == "__main__":
    try:
        mp.set_start_method('spawn')
        cfg_path = "../config/cfl_cifar.yaml"
        main(cfg_path)
    except Exception as e:
        logger.error(traceback.format_exc())