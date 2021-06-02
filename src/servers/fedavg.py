import random
import numpy as np
import copy
import torch
from collections import OrderedDict
from tqdm import tqdm

class FedAvg:
    def __init__(self, model, clients, criterion, dataloader):
        self.model = model
        self.clients = clients
        self.dataloader = dataloader
        self.criterion = criterion
        self.device = 'cpu'

    def aggregate(self, sample=None):
        # sample is not None, then FedProx aggregate
        if sample is None:
            sample = range(len(self.clients))
        
        sample = [k for k in sample if self.clients[k].safety()]

        total_size = sum([len(self.clients[k]) for k in sample])

        model_names = ["model"]
        for name in model_names:
            update_state = OrderedDict()
            for k in sample:
                client = self.clients[k]
                local_state = getattr(client, name).state_dict()
                weight = len(client) / total_size
                for key, value in local_state.items():
                    if not key in update_state.keys():
                        update_state[key] = value * weight
                    else:
                        update_state[key] += value * weight
            getattr(self, name).load_state_dict(update_state)

    def eval(self):
        self.model.eval()
        avg_loss = 0
        avg_accuracy = 0
        with torch.no_grad():
            for img, label in self.dataloader:
                label = label.to(self.device)
                img = img.to(self.device)
                logits = self.model(img)
                loss = self.criterion(logits, label)
                avg_loss += loss.item()

                pred = logits.argmax(dim=1, keepdim=True)
                avg_accuracy += pred.eq(label.view_as(pred)).sum().item()

        avg_loss = avg_loss / len(self.dataloader)
        avg_accuracy = 100. * avg_accuracy / len(self.dataloader.dataset)
        return avg_loss, avg_accuracy

    def to(self, device):
        self.device = device
        self.model.to(device)

    def __len__(self):
        return len(self.clients)

    def sampling(self, num_sample):
        return random.sample(range(len(self.clients)), num_sample)

class FedAvgLanguage(FedAvg):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def eval(self):
        self.model.eval()
        avg_loss = 0
        avg_accuracy = 0
        with torch.no_grad():
            hidden_state = self.model.init_hidden(self.dataloader.batch_size)
            for inputs, targets in tqdm(self.dataloader,total=len(self.dataloader), leave=False):
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                logits, hidden_state = self.model(inputs, hidden_state)
                loss = self.criterion(logits, targets)
                avg_loss += loss.item()

                pred = logits.argmax(dim=1, keepdim=True)
                avg_accuracy += pred.eq(targets.view_as(pred)).sum().item()

        avg_loss = avg_loss / len(self.dataloader)
        avg_accuracy = 100. * avg_accuracy / len(self.dataloader.dataset)
        return avg_loss, avg_accuracy