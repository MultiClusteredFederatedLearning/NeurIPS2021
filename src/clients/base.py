from copy import deepcopy
import numpy as np
import torch

class BaseClient:
    def __init__(self, model, optimizer, criterion, dataloader, scheduler=None):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.dataloader = dataloader

        self.scheduler = scheduler

        self.device = 'cpu'

    def local_update(self, epoch):
        self.model.train()
        for i in range(epoch):
            for inputs, targets in self.dataloader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                self.optimizer.zero_grad()

                logits = self.model(inputs)
                loss = self.criterion(logits, targets)

                loss.backward()
                self.optimizer.step()

    def receive_param(self, state_dict):
        self.model.load_state_dict(state_dict)
    
    def send_param(self):
        return self.model.state_dict()

    def __len__(self):
        return len(self.dataloader.dataset)

    def to(self, device):
        self.device = device
        self.model = self.model.to(device)

    def safety(self):
        for param in self.model.parameters():
            if param.isnan().any():
                return False
        return True