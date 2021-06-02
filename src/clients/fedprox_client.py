from copy import deepcopy
import numpy as np
from clients.base import BaseClient

class FedProxClient(BaseClient):
    def __init__(self, model, optimizer, criterion, dataloader, scheduler=None):
        super().__init__(model, optimizer, criterion, dataloader, scheduler)

    def local_update(self, epoch):
        global_model = deepcopy(self.model)
        self.model.train()
        for i in range(epoch):
            for inputs, targets in self.dataloader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                self.optimizer.zero_grad()

                logits = self.model(inputs)
                loss = self.criterion(logits, targets)

                loss.backward()
                self.optimizer.step(global_model.parameters())

    def receive_param(self, state_dict):
        self.model.load_state_dict(state_dict)