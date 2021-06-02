from clients.base import BaseClient

class LanguageClient(BaseClient):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def local_update(self, epoch):
        self.model.train()
        for i in range(epoch):
            hidden_state = self.model.init_hidden(self.dataloader.batch_size)
            for inputs, targets in self.dataloader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                self.optimizer.zero_grad()

                hidden_state = self.model.repackage_hidden(hidden_state)
                logits, hidden_state = self.model(inputs, hidden_state)
                loss = self.criterion(logits, targets)

                loss.backward()
                self.optimizer.step()

                if self.scheduler is not None:
                    self.scheduler.step()
