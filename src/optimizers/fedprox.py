import torch
from torch.optim import Optimizer

class FedProxSGD(Optimizer):
    def __init__(self, params, lr, momentum=0, dampening=0,
                 weight_decay=0, regularization=0, nesterov=False):

        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, regularization=regularization,
                        nesterov=nesterov)

        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        
        super(FedProxSGD, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(FedProxSGD, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)
        

    @torch.no_grad()
    def step(self, global_model, closure=None):
        """Performs a single optimization step.
        Arguments:
            global_model: 
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']
            nesterov = group['nesterov']
            regularization = group['regularization']

            for p, gp in zip(group['params'], global_model):
                if p.grad is None:
                    continue
                d_p = p.grad
                if weight_decay != 0:
                    d_p = d_p.add(p, alpha=weight_decay)
                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.clone(d_p).detach()
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(d_p, alpha=1 - dampening)
                    if nesterov:
                        d_p = d_p.add(buf, alpha=momentum)
                    else:
                        d_p = buf
                        
                if regularization != 0:
                    d_p = d_p.add(p.data - gp.data, alpha=regularization)

                p.add_(d_p, alpha=-group['lr'])
        

        return loss