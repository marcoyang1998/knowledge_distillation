import torch
import logging

class Tri_state_adam(object):
    # a tri-state adam optim, first phase warmup, second phase remain the same, third phase decay linearly
    def __init__(self, phase, total_steps, init_lr, warmup_lr, end_lr, optim):
        self.optimizer = optim
        self._step = 0
        self._rate = init_lr
        self.phase = phase
        self.total_steps = total_steps
        self.init_lr = init_lr
        self.warmup_lr = warmup_lr
        self.end_lr = end_lr
        self.warmup_factor = (warmup_lr-init_lr)/(phase[0]*total_steps)
        self.decay_factor = (warmup_lr-end_lr)/(phase[2]*total_steps)

    @property
    def param_groups(self):
        return self.optimizer.param_groups

    def rate(self):
        if self._step < self.phase[0]*self.total_steps:
            return self.init_lr + self._step * self.warmup_factor
        elif self._step >= self.phase[0]*self.total_steps and self._step < (1-self.phase[-1])*self.total_steps:
            return self.warmup_lr
        else:
            return self.warmup_lr - (self._step - (1-self.phase[2])*self.total_steps)*self.decay_factor


    def step(self):
        rate = self.rate()
        if self._step%50==0:
            print('Current learning rate: {} at step: {}'.format(rate, self._step))
        self._step += 1
        self._rate = rate
        for p in self.optimizer.param_groups:
            p["lr"] = rate
        self.optimizer.step()

    def zero_grad(self):
        """Reset gradient."""
        self.optimizer.zero_grad()

    def state_dict(self):
        """Return state_dict."""
        return {
            "_step": self._step,
            "phase": self.phase,
            "init_lr": self.init_lr,
            "end_lr": self.end_lr,
            "warmup_lr": self.warmup_lr,
            "warmup_factor": self.warmup_factor,
            "decay_factor": self.decay_factor,
            "total_steps": self.total_steps,
            "_rate": self._rate,
            "optimizer": self.optimizer.state_dict(),
        }

def get_opt(model_params, phase, total_steps, init_lr, warmup_lr, end_lr):
    """Get standard NoamOpt."""
    base = torch.optim.Adam(model_params, lr=0, betas=(0.9, 0.98), eps=1e-08)
    return Tri_state_adam(phase, total_steps, init_lr, warmup_lr, end_lr, base)