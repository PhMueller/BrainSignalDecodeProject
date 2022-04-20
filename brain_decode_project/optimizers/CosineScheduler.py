from typing import Union

import numpy as np
from torch.optim import Optimizer


# https://github.com/robintibor/adamw-eeg-eval/blob/master/adamweegeval/schedulers.py
# https://github.com/pytorch/pytorch/pull/1370/files
# Schedule weight decay should be enabled for AdamW
class AdaptedCosineScheduler():
    def __init__(self, optimizer, eta_min: Union[int, float] = 0, T_max: Union[int, float] = -1):
        if not isinstance(optimizer, Optimizer):
            raise TypeError('{} is not an Optimizer'.format(
                type(optimizer).__name__))

        self.optimizer = optimizer

        for group in optimizer.param_groups:
            group.setdefault('initial_lr', group['lr'])
            group.setdefault('initial_weight_decay', group['weight_decay'])

        self.eta_min = eta_min
        self.T_max = T_max
        self._step_count = 0

    # Starting from epoch 0
    def step(self):

        self._step_count = min(self._step_count + 1, self.T_max)
        progress = self._step_count / self.T_max

        assert 0.0 <= progress <= 1.0
        decay = 0.5 * (1.0 + np.cos(np.pi * progress))

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = max(param_group['initial_lr'] * decay, self.eta_min)
            param_group['weight_decay'] = param_group['initial_weight_decay'] * decay

        self._last_lr = [group['lr'] for group in self.optimizer.param_groups]

    def state_dict(self):
        """Returns the state of the scheduler as a :class:`dict`.

        It contains an entry for every variable in self.__dict__ which
        is not the optimizer.
        """
        return {key: value for key, value in self.__dict__.items() if key != 'optimizer'}

    def load_state_dict(self, state_dict):
        """Loads the schedulers state.

        Args:
            state_dict (dict): scheduler state. Should be an object returned
                from a call to :meth:`state_dict`.
        """
        self.__dict__.update(state_dict)

    def get_last_lr(self):
        """ Return last computed learning rate by current scheduler.
        """
        return self._last_lr
