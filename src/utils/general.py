import sys
from functools import wraps
import logging
import os
import random
import time
from contextlib import contextmanager
from typing import Union
from pathlib import Path

import numpy as np
import torch
from hydra.experimental import compose, initialize
from hydra._internal.hydra import Hydra as BaseHydra

def seed_everything(seed=1234):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True


@contextmanager
def timer(name: str, logger: Union[logging.Logger, None] = None):
    t0 = time.time()
    yield
    msg = f'[{name}] done in {time.time()-t0:.3f} s'
    if logger:
        logger.info(msg)
    else:
        print(msg)


def tail_recursive(func):
    self_func = [func]
    self_firstcall = [True]
    self_CONTINUE = [object()]
    self_argskwd = [None]

    @wraps(func)
    def _tail_recursive(*args, **kwd):
        if self_firstcall[0] == True:
            func = self_func[0]
            CONTINUE = self_CONTINUE
            self_firstcall[0] = False
            try:
                while True:
                    result = func(*args, **kwd)
                    if result is CONTINUE:  # update arguments
                        args, kwd = self_argskwd[0]
                    else:  # last call
                        return result
            finally:
                self_firstcall[0] = True
        else:  # return the arguments of the tail call
            self_argskwd[0] = args, kwd
            return self_CONTINUE

    return _tail_recursive

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def load_conf(path):
    # config_pathをos.getcwd()基準から __file__基準に変更
    config_path = Path(path).parent
    config_name = Path(path).stem

    # TODO: 対応
    # (os.getcwd() / Path(path).parent).resolve() == (Path(__file__).parent / config_path).resolve()
    target = (os.getcwd() / Path(path).parent).resolve()

    config_path = '..' / config_path
    
    with initialize(config_path=config_path, job_name=None):
        cfg = compose(config_name=config_name, overrides=[arg for arg in sys.argv[1:] if '=' in arg])
    return cfg