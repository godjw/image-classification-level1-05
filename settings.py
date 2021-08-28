import os
import glob
from pathlib import Path
import random
import re

import torch

import numpy as np


class SettingsHelper:
    def __init__(self, args, device=torch.device('cuda')):
        self.args = args
        self.device = device
        self._set_seed(seed=args.seed)

    def _set_seed(self, seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # multi-GPU
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(seed)
        random.seed(seed)

    def get_save_dir(self, dump=False):
        save_dir = Path(os.path.join(self.args.model_dir, self.args.name))
        if not save_dir.exists() or dump:
            return str(save_dir)
        else:
            dirs = glob.glob(f'{save_dir}*')
            matches = [re.search(rf'{save_dir.stem}(\d+)', d) for d in dirs]
            i = [int(m.groups()[0]) for m in matches if m]
            n = max(i) + 1 if i else 2
            return f'{save_dir}{n}'
