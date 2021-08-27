from argparse import ArgumentParser
import json
from dataclasses import dataclass

@dataclass
class ConfigParser(ArgumentParser):
    # basic settings
    data_dir: str = None
    eval_dir_name: str = None
    train_dir_name: str = None
    trial_name: str = None

    # hyperparameters
    n_epochs: int = 10
    batch_size: int = 64
    learning_rate: float = 1e-3
    num_workers: int = 3
    stratified_k_fold: int = 2

    def __init__(self, description):
        super().__init__(description=description)
        self.add_argument('--config', metavar='-C', help='parse configuration written in .json')
        self.add_argument('--name', metavar='-N', help='set the name of current trial')
        self._set_config()
    
    def _set_config(self):
        _args_dict = vars(self.parse_args())
        with open(_args_dict['config'], 'r') as config_json:
            contents = config_json.read()
            configs = json.loads(contents)

            for arg in self.__annotations__.keys():
                if arg in configs['config']:
                    exec(f'self.{arg} = configs["config"][arg]')
        self.trial_name = _args_dict['name']
