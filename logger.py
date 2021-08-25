import os
from io import StringIO
from contextlib import redirect_stdout

from torchsummary import summary

from data_utils import MetadataHelper

class Logger:
    def __init__(self, helper: MetadataHelper):
        self.summary = {}

    def summarize_transform(self, transform):
        self.summary['transform'] = self._hook(
            print,
            "----------------------------------------------------------------",
            'transform:',
            "================================================================",
            transform,
            "----------------------------------------------------------------",
            sep='\n'
        )

    def summarize_model(self, model, input_size):
        self.summary['model'] = self._hook(summary, model=model, input_size=input_size)

    def export(self, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, 'log'), 'w') as f:
            f.write(self.summary['transform'])
        with open(os.path.join(output_dir, 'log'), 'a') as f:
            f.write(self.summary['model'])

    def _hook(self, func, *args, **kwargs):
        iostream = StringIO()
        with redirect_stdout(iostream):
            func(*args, **kwargs)
        return iostream.getvalue()
