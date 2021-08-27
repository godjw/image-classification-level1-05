import configparser
import os
from collections import Counter

import torch
from torch.utils.data import Dataset

import numpy as np
import pandas as pd
from PIL import Image
from sklearn import model_selection

from config_parser import ConfigParser

class MaskClassifierDataset(Dataset):
    def __init__(self, img_paths, labels, transform=None):
        self.img_paths = img_paths
        self.labels = labels
    
        self.transform = transform

    def __getitem__(self, index):
        img = Image.open(self.img_paths[index])
        if self.transform:
            img = self.transform(img)
        return img, torch.tensor(self.labels[index]).unsqueeze(dim=0)

    def __len__(self):
        return len(self.labels)

class EvalDataset(Dataset):
    def __init__(self, img_paths, transform=None):
        self.img_paths = img_paths

        self.transform = transform

    def __getitem__(self, index):
        image = Image.open(self.img_paths[index])
        if self.transform:
            image = self.transform(image)
        return image

    def __len__(self):
        return len(self.img_paths)

class MetadataHelper:
    def __init__(self, config: ConfigParser):
        self.config = config
        self._set_paths()
        self._set_metadata()
        self._set_paths_and_labels()
    
    def _set_paths(self):
        self.dirs = {
            'train_root_dir': os.path.join(self.config.data_dir, self.config.train_dir_name),
            'eval_root_dir': os.path.join(self.config.data_dir, self.config.eval_dir_name)
        }

    def get_paths(self):
        return self.dirs

    def _set_metadata(self):
        self.metadata = {
            'train_metadata': pd.read_csv(os.path.join(self.dirs['train_root_dir'], 'train.csv')).drop(columns=['id', 'race']),
            'eval_metadata': pd.read_csv(os.path.join(self.dirs['eval_root_dir'], 'info.csv'))
        }

    def get_metadata(self):
        return self.metadata

    def _set_paths_and_labels(self):
        self.paths_and_labels = {
            'train_img_paths': [],
            'train_labels': []
        }
        label_dict = {'male': 0, 'female': 3}
        for gender, age, img_dir_name in self.metadata['train_metadata'].to_numpy():
            _label = label_dict[gender] + (age // 30)
            for file_name in os.listdir(img_dir := os.path.join(self.dirs['train_root_dir'], 'images', img_dir_name)):
                if file_name.startswith('.'):
                    continue
                label = _label
                if file_name.startswith('incorrect'):
                    label += 6
                elif file_name.startswith('normal'):
                    label += 12
                self.paths_and_labels['train_img_paths'].append(os.path.join(img_dir, file_name))
                self.paths_and_labels['train_labels'].append(label)
        
        img_dir = os.path.join(self.dirs['eval_root_dir'], 'images')
        self.paths_and_labels['eval_img_paths'] = [os.path.join(img_dir, img_id) for img_id in self.metadata['eval_metadata'].ImageID]

    def get_paths_and_labels(self):
        return self.paths_and_labels

    def __repr__(self):
        description = f'''
        total number of images: {len(self.paths_and_labels["train_labels"])}
        for each class: {Counter(self.paths_and_labels["train_labels"])}
        '''
        return description
