import os

import torch
from torch import nn, optim
from torch.utils.data import DataLoader

import torchvision
import torchvision.transforms as T

from torchsummary import summary
from sklearn import model_selection, metrics
import pandas as pd
from tqdm import tqdm

from config_parser import ConfigParser
from data_utils import *
import models
from logger import Logger

device = torch.device('cuda' if torch.cuda.is_available else 'cpu')

config = ConfigParser(description='mask classification learner')
helper = MetadataHelper(config=config)
logger = Logger(helper=helper)

paths_and_labels = helper.get_paths_and_labels()

train_img_paths, val_img_paths, train_labels, val_labels = model_selection.train_test_split(
    paths_and_labels['train_img_paths'],
    paths_and_labels['train_labels'],
    test_size=0.2,
    shuffle=True,
    stratify=paths_and_labels['train_labels']
)

base_transforms = [
    T.Resize((512, 384), T.InterpolationMode.BICUBIC),
    T.ToTensor(),
    T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.2, 0.2, 0.2))
]
train_dataset = MaskClassifierDataset(
    img_paths=train_img_paths,
    labels=train_labels,
    transform=T.Compose([
        *base_transforms,
        T.RandomHorizontalFlip(p=0.5)
    ])
)
val_dataset = MaskClassifierDataset(
    img_paths=val_img_paths,
    labels=val_labels,
    transform=T.Compose(base_transforms)
)
logger.summarize_transform(transform=train_dataset.transform)

train_loader = DataLoader(dataset=train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=2, drop_last=True)
val_loader = DataLoader(dataset=val_dataset, num_workers=2)

model = models.MaskClassifierModel(num_classes=18).to(device)
logger.summarize_model(model=model, input_size=(3, 512, 384))

criterion = nn.MultiLabelSoftMarginLoss().to(device)
optimizer = optim.Adam(params=model.parameters(), lr=config.learning_rate)

for epoch in range(1, config.n_epochs + 1):
    running_loss = 0
    accumulated_accuracy = 0
    accumulated_f1 = 0
    for imgs, labels in tqdm(train_loader):
        imgs = imgs.to(device)
        labels = labels.to(device)

        predictions = model(imgs)
        labels_one_hot = torch.zeros_like(predictions).scatter_(1, labels, 1)
        loss = criterion(predictions, labels_one_hot)

        running_loss += loss.item()
        accumulated_accuracy += (predictions.argmax(dim=1).unsqueeze(dim=1) == labels).float().mean(dim=0).item()
        accumulated_f1 += metrics.f1_score(predictions.argmax(dim=1).unsqueeze(dim=1).cpu(), labels.cpu(), average='macro')

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    print(f'epoch: {epoch:02d}/{config.n_epochs}\tacc: {(accumulated_accuracy / len(train_loader)) * 100:0.2f}%\tf1: {accumulated_f1 / len(train_loader):.3f}\tloss: {running_loss / len(train_loader):0.3f}')

logger.export(output_dir=config.trial_name)
