import torch
from torch import nn, optim
from torch.utils.data import DataLoader

import torchvision
import torchvision.transforms as T

from sklearn import model_selection, metrics
from imblearn import over_sampling
from tqdm import tqdm

from config_parser import ConfigParser
from data_utils import *
import models
from logger import Logger
from validation import validate
from inference import save_submission

device = torch.device('cuda' if torch.cuda.is_available else 'cpu')

config = ConfigParser(description='mask classification learner')
helper = MetadataHelper(config=config)
logger = Logger(helper=helper)

model = torchvision.models.resnet18(pretrained=True)
model.fc = nn.Sequential(
    nn.Linear(in_features=512, out_features=256),
    nn.ReLU(),
    nn.Dropout(p=0.5),
    nn.Linear(in_features=256, out_features=18)
)
model.layer1.requires_grad_(False)
model.layer2.requires_grad_(False)

model = model.to(device)
logger.summarize_model(model=model, input_size=(3, 512, 384))

criterion = nn.MultiLabelSoftMarginLoss().to(device)
optimizer = optim.Adam(params=model.parameters(), lr=config.learning_rate)

base_transforms = [
    T.Resize((512, 384), T.InterpolationMode.BICUBIC),
    T.ToTensor(),
    T.Normalize(
        mean=(0.56019358, 0.52410121, 0.501457),
        std=(0.23318603, 0.24300033, 0.24567522)
    )
]
train_transforms = [
    *base_transforms,
    T.RandomHorizontalFlip(p=0.5),
    T.RandomAutocontrast(p=0.5),
    T.RandomAdjustSharpness(2, p=0.5)
]
logger.summarize_transform(transform=T.Compose(train_transforms))

paths_and_labels = helper.get_paths_and_labels()
train_img_paths = np.array(paths_and_labels['train_img_paths'])
train_labels = np.array(paths_and_labels['train_labels'])
train_idxs = np.arange(0, len(train_labels))

skf = model_selection.StratifiedKFold(n_splits=config.stratified_k_fold)
for epoch, (train_idxs, val_idxs) in enumerate(skf.split(train_img_paths, train_labels), 1):
    train_dataset = MaskClassifierDataset(
        img_paths=train_img_paths[train_idxs],
        labels=train_labels[train_idxs],
        transform=T.Compose(train_transforms)
    )
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        drop_last=True
    )
    
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
    
    print(f'epoch: {epoch}/{config.n_epochs}\tacc: {(accumulated_accuracy / len(train_loader)) * 100:0.2f}%\tf1: {accumulated_f1 / len(train_loader):.3f}\tloss: {running_loss / len(train_loader):0.3f}')

    val_dataset = MaskClassifierDataset(
        img_paths=train_img_paths[val_idxs],
        labels=train_labels[val_idxs],
        transform=T.Compose(base_transforms)
    )
    val_loader = DataLoader(
        dataset=val_dataset,
        num_workers=config.num_workers
    )
    validate(model=model, data_loader=val_loader, device=device)

logger.export(output_dir=config.trial_name)
save_submission(model=model, transforms=T.Compose(base_transforms), helper=helper, device=device)
