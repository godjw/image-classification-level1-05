import torch
from torch.utils.data import dataloader

from tqdm import tqdm
from sklearn import metrics

def validate(model, data_loader, device):
    accumulated_accuracy = 0
    accumulated_f1 = 0

    model.eval()

    with torch.no_grad():
        _len = len(data_loader)
        for imgs, labels in tqdm(data_loader, colour='GREEN'):
            imgs = imgs.to(device)
            labels = labels.to(device)

            predictions = model(imgs)

            accumulated_accuracy += (predictions.argmax(dim=1).unsqueeze(dim=1) == labels).float().mean(dim=0).item()
            accumulated_f1 += metrics.f1_score(predictions.argmax(dim=1).unsqueeze(dim=1).cpu(), labels.cpu(), average='macro')
        
        print(f'acc: {(accumulated_accuracy / _len) * 100:0.2f}%\tf1: {accumulated_f1 / _len:.3f}')
    
    model.train()
