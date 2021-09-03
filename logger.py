import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

from dataset import *


def get_lr(optimizer):
    """
    Returns current learning rate.

    Args:
        optimizer: Optimizer classes torch.optim includes
    """
    for param_group in optimizer.param_groups:
        return param_group["lr"]


def save_confusion_matrix(num_classes, labels, preds, save_path):
    """
    Saves confusion matrix that evaluates the accuracy of a classification.

    Args:
        num_classes: Number of classes
        labels: Ground truth target labels
        preds: Estimated target predictions
        save_path: A path confusion matrix to be saved
    """
    confusion = confusion_matrix(y_true=labels, y_pred=preds, normalize="true")
    df = pd.DataFrame(confusion, index=list(range(num_classes)), columns=list(range(num_classes)))
    df = df.fillna(0)

    plt.figure(figsize=(10, 9))
    plt.tight_layout()
    plt.suptitle("Confusion Matrix")
    sns.heatmap(df, cmap=sns.color_palette("Blues"), annot=True, fmt=".2f", linewidth=0.1, square=True)
    plt.xlabel("Predicted")
    plt.ylabel("True")

    plt.savefig(save_path)
