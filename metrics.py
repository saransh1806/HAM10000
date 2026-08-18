import torch
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import numpy as np

def get_predictions(model, dataloader, device):
    """
    Runs the model on the full dataloader and returns true labels and predicted labels.
    """
    model.eval()
    all_preds = []
    all_labels = []

    with torch.inference_mode():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            logits = model(X)
            preds = torch.argmax(logits, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

    return np.array(all_labels), np.array(all_preds)


def plot_confusion_matrix(y_true, y_pred, class_names, model_name):
    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")

    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title(f"Confusion Matrix - {model_name}")

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                     color="white" if cm[i, j] > cm.max() / 2 else "black")

    fig.colorbar(im)
    plt.tight_layout()
    plt.show()


def print_classification_report(y_true, y_pred, class_names, model_name):
    print(f"\nClassification Report - {model_name}")
    print(classification_report(y_true, y_pred, target_names=class_names))


def evaluate_model(model, dataloader, class_names, device, model_name="Model"):
    y_true, y_pred = get_predictions(model, dataloader, device)
    print_classification_report(y_true, y_pred, class_names, model_name)
    plot_confusion_matrix(y_true, y_pred, class_names, model_name)
    return y_true, y_pred