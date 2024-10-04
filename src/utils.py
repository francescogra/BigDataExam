import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn.decomposition import PCA
import numpy as np

def accuracy(output: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Calculate accuracy."""
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)

def plot_confusion_matrix(y_true, y_pred):
    conf_matrix = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 7))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    plt.show()

def apply_pca_and_plot(embeddings, labels, title):
    # Rimuovi righe con NaN
    embeddings = embeddings[~np.isnan(embeddings).any(axis=1)]

    if embeddings.shape[0] == 0:
        print(f"Tutti gli embeddings contengono NaN.")
        return

    # Applica PCA agli embeddings
    pca = PCA(n_components=2)
    reduced_embeddings = pca.fit_transform(embeddings)

    # Plot PCA
    plt.figure(figsize=(10, 7))
    plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], c=labels[~np.isnan(embeddings).any(axis=1)], cmap='viridis', s=50)
    plt.colorbar()
    plt.title(title)
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.show()
