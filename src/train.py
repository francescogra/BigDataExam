import time
import torch
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import f1_score, precision_score, recall_score
from src.model import GCN
from src.utils import accuracy, plot_confusion_matrix

def train(epoch: int, model: GCN, optimizer: optim.Optimizer, features: torch.Tensor, adj: torch.sparse.FloatTensor, idx_train: torch.Tensor, idx_val: torch.Tensor, labels: torch.Tensor, train_loss_per_model: torch.Tensor, val_loss_per_model: torch.Tensor):
#def train(epoch: int, model: GCN, optimizer: optim.Optimizer, features: torch.Tensor, adj: torch.sparse.FloatTensor, idx_train: torch.Tensor, idx_val: torch.Tensor, labels: torch.Tensor):
    t = time.time()
    model.train()
    optimizer.zero_grad()
    output = model(features, adj)

    loss_train = F.cross_entropy(output[idx_train], labels[idx_train])
    acc_train = accuracy(output[idx_train], labels[idx_train])

    loss_train.backward()
    optimizer.step()
    model.eval()

    output = model(features, adj)
    loss_val = F.cross_entropy(output[idx_val], labels[idx_val])
    acc_val = accuracy(output[idx_val], labels[idx_val])

    #train_loss_per_model.append(loss_train.item())
    #val_loss_per_model.append(loss_val.item())
    train_loss_per_model.append(loss_train.item())
    val_loss_per_model.append(loss_val.item())

    print(f'Epoch: {epoch+1:04d}',
          f'loss_train: {loss_train.item():.4f}',
          f'acc_train: {acc_train.item():.4f}',
          f'loss_val: {loss_val.item():.4f}',
          f'acc_val: {acc_val.item():.4f}',
          f'time: {time.time() - t:.4f}s')

def test(model: GCN, features: torch.Tensor, adj: torch.sparse.FloatTensor, idx_test: torch.Tensor, labels: torch.Tensor):
    model.eval()
    output = model(features, adj)

    loss_test = F.cross_entropy(output[idx_test], labels[idx_test])
    acc_test = accuracy(output[idx_test], labels[idx_test])

    y_pred = output[idx_test].max(1)[1]  # Get the predicted labels
    # Detach y_pred from the computational graph and convert it to a NumPy array
    y_pred_np = y_pred.detach().cpu().numpy()
    labels_np = labels[idx_test].cpu().numpy()

    f1 = f1_score(labels_np, y_pred_np, average='weighted')
    precision = precision_score(labels_np, y_pred_np, average='weighted', zero_division=0)
    recall = recall_score(labels_np, y_pred_np, average='weighted')

    print("Test set results:",
          "loss= {:.4f}".format(loss_test.item()),
          "accuracy= {:.4f}".format(acc_test.item()),
          "f1= {:.4f}".format(f1),
          "precision= {:.4f}".format(precision),
          "recall= {:.4f}".format(recall))

    #plot_confusion_matrix(labels_np, y_pred_np)
    plot_confusion_matrix(labels_np, y_pred_np)


