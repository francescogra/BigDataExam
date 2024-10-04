import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import torch
import numpy as np
import torch.optim as optim
from src.train import train, test
from src.model import GCN
from src.config import Args
from src.data_loader import load_data  # Importa la funzione load_data dal tuo dataloader
from src.utils import apply_pca_and_plot

def main():
    args = Args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    # Carica i dati utilizzando la funzione load_data()
    adj, features, labels, idx_train, idx_val, idx_test = load_data()

    # Applica PCA ai dati originali
    features_np = features.cpu().numpy()
    apply_pca_and_plot(features_np[idx_test], labels[idx_test], 'PCA of Original Features')

    input_features = features.shape[1]
    model = GCN(n_feat=input_features,
                n_hids=args.n_hid,
                n_class=2,
                dropout=args.dropout)

    optimizer = optim.Adam(model.parameters(),
                           lr=args.lr,
                           weight_decay=args.weight_decay)

    train_loss_per_model = []
    val_loss_per_model = []

    for epoch in range(args.epochs):
        train(epoch, model, optimizer, features, adj, idx_train, idx_val, labels, train_loss_per_model, val_loss_per_model)

    test(model, features, adj, idx_test, labels)

    # Ottieni gli embeddings dopo la GCN
    model.eval()
    with torch.no_grad():
        output = model(features, adj)
    embeddings = output.cpu().numpy()  # Prendi tutti i nodi

    # Applica PCA agli embeddings ottenuti dopo la GCN
    apply_pca_and_plot(embeddings[idx_test], labels[idx_test], f'PCA after Node Embeddings ({model.__class__.__name__})')

    # Plotting the training and validation loss
    # Configurazioni dei limiti per gli assi y
    y_limits = [(None, None), (0, 1), (0, 0.5)]
    titles = [
        'Training and Validation Loss (No Y-Limit)',
        'Training and Validation Loss (Y-Limit: 0 to 1)',
        'Training and Validation Loss (Y-Limit: 0 to 0.4)'
    ]

    epochs = range(1, args.epochs + 1)
    fig, axs = plt.subplots(3, 1, figsize=(10, 15))

    for ax, y_limit, title in zip(axs, y_limits, titles):
        ax.plot(epochs, train_loss_per_model, label=f'Train Loss ({model.__class__.__name__})')
        ax.plot(epochs, val_loss_per_model, label=f'Validation Loss ({model.__class__.__name__})')
        ax.set_xlabel('Epochs')
        ax.set_ylabel('Loss')
        ax.set_title(title, loc='right')
        ax.legend()
        if y_limit[0] is not None and y_limit[1] is not None:
            ax.set_ylim(y_limit)

    plt.tight_layout()
    #plt.savefig('training_validation_loss_all_limits.png')
    plt.show()

if __name__ == "__main__":
    main()
