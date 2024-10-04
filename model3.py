import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import scipy.sparse as sp
import math
import torch
from torch import nn
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import time
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, precision_recall_curve
from sklearn.decomposition import PCA
import seaborn as sns
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import StratifiedShuffleSplit


def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)), dtype=np.int32)
    return labels_onehot

def normalize(mx):
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse_coo_tensor(indices, values, shape)

def load_data(path="./data_uni/", dataset="dataset_uni"):
    print('Loading {} dataset...'.format(dataset))
    df = pd.read_csv("D:\\uni\\magistrale\\materie\\Tesi\\GeneExpressionMetabric.csv", sep=';')
    df = df.iloc[:, 1:]
    # Controllo il bilanciamento delle classi prima di SMOTE
    class_labels = df.iloc[:, -1]
    class_counts = class_labels.value_counts()
    print("Occorrenze delle classi prima di SMOTE:")
    print(class_counts)

    data = df.to_numpy()[1:, :]
    data[:, 0] = np.vectorize(lambda value: value.replace("MB-", ""))(data[:, 0])
    data = np.char.replace(data.astype(str), ',', '.')
    data[:, 1:-3] = data[:, 1:-3].astype(np.float64)

    idx_features_labels = np.array(data)
    features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
    labels = encode_onehot(idx_features_labels[:, -1])
    idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
    idx_map = {j: i for i, j in enumerate(idx)}

    edges_unordered = np.genfromtxt("{}network.txt".format("./data_uni/"), dtype=np.int32)
    edges_mapped = [(idx_map.get(edge[0]), idx_map.get(edge[1])) for edge in edges_unordered if
                    idx_map.get(edge[0]) is not None and idx_map.get(edge[1]) is not None]
    edges = np.array(edges_mapped, dtype=np.int32)

    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])), shape=(labels.shape[0], labels.shape[0]),
                        dtype=np.float32)
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    features_dense = features.toarray() if sp.issparse(features) else features
    mean = np.mean(features_dense, axis=1)[:, np.newaxis]
    std = np.std(features_dense, axis=1)[:, np.newaxis]
    std[std == 0] = 1
    features_zscore = (features_dense - mean) / std
    features = sp.csr_matrix(features_zscore)
    adj = normalize(adj + sp.eye(adj.shape[0]))

    # Convert to dense matrix for SMOTE
    features_dense = features.toarray()
    labels_dense = np.argmax(labels, axis=1)

    # Applicare SMOTE
    smote = SMOTE(random_state=42)
    features_resampled, labels_resampled = smote.fit_resample(features_dense, labels_dense)

    # Controllo il bilanciamento delle classi dopo SMOTE
    class_counts_after_smote = np.bincount(labels_resampled)
    print("Occorrenze delle classi dopo SMote:")
    print(class_counts_after_smote)

    # Convert back to sparse matrix and one-hot encoding
    features_resampled = sp.csr_matrix(features_resampled, dtype=np.float32)
    labels_resampled = encode_onehot(labels_resampled)

    # Costruzione di una nuova matrice di adiacenza per i dati campionati
    num_samples = features_resampled.shape[0]
    adj_resampled = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0] % num_samples, edges[:, 1] % num_samples)),
                                  shape=(num_samples, num_samples), dtype=np.float32)
    adj_resampled = adj_resampled + adj_resampled.T.multiply(adj_resampled.T > adj_resampled) - adj_resampled.multiply(
        adj_resampled.T > adj_resampled)
    adj_resampled = normalize(adj_resampled + sp.eye(adj_resampled.shape[0]))

    # Stratified train-validation-test split
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
    train_val_idx, test_idx = next(sss.split(features_resampled, np.argmax(labels_resampled, axis=1)))

    sss_val = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=43)  # Different random_state
    train_idx, val_idx = next(
        sss_val.split(features_resampled[train_val_idx], np.argmax(labels_resampled[train_val_idx], axis=1)))

    # Map the train_val indices back to original indices
    val_idx = train_val_idx[val_idx]
    train_idx = train_val_idx[train_idx]

    # Ensure balanced classes in train, validation, test sets
    train_labels = np.argmax(labels_resampled[train_idx], axis=1)
    val_labels = np.argmax(labels_resampled[val_idx], axis=1)
    test_labels = np.argmax(labels_resampled[test_idx], axis=1)

    print("Train set class counts:", np.bincount(train_labels))
    print("Validation set class counts:", np.bincount(val_labels))
    print("Test set class counts:", np.bincount(test_labels))

    # Convert to PyTorch tensors
    idx_train = torch.LongTensor(train_idx)
    idx_val = torch.LongTensor(val_idx)
    idx_test = torch.LongTensor(test_idx)

    features = torch.FloatTensor(np.array(features_resampled.todense()))
    labels = torch.LongTensor(np.where(labels_resampled)[1])
    adj = sparse_mx_to_torch_sparse_tensor(adj_resampled)

    return adj, features, labels, idx_train, idx_val, idx_test

adj, features, labels, idx_train, idx_val, idx_test = load_data()

class GraphConvolution(Module):

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()

        # create Weight and Bias trainable parameters
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        # standard weight to be uniform
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        # Ensure adj is a dense matrix
        if adj.is_sparse:
            adj = adj.to_dense()

        # Normalization: D^-1/2 * A_hat * D^-1/2
        identity = torch.eye(adj.size(0)).to(adj.device)
        adj_hat = adj + identity
        D = torch.diag(torch.sum(adj_hat, dim=1))
        D_inv_sqrt = torch.diag(torch.pow(D.diag(), -0.5))
        adj_norm = torch.matmul(torch.matmul(D_inv_sqrt, adj_hat), D_inv_sqrt)

        support = torch.mm(input, self.weight)

        output = torch.mm(adj_norm, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output


    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

class GraphSAGEConv(nn.Module):
    def __init__(self, in_features, out_features, aggregator_type='mean', bias=True):
        super(GraphSAGEConv, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.aggregator_type = aggregator_type

        # Dimensione del peso modificata per gestire caratteristiche concatenate
        self.weight = Parameter(torch.FloatTensor(in_features * 2, out_features))

        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        neighbors = torch.spmm(adj, input)
        if self.aggregator_type == 'mean':
            degree = adj.sum(1).to_dense()
            degree[degree == 0] = 1  # Evita la divisione per zero
            agg_neighbors = neighbors / degree.unsqueeze(1)
        elif self.aggregator_type == 'sum':
            agg_neighbors = neighbors
        else:
            raise NotImplementedError

        # Concatena le caratteristiche di input e quelle aggregate dai vicini
        support = torch.cat([input, agg_neighbors], dim=1)

        # Dimensione del peso modificata per gestire caratteristiche concatenate
        support = torch.mm(support, self.weight)

        if self.bias is not None:
            support = support + self.bias

        return support


class GATConv(nn.Module):
    def __init__(self, in_features, out_features, num_heads=1, concat=True, bias=True):
        super(GATConv, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_heads = num_heads
        self.concat = concat
        self.attention = Parameter(torch.FloatTensor(2 * out_features, num_heads))
        self.weight = Parameter(torch.FloatTensor(in_features, out_features * num_heads))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features * num_heads))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        h = torch.mm(input, self.weight)
        N = h.size()[0]
        a_input = torch.cat([h.repeat(1, N).view(N * N, -1), h.repeat(N, 1)], dim=1).view(N, -1, 2 * self.out_features)

        e = F.leaky_relu(torch.matmul(a_input, self.attention).squeeze(2))
        attention = F.softmax(e, dim=1)
        attention = F.dropout(attention, 0.6, training=self.training)
        h_prime = torch.matmul(attention, h)
        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

class GCN(nn.Module):
    def __init__(self, n_feat, n_hids, n_class, dropout):
        super(GCN, self).__init__()
        layers_units = [n_feat] + n_hids
        self.graph_layers = nn.ModuleList(
            [GraphConvolution(layers_units[idx], layers_units[idx + 1]) for idx in range(len(layers_units) - 1)]
        )
        self.output_layer = GraphConvolution(layers_units[-1], n_class)
        self.dropout = dropout

    def forward(self, x, adj):
        for graph_layer in self.graph_layers:
            x = F.relu(graph_layer(x, adj))
            x = F.dropout(x, self.dropout, training=self.training)
        x = self.output_layer(x, adj)
        return torch.log_softmax(x, dim=1)

class GraphSAGE(nn.Module):
    def __init__(self, n_feat, n_hids, n_class, dropout):
        super(GraphSAGE, self).__init__()
        layers_units = [n_feat] + n_hids
        self.graph_layers = nn.ModuleList(
            [GraphSAGEConv(layers_units[idx], layers_units[idx + 1]) for idx in range(len(layers_units) - 1)]
        )
        self.output_layer = GraphSAGEConv(layers_units[-1], n_class)
        self.dropout = dropout

    def forward(self, x, adj):
        for graph_layer in self.graph_layers:
            x = F.relu(graph_layer(x, adj))
            x = F.dropout(x, self.dropout, training=self.training)
        x = self.output_layer(x, adj)
        return torch.log_softmax(x, dim=1)

class GAT(nn.Module):
    def __init__(self, n_feat, n_hids, n_class, dropout, num_heads=1):
        super(GAT, self).__init__()
        layers_units = [n_feat] + n_hids
        self.graph_layers = nn.ModuleList(
            [GATConv(layers_units[idx], layers_units[idx + 1], num_heads) for idx in range(len(layers_units) - 1)]
        )
        self.output_layer = GATConv(layers_units[-1], n_class, num_heads, concat=False)
        self.dropout = dropout

    def forward(self, x, adj):
        for graph_layer in self.graph_layers:
            x = F.elu(graph_layer(x, adj))
            x = F.dropout(x, self.dropout, training=self.training)
        x = self.output_layer(x, adj)
        return torch.log_softmax(x, dim=1)

def accuracy(output, labels):
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

def apply_pca_and_plot(models, features, adj, labels, idx_test):
    for i, model in enumerate(models):
        model.eval()
        with torch.no_grad():
            output = model(features, adj)
        embeddings = output[idx_test].cpu().numpy()  # Prendi solo i nodi di test

        # Rimuovi righe con NaN
        embeddings = embeddings[~np.isnan(embeddings).any(axis=1)]

        if embeddings.shape[0] == 0:
            print(f"Tutti gli embeddings contengono NaN per il modello {model.__class__.__name__}.")
            continue

        # Applica PCA agli embeddings
        pca = PCA(n_components=2)
        reduced_embeddings = pca.fit_transform(embeddings)

        # Plot PCA
        plt.figure(figsize=(10, 7))
        plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], c=labels[idx_test][~np.isnan(embeddings).any(axis=1)], cmap='viridis', s=50)
        plt.colorbar()
        plt.title(f'PCA of Node Embeddings ({model.__class__.__name__})')
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.show()

class Args:
    no_cuda = False
    seed = 42
    epochs = 1200
    lr = 0.0001
    weight_decay = 0.04
    n_hid = [128, 64]
    dropout = 0.3

args = Args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

input_features = features.shape[1]
gcn_model = GCN(n_feat=input_features, n_hids=args.n_hid, n_class=2, dropout=args.dropout)
sage_model = GraphSAGE(n_feat=input_features, n_hids=args.n_hid, n_class=2, dropout=args.dropout)
gat_model = GAT(n_feat=input_features, n_hids=args.n_hid, n_class=2, dropout=args.dropout)

models = [gcn_model, sage_model, gat_model]

optimizers = [
    optim.Adam(gcn_model.parameters(), lr=args.lr, weight_decay=args.weight_decay),
    optim.Adam(sage_model.parameters(), lr=args.lr, weight_decay=args.weight_decay),
    optim.Adam(gat_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
]

train_loss_per_model = [[] for _ in models]
val_loss_per_model = [[] for _ in models]

def train(epoch, models, optimizers, train_loss_per_model, val_loss_per_model):
    t = time.time()
    for i, (model, optimizer) in enumerate(zip(models, optimizers)):
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

        train_loss_per_model[i].append(loss_train.item())
        val_loss_per_model[i].append(loss_val.item())

        print(f'Epoch: {epoch + 1:04d} | Model: {model.__class__.__name__} | loss_train: {loss_train.item():.4f} | acc_train: {acc_train.item():.4f} | loss_val: {loss_val.item():.4f} | acc_val: {acc_val.item():.4f} | time: {time.time() - t:.4f}s')

t_start = time.time()
for epoch in range(args.epochs):
    train(epoch, models, optimizers, train_loss_per_model, val_loss_per_model)

print("Model training is complete!")
print("Total model training time: {:.4f}s".format(time.time() - t_start))



def test(models):
    outputs = []
    for model in models:
        model.eval()
        output = model(features, adj)
        outputs.append(output)
    output = torch.stack(outputs).mean(0)

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

    plot_confusion_matrix(labels_np, y_pred_np)

test(models)

apply_pca_and_plot(models, features, adj, labels, idx_test)

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
    for i, model in enumerate(models):
        ax.plot(epochs, train_loss_per_model[i], label=f'Train Loss ({model.__class__.__name__})')
        ax.plot(epochs, val_loss_per_model[i], label=f'Validation Loss ({model.__class__.__name__})')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Loss')
    ax.set_title(title, loc='right')
    ax.legend()
    if y_limit[0] is not None and y_limit[1] is not None:
        ax.set_ylim(y_limit)

plt.tight_layout()
#plt.savefig('training_validation_loss_all_limits.png')
plt.show()
