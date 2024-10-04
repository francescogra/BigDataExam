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
import matplotlib
matplotlib.use('TkAgg')  # Utilizza il backend TkAgg
import matplotlib.pyplot as plt

output = None

"""
encode_onehot(): restituisce l'array labels_onehot, che contiene i vettori one-hot corrispondenti alle etichette originali. 
serve per mappare le classi in formato numerico
"""
def encode_onehot(labels):
    """ Turn the label into A one hot vector """
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)), dtype=np.int32)
    return labels_onehot

"""
normalizza una matrice sparsa riga per riga, assicurando che la somma degli elementi di ciascuna riga sia 1 (o 0, se la riga era completamente zero). 
per normalizzare i valori """
def normalize(mx):
    """ Normalize sparse matrix by row """
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """ Convert a sparse matrix from scipy format to torch format """
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse_coo_tensor(indices, values, shape)

def load_data(path="./data_uni/", dataset="dataset_uni"):
    """Load network dataset"""
    print('Loading {} dataset...'.format(dataset))

    df = pd.read_csv("D:\\uni\\magistrale\\materie\\Tesi\\GeneExpressionMetabric2.csv", sep=';')

    # Remove the first column
    df = df.iloc[:, 1:]

    # reverse in numpy
    data = df.to_numpy()

    # Remove the first row of data which contains the column labels
    data = data[1:, :]

    # Define a function to replace "MB-" with an empty string
    def replace_func(value):
        return value.replace("MB-", "")

    # Create a vectorized version of the function
    vfunc = np.vectorize(replace_func)

    # Apply the vectorized function to the first column of the matrix
    data[:, 0] = vfunc(data[:, 0])

    # Ensure the entire array is of type string for string operations
    data = data.astype(str)

    # Replace all occurrences of the comma with a period in the entire array
    data = np.char.replace(data, ',', '.')

    # Convert the remaining columns to float after replacing commas with periods, except the first and the last 3 columns
    data[:, 1:-3] = data[:, 1:-3].astype(np.float64)

    #shuffle matrix rows to avoid overfitting
    np.random.shuffle(data)

    # convert data matrix in numpy array
    idx_features_labels = np.array(data)

    # takes the characteristics of each patient, that is, each gene and store it in a sparse matrix format
    features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)

    # Take the years of each patient as a label and convert it into a one hot vector
    labels = encode_onehot(idx_features_labels[:, -1])
    # Take out the id of each patient
    idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
    idx_map = {j: i for i, j in enumerate(idx)}

    # network data is converted to numpy vector
    edges_unordered = np.genfromtxt("{}network.txt".format("./data_uni/"), dtype=np.int32)

    # Map the id in the network data to the interval [0, 212]
    edges_mapped = []
    for edge in edges_unordered:
        src_mapped = idx_map.get(edge[0])
        trg_mapped = idx_map.get(edge[1])
        if src_mapped is not None and trg_mapped is not None:
            edges_mapped.append((src_mapped, trg_mapped))

    edges = np.array(edges_mapped, dtype=np.int32)

    # Store the gene relationship between patient in a sparse matrix format
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(labels.shape[0], labels.shape[0]),
                        dtype=np.float32)
    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    # Normalizzazione Z-score delle caratteristiche
    features_dense = features.toarray() if sp.issparse(features) else features
    mean = np.mean(features_dense, axis=1)[:, np.newaxis]
    std = np.std(features_dense, axis=1)[:, np.newaxis]
    std[std == 0] = 1  # Evita la divisione per zero
    features_zscore = (features_dense - mean) / std
    features = sp.csr_matrix(features_zscore)

    # Normalizzazione della matrice di adiacenza
    adj = normalize(adj + sp.eye(adj.shape[0]))

    # Produce the final vector
    idx_train = range(0, 119)
    idx_val = range(120, 159)
    idx_test = range(160, 212)

    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(np.where(labels)[1])
    adj = sparse_mx_to_torch_sparse_tensor(adj)
    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    # Aggiungi print per vedere le dimensioni dei dati caricati
    print(f'features.shape: {features.shape}')
    print(f'labels.shape: {labels.shape}')
    print(f'adj.shape: {adj.shape}')
    print(f'idx_train: {list(idx_train)}')
    print(f'idx_val: {list(idx_val)}')
    print(f'idx_test: {list(idx_test)}')

    return adj, features, labels, idx_train, idx_val, idx_test

# Load data, including transformed adjacency matrix, graph network input feature, classification label, training data, verification data, test data
adj, features, labels, idx_train, idx_val, idx_test = load_data()

class GraphConvolution(Module):
    """
     Attributes
     ----------
     in_features: int
         The size of the image convolution input feature vector, namely $|H^{(l)}|$
     out_features: int
         The size of the image convolution output vector, namely $|H^{(l+1)}|$
     bias: bool
         Whether to use the offset vector, the default is True, that is, the default is to use the offset vector
     weight: Parameter
         Trainable parameters in graph convolution,

     Methods
     -------
     __init__(self, in_features, out_features, bias=True)
         The constructor of the graph convolution, defines the size of the input feature, the size of the output vector, whether to use offset, parameters
     reset_parameters(self)
         Initialize the parameters in the graph convolution
     forward(self, input, adj)
         Forward propagation function, input is the feature input, and adj is the transformed adjacency matrix $N(A)=D^{-1}\tilde{A}$. Completing the calculation logic of forward propagation, $N(A) H^{(l)} W^{(l)}$
     __repr__(self)
         Refactored class name expression
     """

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
        # H * W
        support = torch.mm(input, self.weight)

        # N(A) * H * W # Addition aggregation by multiplying
        output = torch.spmm(adj, support)

        if self.bias is not None:

            # N(A) * H * W + b
            return output + self.bias
        else:

            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class GCN(nn.Module):
    '''
     Attributes
     ----------
     nfeat: int
         The size of the input feature of the graph convolution
     nhid: int
         The size of the hidden layer vector of the graph convolution,
         that is, the size of the output vector of the first layer of the convolutional layer
     n_class: int
         Number of classifier categories
     dropout: float
         The size of the dropout probability

     Methods
     -------
     __init__(self, n_feat, n_hid, n_class, dropout)
         Two-layer graph convolutional neural network constructor, defining the dimension of the input feature, the dimension of the hidden layer, the number of classifier categories, and the dropout rate
     forward(self, x, adj)
         Forward propagation function, x is the input feature of the graph network, adj is the adjacency matrix that has been transformed $N(A)$
     '''

    def __init__(self, n_feat, n_hids, n_class, dropout):
        super(GCN, self).__init__()

        # Define the layers of graph convolutional layer
        #Crea una lista che rappresenta le dimensioni delle unità di ogni layer della rete, includendo sia l'input che gli strati nascosti.
        layers_units = [n_feat] + n_hids

        #Crea una lista di GraphConvolution che costituiranno gli strati nascosti della rete.
        #Ogni GraphConvolution è definita usando le dimensioni degli strati calcolate in layers_units
        self.graph_layers = nn.ModuleList(
            [GraphConvolution(layers_units[idx], layers_units[idx + 1]) for idx in range(len(layers_units) - 1)]
        )
        #Crea uno strato di output GraphConvolution che mappa l'ultimo strato nascosto al numero di classi di output.
        self.output_layer = GraphConvolution(layers_units[-1], n_class)

        self.dropout = dropout

    def forward(self, x, adj):
        residual = x
        for graph_layer in self.graph_layers:
            #Applica la funzione di attivazione ReLU all'output di ogni strato nascosto.
            x = F.relu(graph_layer(x, adj))
            #Applica dropout all'output di ogni strato nascosto.
            x = F.dropout(x, self.dropout, training=self.training)
            if x.size() == residual.size():
                x += residual
            residual = x
        #Passa l'output dell'ultimo strato nascosto allo strato di output.
        x = self.output_layer(x, adj)
        #return torch.sigmoid(x).squeeze()  # Applica la funzione sigmoide lungo la dimensione corretta
        # Applica una funzione di softmax logaritmica all'output, classifica
        return torch.log_softmax(x, dim=1)

def accuracy(output, labels):

    # restituisce gli indici delle classi predette dall'output della rete neurale e assicura che il tipo dei dati
    # dell'output delle predizioni sia lo stesso di quello delle etichette
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)

# Training hyperparameter configuration
class Args:
    no_cuda = False
    seed = 42
    epochs = 600  # Numero di epoche aumentato
    lr = 0.0002  # Learning rate ridotto ulteriormente
    weight_decay = 1e-3
    n_hid = [64, 32]  # Tre livelli nascosti con dimensioni maggiori
    dropout = 0.5

args = Args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

input_features = features.shape[1]
model = GCN(n_feat=input_features,
            n_hids=args.n_hid,
            n_class=2,
            dropout=args.dropout)

optimizer = optim.Adam(model.parameters(),
                       lr=args.lr,
                       weight_decay=args.weight_decay)

# Model training function, epoch is the number of iterations
def train(epoch):
    # Record the start time of the epoch iteration
    t = time.time()
    # Mark the GCN model is in train mode
    model.train()
    # In each epoch, you need to clear the previously calculated gradient
    optimizer.zero_grad()

    # Input the graph network input feature and the transformed adjacency matrix adj into the graph convolutional neural network GCN model, and the output is obtained through forward propagation,
    # which is the predicted probability of the classification category
    output = model(features, adj)

    # Find the corresponding output probability and label according to the data index of the training set,
    # and then calculate the loss and accuracy
    loss_train = F.cross_entropy(output[idx_train], labels[idx_train])
    acc_train = accuracy(output[idx_train], labels[idx_train])

    # Error back propagation
    loss_train.backward()

    # The optimizer starts to optimize the trainable parameters in GCN
    optimizer.step()

    # Use the validation set data to verify the epoch training results.
    # The verification process needs to close the train mode and open the eval model
    model.eval()

    # Same forward propagation
    output = model(features, adj)

    # Find the corresponding output probability and label according to the data index of the validation set,
    # and then calculate the loss and accuracy
    loss_val = F.cross_entropy(output[idx_val], labels[idx_val])
    acc_val = accuracy(output[idx_val], labels[idx_val])

    # Print all the results and the time required
    print('Epoch: {:04d}'.format(epoch + 1),
          'loss_train: {:.4f}'.format(loss_train.item()),
          'acc_train: {:.4f}'.format(acc_train.item()),
          'loss_val: {:.4f}'.format(loss_val.item()),
          'acc_val: {:.4f}'.format(acc_val.item()),
          'time: {:.4f}s'.format(time.time() - t))

# Record the start time of model training
t_start = time.time()
# Start iterative training of the GCN model, the number of iterations is set to args.epochs
for epoch in range(args.epochs):
    train(epoch)

print("Model training is complete!")
print("Total model training time: {:.4f}s".format(time.time() - t_start))

def test():
    # First mark the model as eval mode
    model.eval()
    # Input the graph network input feature and the transformed adjacency matrix adj into the two-layer graph convolutional neural network GCN model,
    # and the output is obtained through forward propagation, which is the predicted probability of the classification category
    output = model(features, adj)
    # Find the corresponding output probability and label according to the data index of the test set,
    # and then calculate the loss and accuracy
    loss_test = F.cross_entropy(output[idx_test], labels[idx_test])
    acc_test = accuracy(output[idx_test], labels[idx_test])

    y_pred = output[idx_test].max(1)[1]  # Get the predicted labels

    # Detach y_pred from the computational graph and convert it to a NumPy array
    y_pred_np = y_pred.detach().cpu().numpy()
    labels_np = labels[idx_test].cpu().numpy()

    f1 = f1_score(labels_np, y_pred_np, average='weighted')
    precision = precision_score(labels_np, y_pred_np, average='weighted')
    recall = recall_score(labels_np, y_pred_np, average='weighted')

    # Since precision_recall_curve requires probability estimates of the positive class, and this is a binary classification,
    # you should use the probability of the positive class.
    y_prob = torch.softmax(output[idx_test], dim=1)[:, 1].detach().cpu().numpy()

    precision_c, recall_c, thresholds = precision_recall_curve(labels_np, y_prob)
    plt.plot(recall_c, precision_c, label='Precision-Recall curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend()
    plt.show()

    print("Test set results:",
          "loss= {:.4f}".format(loss_test.item()),
          "accuracy= {:.4f}".format(acc_test.item()),
          "f1= {:.4f}".format(f1),
          "precision= {:.4f}".format(precision),
          "recall= {:.4f}".format(recall))

test()

