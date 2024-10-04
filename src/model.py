import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import math
from torch.nn.modules.module import Module

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

        return torch.log_softmax(x, dim=1) # originale
        #return torch.softmax(x, dim=1)
        #return torch.sigmoid(x)
        #return torch.tanh(x)
        #return F.softplus(x)


"""
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
"""