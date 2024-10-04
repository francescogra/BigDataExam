import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
import os
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import StratifiedShuffleSplit

def encode_onehot(labels: np.ndarray) -> np.ndarray:
    """Convert labels to one-hot encoding."""
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)), dtype=np.int32)
    return labels_onehot

def normalize(mx: sp.spmatrix) -> sp.spmatrix:
    """Row-normalize sparse matrix."""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def sparse_mx_to_torch_sparse_tensor(sparse_mx: sp.spmatrix) -> torch.sparse.FloatTensor:
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse_coo_tensor(indices, values, shape)

def load_data():
    """Load and preprocess dataset."""
    base_path = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.abspath(os.path.join(base_path, "../data_uni"))

    # Carica il dataset CSV
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

    edges_unordered = np.genfromtxt(os.path.join(data_path, "network.txt"), dtype=np.int32)
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
    adj_resampled = adj_resampled + adj_resampled.T.multiply(adj_resampled.T > adj_resampled) - adj_resampled.multiply(adj_resampled.T > adj_resampled)
    adj_resampled = normalize(adj_resampled + sp.eye(adj_resampled.shape[0]))

    # Stratified train-validation-test split
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
    train_val_idx, test_idx = next(sss.split(features_resampled, np.argmax(labels_resampled, axis=1)))

    sss_val = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=43)  # Different random_state
    train_idx, val_idx = next(sss_val.split(features_resampled[train_val_idx], np.argmax(labels_resampled[train_val_idx], axis=1)))

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
