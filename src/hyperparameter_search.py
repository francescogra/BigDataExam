from itertools import product
import torch
import numpy as np
import torch.optim as optim
from src.train import train, test
from src.model import GCN
from src.config import Args
from src.data_loader import load_data
import os
import sys

# Definire il grid search per i parametri
param_grid = {
    'epochs': [400, 800, 1200],
    'lr': [0.0001, 0.00001, 0.0005, 0.00005, 0.001, 0.005],
    'weight_decay': [4e-5, 2e-5, 4e-6, 4e-7, 4e-3, 4e-2],
    'n_hid': [[64, 32], [128, 64], [32, 16]],
    'dropout': [0.5, 0.4, 0.3, 0.6, 0.7]
}

all_configs = list(product(*param_grid.values()))

def get_last_two_lines(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
        last_two_lines = lines[-2:]
        return ''.join(last_two_lines)

def run_test(config, output_file):
    args = Args()
    args.epochs, args.lr, args.weight_decay, args.n_hid, args.dropout = config
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    adj, features, labels, idx_train, idx_val, idx_test = load_data()
    input_features = features.shape[1]

    model = GCN(n_feat=input_features,
                n_hids=args.n_hid,
                n_class=2,
                dropout=args.dropout)

    if args.cuda:
        model.cuda()
        features = features.cuda()
        adj = adj.cuda()
        labels = labels.cuda()
        idx_train = idx_train.cuda()
        idx_val = idx_val.cuda()
        idx_test = idx_test.cuda()

    optimizer = optim.Adam(model.parameters(),
                           lr=args.lr,
                           weight_decay=args.weight_decay)

    # Redirecting standard output to a file
    orig_stdout = sys.stdout
    sys.stdout = open(output_file, 'w')

    print("Configuration:", config)  # Stampiamo la configurazione prima di eseguire il test

    for epoch in range(args.epochs):
        train(epoch, model, optimizer, features, adj, idx_train, idx_val, labels)
        # Flush stdout to make sure all print statements are written to file
        sys.stdout.flush()

    test(model, features, adj, idx_test, labels)

    # Restore standard output
    sys.stdout.close()
    sys.stdout = orig_stdout


output_dir = 'results'
os.makedirs(output_dir, exist_ok=True)

for i, config in enumerate(all_configs):
    output_file = os.path.join(output_dir, f'output_{i + 1}.txt')  # Definisci il nome del file di output per questa run
    run_test(config, output_file)

    # Otteniamo le ultime due righe dal file di output dopo che il test è stato eseguito e il file è stato chiuso
    last_two_lines = get_last_two_lines(output_file)

    # Costruiamo il nome del file di output usando le ultime due righe
    output_file_name = f'output_{i + 1}_' + last_two_lines.strip().replace(' ', '_').replace('\n', '').replace(':', '=') + '.txt'

    # Rinomina il file di output
    os.rename(output_file, os.path.join(output_dir, output_file_name))

