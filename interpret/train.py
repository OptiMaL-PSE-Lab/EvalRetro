import numpy as np
import pandas as pd
import torch
from torch_geometric.loader import DataLoader
from pathlib import Path
from tqdm import tqdm 

from model_utils import ReactionDataset, EarlyStopping, entropy_loss
from model import GCN, EGAT, DMPNN
from utils import create_training_labels

import warnings

warnings.filterwarnings("ignore")


def create_training_test_loaders(train_rxn, test_rxn, batch_size, shuffle=True):
    len_train = len(train_rxn)

    train_rxn.extend(test_rxn)
    rxn_smi, y_bonds, y_atoms = [], [], []
    for rxn in train_rxn:
        tup = create_training_labels(rxn)
        rxn_smi.append(tup.rxnsmile), y_bonds.append(tup.y_bonds), y_atoms.append(tup.y_atoms)
    
    train_dataset = ReactionDataset(rxn_smi[:len_train], y_bonds[:len_train], y_atoms[:len_train])
    test_dataset = ReactionDataset(rxn_smi[len_train:], y_bonds[len_train:], y_atoms[len_train:])

    node_features = train_dataset[0].num_node_features
    edge_features = train_dataset[0].num_edge_features

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=4)
    return train_loader, test_loader, node_features, edge_features

def train_gnn(loader, model, device, optimizer):
    total_loss = total_examples = 0
    for data in tqdm(loader):
        data = data.to(device)
        optimizer.zero_grad()
        atom_log, bond_log = model(data.x, data.edge_index, data.edge_attr, info_batch=data.batch, return_type="Both", train=True)

        y_bonds, y_atoms = data.y_bonds.view(-1, 1), data.y_atoms.view(-1, 1)
        y_bonds, y_atoms = y_bonds.to(device), y_atoms.to(device)
        loss = entropy_loss(atom_log, bond_log, y_atoms, y_bonds)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        total_examples += y_atoms.shape[0] + y_bonds.shape[0]
    return total_loss / total_examples

@torch.no_grad()
def test_gnn(loader, model, device):
    total_loss = total_examples = 0
    for data in loader:
        data = data.to(device)
        atom_log, bond_log = model(data.x, data.edge_index, data.edge_attr, info_batch=data.batch, return_type="Both", train=True)
        y_bonds, y_atoms = data.y_bonds.view(-1, 1), data.y_atoms.view(-1, 1)
        y_bonds, y_atoms = y_bonds.to(device), y_atoms.to(device)
        loss = entropy_loss(atom_log, bond_log, y_atoms, y_bonds)
        total_loss += loss.item()
        total_examples +=  y_atoms.shape[0] + y_bonds.shape[0]
    return total_loss / total_examples

@torch.no_grad()
def calculate_accuracy(loader, model, device):
    total_examples = correct_examples = 0
    for data in loader:
        data = data.to(device)
        atom_log, bond_log = model(data.x, data.edge_index, data.edge_attr, info_batch=data.batch, return_type="Both", train=True)
        atom_pred = torch.round(atom_log)
        bond_pred = torch.round(bond_log)
        y_bonds, y_atoms = data.y_bonds.view(-1, 1), data.y_atoms.view(-1, 1)
        y_bonds, y_atoms = y_bonds.to(device), y_atoms.to(device)
        atom_correct = torch.sum(atom_pred == y_atoms)
        bond_correct = torch.sum(bond_pred == y_bonds)
        correct_examples += atom_correct.item() + bond_correct.item()
        total_examples += y_atoms.shape[0] + y_bonds.shape[0]
    return correct_examples / total_examples

# Define main function in script
def main(model_type, epochs, batch_size, save_path:Path):
    data_path = Path(__file__).parent / 'data'
    df_train = pd.read_csv(data_path / 'canonicalized_train.csv')
    df_test = pd.read_csv(data_path / 'canonicalized_test.csv')
    rxn_train = df_train['rxn'].tolist()
    rxn_test = df_test['rxn'].tolist()
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    train_loader, test_loader, node_dim, edge_dim  = create_training_test_loaders(rxn_train, rxn_test, batch_size=batch_size, shuffle=True)

    if model_type == 'GCN':
        model = GCN(node_dim, 100, n_layers=3)
    elif model_type == 'EGAT':
        model = EGAT(node_dim, edge_dim, 200, n_layers=4)
    elif model_type == 'DMPNN':
        model = DMPNN(node_dim, edge_dim, 256, 200, n_layers=5)
    else:
        raise ValueError("Requested algorithm not implemented. Please choose from GCN, EGAT, DMPNN.")
    
    model = model.to(device)
    
    early_stopping = EarlyStopping(patience=8)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss = {"train_loss":[], "valid_loss":[]}
    best_val = np.inf
    for epoch in range(epochs):
        model.train()
        train_loss = train_gnn(train_loader, model, device, optimizer)
        loss["train_loss"].append(train_loss)

        model.eval()
        test_loss = test_gnn(test_loader, model, device)
        loss["valid_loss"].append(test_loss)
        
        print(f'Epoch: {epoch}, Train Loss: {train_loss}, Valid Loss: {test_loss}')
        if epoch % 3 == 0:
            # Calculate accuracy
            val_acc = calculate_accuracy(test_loader, model, device)
            print(f'Epoch: {epoch}, Valid Accuracy: {val_acc}')

        if test_loss < best_val:
            best_val = test_loss
            model_path = save_path / f"{epoch}_{model_type}_model.pt"
            torch.save(model.state_dict(), model_path)
            print("New model saved under %s" % model_path)

        if early_stopping.early_stop(test_loss) and not early_stopping.deactivate:
                print("Stopping early due to lack of improvement")
                break

    return loss

if __name__ == '__main__':
    import argparse
    save_path = Path(__file__).parent / 'models'
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', type=str, default='DMPNN', help='GCN, EGAT or DMPNN')
    parser.add_argument('--epochs', type=int, default=60, help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size')
    parser.add_argument('--save_path', type=str, default=save_path, help='path to save model')
    args = parser.parse_args()
    loss = main(args.model_type, args.epochs, args.batch_size, args.save_path)
