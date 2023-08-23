from rdkit import Chem
import numpy as np
import torch
from torch_geometric.data import Data
from torch.utils.data import Dataset


def atom_feature(atom):
    """
    Returns a list of atom features for one graph
    """
    return [
        atom.GetAtomicNum(),
        atom.GetDegree(),
        atom.GetNumImplicitHs(),
        atom.GetIsAromatic(),
        atom.GetFormalCharge(),
        atom.GetHybridization(),
        atom.GetExplicitValence()
    ]


def bond_feature(bond):
    """
    Returns a list of bond features for one graph
    """
    return [bond.GetBondType(), bond.GetStereo(), bond.GetIsConjugated()]


def smiles_to_pyg(rxn_smi, y_bonds, y_atoms, is_product=False):
    """
    Returns pyg.Data object from smile string
    """
    if not is_product:
        smiles = rxn_smi.split('>>')
        rxn_smi = smiles[1]
    mol = Chem.MolFromSmiles(rxn_smi)
    if mol is None:
        return None  # Check if object exists

    bond_idx_pairs = (
        (b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in mol.GetBonds()
    )  # Generator of atom indeces for bonds
    # Generating atom pairs for each bond pair
    atom_pairs = [a for (i, j) in bond_idx_pairs for a in ((i, j), (j, i))]
    bonds = (
        mol.GetBondBetweenAtoms(i, j) for (i, j) in atom_pairs
    )  # Finally generating all bonds

    # Using helper functions
    atom_features = [atom_feature(a) for a in mol.GetAtoms()]
    bond_features = [bond_feature(b) for b in bonds]

    # Finally returning graph data object
    return Data(
        edge_index=torch.LongTensor(list(zip(*atom_pairs))),
        x=torch.FloatTensor(atom_features),
        edge_attr=torch.FloatTensor(bond_features),
        y_bonds=torch.FloatTensor(y_bonds),
        y_atoms=torch.FloatTensor(y_atoms),
        smiles=rxn_smi,
    )

class EarlyStopping:
    def __init__(self, patience=5, min_delta=0, deactivate=False):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf
        self.deactivate = deactivate

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False
    
class ReactionDataset(Dataset):
    def __init__(self, smiles, label_0, label_1):
        mols = [smiles_to_pyg(smi, y_bonds, y_atoms) for smi, y_bonds, y_atoms in zip(smiles, label_0, label_1)]
        self.mols = [mol for mol in mols if mol is not None]

    def __len__(self):
        return len(self.mols)
    
    def __getitem__(self, idx):
        return self.mols[idx]
    

def entropy_loss(atom_log, bond_log, y_atom, y_bond):
    """
    Returns entropy loss for atom and bond predictions
    Improves results compared to BCE loss (to avoid sigmoid saturation)
    """
    eps = 1e-7
    atom_loss = -1*torch.sum(y_atom * torch.log(atom_log+eps), dim=0)
    bond_loss = -1*torch.sum(y_bond * torch.log(bond_log+eps), dim=0)
 

    return atom_loss + bond_loss