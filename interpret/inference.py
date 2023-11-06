import numpy as np
import pandas as pd
import logging
import torch
from torch_geometric.explain import Explainer
from torch_geometric.explain.algorithm import GNNExplainer
from torch_geometric.explain.metric import fidelity
from pathlib import Path
from tqdm import tqdm
from matplotlib import pyplot as plt
from matplotlib import colormaps as cm
from matplotlib.colors import LinearSegmentedColormap

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.Draw import SimilarityMaps, rdMolDraw2D

from model_utils import smiles_to_pyg
from model import GCN, EGAT, DMPNN
from utils import create_training_labels

logging.basicConfig(filename='figs/inference.log',level=logging.INFO, format='%(name)s  - %(message)s')

stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.INFO)
f = logging.Formatter('%(message)s')
stream_handler.setFormatter(f)
logging.getLogger().addHandler(stream_handler)


def compute_top_k(bond_log, atom_log, gt_bonds, gt_atoms, k=5):
    """
    This function returns a vector of size k with binary numbers 
    """
    top_k = np.zeros(k)
    combined_gt = np.concatenate((gt_bonds, gt_atoms)).reshape(-1)
    combined_log = np.concatenate((bond_log, atom_log)).reshape(-1)
    no_entries = np.where(combined_gt == 1)[0].shape[0] # Check if single-edit prediction
    if no_entries > 1:
        return top_k
    # sort the combined log in descending order
    sorted_log = np.argsort(combined_log)
    sorted_log = sorted_log[::-1]
    top_k_idx = sorted_log[:k]

    for i, idx in enumerate(top_k_idx):
        if combined_gt[idx] == 1:
            top_k[i:] = 1
            return top_k
    
    return top_k

def test_models(model_type, saved_model, rxn_test, k=5):
    
    # no speed up on GPU as batch = 1    
    device = torch.device('cpu')

    if model_type == 'GCN':
        model = GCN(7, 100, n_layers=3)
        model.load_state_dict(torch.load(saved_model, map_location=device))
        model.eval()
    elif model_type == 'EGAT':
        model = EGAT(7, 3, 200, n_layers=4)
        model.load_state_dict(torch.load(saved_model, map_location=device))
        model.eval()
    elif model_type == 'DMPNN':
        model = DMPNN(7, 3, 256, 200, n_layers=5)
        model.load_state_dict(torch.load(saved_model, map_location=device))
        model.eval()
    else:
        raise ValueError("Model type not supported")
    
    model = model.to(device)
    model.eval()
    top_k_total = np.zeros(k)
    for rxn in tqdm(rxn_test):
        data_tup = create_training_labels(rxn)
        data = smiles_to_pyg(data_tup.rxnsmile, data_tup.y_bonds, data_tup.y_atoms)
        data = data.to(device)
        atom_log, bond_log = model(data.x, data.edge_index, data.edge_attr, info_batch=data.batch, return_type="Both", train=False)
        bond_log, atom_log = bond_log.detach().numpy(), atom_log.detach().numpy()
        gt_bonds, gt_atoms = data.y_bonds.detach().numpy(), data.y_atoms.detach().numpy()
        top_k = compute_top_k(bond_log, atom_log, gt_bonds, gt_atoms, k)
        top_k_total += top_k
    
    logging.info(f'The top-{k} accuracy for {model_type} is \n {top_k_total/len(rxn_test)}')

def explain_model(rxn, name, model_type, saved_model):
    device = torch.device('cpu')
    y_bonds = y_atoms = np.zeros(10)
    data = smiles_to_pyg(rxn, y_bonds, y_atoms, is_product=True)
    data = data.to(device)
    if name == 'DMPNN_Test_1':
        hp = 8
        lr = 0.135
    elif name == 'EGAT_Test_2':
        hp = 6
    else: 
        hp = 5
        lr = 0.14
    
    # get number of atom sin smile
    mol = Chem.MolFromSmiles(rxn)
    num_atoms = mol.GetNumAtoms()
    
    if model_type == 'GCN':
        model = GCN(7, 100, n_layers=3)
        model.load_state_dict(torch.load(saved_model, map_location=device))
    elif model_type == 'EGAT':
        model = EGAT(7, 3, 200, n_layers=4) 
        model.load_state_dict(torch.load(saved_model, map_location=device))
    elif model_type == 'DMPNN':
        model = DMPNN(7, 3, 256, 200, n_layers=5)
        model.load_state_dict(torch.load(saved_model, map_location=device))
    else:
        raise ValueError("Model type not supported")
    
    model.eval()
    explainer = Explainer(
        model=model,
        algorithm=GNNExplainer(epochs=1600, lr=lr), 
        explanation_type='model',
        node_mask_type='object',
        model_config=dict(
            mode='multiclass_classification',
            task_level='node',
            return_type='probs',
        
        ),
        threshold_config=dict(
            threshold_type="topk",
            value=hp,
         )
    )
    explanation = explainer(data.x, data.edge_index, edge_attr=data.edge_attr, info_batch=data.batch, return_type="Explain",train=False)
    visualise_explanation(explanation, rxn, name, explanation.target, correction=False)
    explanation._model_args = ['edge_attr', 'return_type', 'train']
    # print prediction
    logging.info(f'\n The prediction for {name} is {explanation.target}')
    logging.info(f'The fidelity for (subgraph, without subgraph) for {name} is {fidelity(explainer, explanation)}')

    return explanation

def explain_model_gt(rxn, name, model_type, saved_model, target):
    device = torch.device('cpu')
    y_bonds = y_atoms = np.zeros(10)
    data = smiles_to_pyg(rxn, y_bonds, y_atoms, is_product=True)
    data = data.to(device)
    
    # get number of atom sin smile
    mol = Chem.MolFromSmiles(rxn)
    num_atoms = mol.GetNumAtoms()
    
    if model_type == 'GCN':
        model = GCN(7, 100, n_layers=3)
        model.load_state_dict(torch.load(saved_model,map_location=device))
    elif model_type == 'EGAT':
        model = EGAT(7, 3, 200, n_layers=4) 
        model.load_state_dict(torch.load(saved_model, map_location=device))
    elif model_type == 'DMPNN':
        model = DMPNN(7, 3, 256, 200, n_layers=5)
        model.load_state_dict(torch.load(saved_model, map_location=device))
    else:
        raise ValueError("Model type not supported")

    model.eval()
    explainer = Explainer(
        model=model,
        algorithm=GNNExplainer(epochs=1200, lr=0.013), 
        explanation_type='phenomenon',
        node_mask_type='object',
        model_config=dict(
            mode='multiclass_classification',
            task_level='node',
            return_type='probs',
        
        ),
        threshold_config=dict(
            threshold_type="topk",
            value=5,
         )
    )
    explanation = explainer(data.x, data.edge_index, edge_attr=data.edge_attr, info_batch=data.batch, return_type="Explain",train=False, target=torch.LongTensor([target]).squeeze())
    visualise_explanation(explanation, rxn, name, explanation.target, correction=True)
    explanation._model_args = ['edge_attr', 'return_type', 'train']

    return explanation

def visualise_explanation(explanation, rxn, name, bond, split=False, custom_weights=None, correction=None):
    folder = name.split('_')[0]
    folder_path = Path(__file__).parent / 'figs' / f'{folder}'
    if folder_path.is_dir():
        pass
    else:
        folder_path.mkdir(parents=True, exist_ok=True)
    
    if split:
        rxn = rxn.split('>>')[1]
    
    if custom_weights:
        custom_weights = np.array(custom_weights)
        mean = np.mean(custom_weights)
        custom_weights = custom_weights - mean
        # all negative weights set to zero
        custom_weights[custom_weights < 0] = 0
        weights = custom_weights.tolist()
    else:
        weights = explanation.node_mask
        weights = torch.squeeze(weights)
        weights = weights.detach().numpy().tolist()
        # Add predicted bond to 
        d2d = rdMolDraw2D.MolDraw2DCairo(1000,1000)
        pmol = Chem.MolFromSmiles(rxn)
        if not correction:
            d2d.DrawMolecule(pmol,highlightAtoms=[],highlightBonds=[bond.item()])
            d2d.FinishDrawing()
            d2d.WriteDrawingText(f'figs/{folder}/bond_{name}.png')

    mol = Chem.MolFromSmiles(rxn)
    AllChem.ComputeGasteigerCharges(mol)
    plt.close(fig)

    # Assign your custom weights to the atoms
    for i, weight in enumerate(weights):
        mol.GetAtomWithIdx(i).SetDoubleProp('_CustomWeight', weight)

    # Generate the similarity map using the custom weights
    PiRdBu_cmap = cm['coolwarm']
    color_map = LinearSegmentedColormap.from_list(
      'PiWG', [PiRdBu_cmap(0), (1.0, 1.0, 1.0), PiRdBu_cmap(1)], N=255)
    fig = SimilarityMaps.GetSimilarityMapFromWeights(mol, weights, colorMap=color_map, contourLines=4, size=(1000, 1000))
    
    if correction:
        name = f'{name}_corrected'
    
    fig.savefig(f'figs/{folder}/{name}.png', dpi=300, bbox_inches='tight')
    
if __name__ == '__main__':
    data_path = Path(__file__).parent / 'data'
    df_test = pd.read_csv(data_path / 'canonicalized_test.csv')
    df_exp_graph = pd.read_csv(data_path / 'graph_explanation.txt')
    df_exp_tf = pd.read_csv(data_path / 'tf_explanation.txt', delimiter=';')
    rxn_test = df_test['rxn'].tolist()
    model_type = 'EGAT'
    saved_model = f'models/{model_type}_model.pt'
    test_models(model_type, saved_model, rxn_test)
    model_type = 'DMPNN'
    saved_model = f'models/{model_type}_model.pt'
    test_models(model_type, saved_model, rxn_test)

    for i, (name, smi, tar) in df_exp_graph.iterrows():
        model_type = name.split('_')[0]
        saved_model = f'models/{model_type}_model.pt'
        rxn_single = smi
        explain_model(rxn_single, name, model_type, saved_model)
        explain_model_gt(rxn_single, name, model_type, saved_model, tar)  

    for i, (name, smi, weight) in df_exp_tf.iterrows():
        rxn_single = smi
        weight = weight.split(',')
        weight = [float(w) for w in weight]
        visualise_explanation(None, rxn_single, name, custom_weights=weight, bond=None)
    