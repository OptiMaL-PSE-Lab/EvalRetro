from itertools import permutations
import pandas as pd
import os
import json
from rdkit import Chem 

import torch
from torch import nn
from time import time

import dgl
from dgllife.utils import smiles_to_bigraph, WeaveAtomFeaturizer, CanonicalBondFeaturizer
from functools import partial

from src.localtransform.scripts.dataset import combine_reactants, get_bonds, get_adm
from src.localtransform.scripts.utils import init_featurizer, load_model, pad_atom_distance_matrix, predict
from src.localtransform.scripts.get_edit import get_bg_partition, combined_edit
from src.localtransform.LocalTemplate.template_collector import Collector

atom_types = ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe',
             'As', 'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb', 'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se', 'Ti',
             'Zn', 'H', 'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr', 'Cr', 'Pt', 'Hg', 'Pb',
             'W', 'Ru', 'Nb', 'Re', 'Te', 'Rh', 'Ta', 'Tc', 'Ba', 'Bi', 'Hf', 'Mo', 'U', 'Sm', 'Os', 'Ir',
             'Ce', 'Gd', 'Ga', 'Cs']

def demap(smiles):
    mol = Chem.MolFromSmiles(smiles)
    [atom.SetAtomMapNum(0) for atom in mol.GetAtoms()]
    return Chem.MolToSmiles(mol)

class LocalTransform():
    def __init__(self, TOPK, dataset="USPTO_480k", device='cuda:0'):
        self.root_dir = os.path.dirname(os.path.abspath(__file__))
        self.data_dir = self.root_dir + '/data/%s' % dataset
        self.k_eval = TOPK
        self.config_path = self.root_dir + '/data/configs/default_config'
        self.model_path = self.root_dir +'/models/LocalTransform_%s.pth' % dataset
        self.device = torch.device(device) if torch.cuda.is_available() else torch.device('cpu')
        self.args = {'data_dir': self.data_dir, 'model_path': self.model_path, 'config_path': self.config_path, 'device': self.device, 'mode': 'test'}
        self.template_dicts, self.template_infos = self.load_templates()
        self.model, self.graph_function = self.init_model()
    

    def load_templates(self):
        template_dicts = {}
        for site in ['real', 'virtual']:
            template_df = pd.read_csv('%s/%s_templates.csv' % (self.data_dir, site))
            template_dict = {template_df['Class'][i]: template_df['Template'][i].split('_') for i in template_df.index}
            template_dicts[site[0]] = template_dict
        template_infos = pd.read_csv('%s/template_infos.csv' % self.data_dir)
        template_infos = {template_infos['Template'][i]: {
            'edit_site': eval(template_infos['edit_site'][i]),
            'change_H': eval(template_infos['change_H'][i]), 
            'change_C': eval(template_infos['change_C'][i]), 
            'change_S': eval(template_infos['change_S'][i])} for i in template_infos.index}
        return template_dicts, template_infos

    def init_model(self):
        self.args = init_featurizer(self.args)
        model = load_model(self.args)
        model.eval()
        smiles_to_graph = partial(smiles_to_bigraph, add_self_loop=True)
        node_featurizer = WeaveAtomFeaturizer(atom_types=atom_types)
        edge_featurizer = CanonicalBondFeaturizer(self_loop=True)
        graph_function = lambda s: smiles_to_graph(s, node_featurizer = node_featurizer, edge_featurizer = edge_featurizer, canonical_atom_order = False)
        return model, graph_function

    def make_inference(self, reactant_list):
        topk = self.k_eval
        fgraphs = []
        dgraphs = []
        for smiles in reactant_list:
            mol = Chem.MolFromSmiles(smiles)
            fgraph = self.graph_function(smiles)
            dgraph = {'atom_distance_matrix': get_adm(mol), 'bonds':get_bonds(smiles)}
            dgraph['v_bonds'], dgraph['r_bonds'] = dgraph['bonds']
            fgraphs.append(fgraph)
            dgraphs.append(dgraph)
        bg = dgl.batch(fgraphs)

        bg.set_n_initializer(dgl.init.zero_initializer)
        bg.set_e_initializer(dgl.init.zero_initializer)
        adm_lists = [graph['atom_distance_matrix'] for graph in dgraphs]
        adms = pad_atom_distance_matrix(adm_lists)
        bonds_dicts = {'virtual': [torch.from_numpy(graph['v_bonds']).long() for graph in dgraphs], 'real': [torch.from_numpy(graph['r_bonds']).long() for graph in dgraphs]}
    
        with torch.no_grad():
            pred_VT, pred_RT, _, _, pred_VI, pred_RI, attentions = predict(self.args, self.model, bg, adms, bonds_dicts)
            pred_VT = nn.Softmax(dim=1)(pred_VT)
            pred_RT = nn.Softmax(dim=1)(pred_RT)
            v_sep, r_sep = get_bg_partition(bg, bonds_dicts)
            start_v, start_r = 0, 0
            predictions = []
            for i, (reactant) in enumerate(reactant_list):
                end_v, end_r = v_sep[i], r_sep[i]
                virtual_bonds, real_bonds = bonds_dicts['virtual'][i].numpy(), bonds_dicts['real'][i].numpy()
                pred_vi, pred_ri = pred_VI[i].cpu(), pred_RI[i].cpu()
                pred_v, pred_r = pred_VT[start_v:end_v], pred_RT[start_r:end_r]
                prediction = combined_edit(virtual_bonds, real_bonds, pred_vi, pred_ri, pred_v, pred_r, topk*10)
                predictions.append(prediction)
                start_v = end_v
                start_r = end_r
        return predictions
            
    def predict_product(self, reactant_list, verbose=0):
        results_product = {}
        if isinstance(reactant_list, str):
            reactant_list = [reactant_list]

        predictions = self.make_inference(reactant_list)

        for i, (reactant, prediction) in enumerate(zip(reactant_list, predictions)):
            pred_types, pred_sites, scores = prediction
            collector = Collector(reactant, self.template_infos, 'nan', False, verbose = verbose > 1)
            for k, (pred_type, pred_site, score) in enumerate(zip(pred_types, pred_sites, scores)):
                template, H_code, C_code, S_code, action = self.template_dicts[pred_type][pred_site[1]]
                pred_site = pred_site[0]
                if verbose > 0:
                    print ('%dth prediction:' % (k+1), template, action, pred_site, score)
                collector.collect(template, H_code, C_code, S_code, action, pred_site, score)
                if len(collector.predictions) >= self.k_eval:
                    break
            sorted_predictions = [k for k, v in sorted(collector.predictions.items(), key=lambda item: -item[1]['score'])]
            results_product[f"set_{i}"] = sorted_predictions[:self.k_eval]
                
        return results_product

if __name__ == "__main__":
    from tqdm import tqdm
    url = 'https://github.com/connorcoley/rexgen_direct/blob/master/human/benchmarking.xlsx?raw=true'
    human_benchmark_rxns = pd.read_excel(url)['Reaction smiles'][:80]
    model = LocalTransform(3)

    batch_size = 10
    products = []
    for rxns in tqdm(zip(*(iter(human_benchmark_rxns),) * batch_size), total = len(human_benchmark_rxns)//batch_size):
        reactant_list = [rxn.split('>>')[0] for rxn in rxns]
        # set atom map number 0 for reactants
        results_dict = model.predict_product(reactant_list, verbose = 0)