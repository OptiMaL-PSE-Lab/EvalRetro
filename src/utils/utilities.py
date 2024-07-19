import argparse
import os
import torch
import numpy as np
import warnings
from rdkit import Chem
from rxnfp.tokenization import SmilesTokenizer
from transformers import BertModel,logging

logging.set_verbosity_warning()

""" 
Input: All scscores for test set, output: dict with scscore diff for each target in test set 
"""

# Specify functions to be imported
__all__ = ['calculate_diff_scscore', 'calculate_rtacc', 'calculate_rtcov', 'load_model_tokenizer', 'adjust_smiles','top_k_accuracy']

def calculate_diff_scscore(scores:dict):
    """ 
     Used to calculate difference in target/reactants sc_score and average along k predicted reactants
    """
    sc_differences = {"SCScore": float, "SCScore_gt": float}
    max_sc = []
    gt_rct = np.array(scores["reactants"][1])
    for i in scores["reactants"]:
        rct = np.array(scores["reactants"][i])
        if np.sum(rct)>0 and i != 1:
            rct = np.max(rct)
            max_sc.append(rct)
    with warnings.catch_warnings():
        warnings.filterwarnings("error")
        avg = np.average(np.array(max_sc))
        sc_differences["SCScore"] = scores["target"] - avg
        sc_differences["SCScore_gt"] = scores["target"] - np.max(gt_rct)

    return sc_differences

def calculate_rtacc(fwd_preds, target):
    """ 
    Calculate round-trip accuracy for a given target 
    Returns: 1) Average accuracy for all k predictions 
                2) Accuracy up until k prediction e.g. top 2 -> average of 1 and 2
    """
    vals = []
    for pred in fwd_preds.values():
        val = target in pred
        vals.append(val)
        
    rt_acc = np.array(vals)
    rt_acc.astype(int)
    rt_acc = np.average(rt_acc, axis=0)
    # Add top k prediction to the dictionary
    #* This will only return values up to len(val) -> if there a less valid smiles than k_back, no value will be returned
    top_k_acc = {f"acc_top_{i+1}":np.round(np.average(vals[:i+1], axis=0),3) for i,_ in enumerate(vals)}

    return rt_acc, top_k_acc

def calculate_rtcov(fwd_preds, target, k_pred_retro):
    """ 
    Calculate round-trip coverage for a given target
    Returns: 1) 1 if target is in any of the predicted product
                2) Coverage up to k predictions
    """
    whole_cov = 0
    top_k_cov = {f"cov_top_{i+1}":0 for i in range(k_pred_retro)}
    for i, pred in enumerate(fwd_preds.values()):
        val = target in pred
        if val: 
            top_k_cov = {f"cov_top_{j+1}":int(1) if i <= j else int(0) for j in range(k_pred_retro)}
            return 1, top_k_cov
    return whole_cov, top_k_cov    

def load_model_tokenizer(model, force_no_cuda=False):
    
    path_rxnfp = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path =  os.path.join(path_rxnfp,'rxnfp', model)

    tokenizer_vocab_path = os.path.join(model_path, 'vocab.txt')
    
    device = torch.device("cuda" if (torch.cuda.is_available() and not force_no_cuda) else "cpu")

    model = BertModel.from_pretrained(model_path)
    model = model.eval()
    model.to(device)

    tokenizer = SmilesTokenizer(
        tokenizer_vocab_path
    )

    return model, tokenizer
 

def adjust_smiles(reactant_smiles, metric_name):
    """
    This is called in case the uncleaned datasets are used for evaluation.
    Since the datasets are not cleaned, some smiles are invalid and cause errors when parsing.
    For the metrics, the smiles are popped from the list of reactants, if they are invalid.
    """
    if metric_name == 'div':
        reactants = [rxn.split('>>')[0] for rxn in reactant_smiles]
    elif metric_name == 'sc':
        reactants = ['.'.join(rxn) for rxn in reactant_smiles]
    else:
        reactants = reactant_smiles

    for i,rxn in enumerate(reactants):
        try:
            Chem.MolToSmiles(Chem.MolFromSmiles(rxn))
        except:
            reactant_smiles.pop(i)
    return reactant_smiles

def top_k_accuracy(gt, reactants):
    """ 
    Assumes smiles have been canonicalized (this is done when initializing the class in alg_classes)
    """
    for k, pred in enumerate(reactants):
        if gt == pred:
            return 1, k+1
    
    return 0, k+1
    

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')