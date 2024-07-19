""" 
Define all evaluation metrics in this module
Currently included: ScScore, Round-trip accuracy and coverage, Class diversity, Duplicity and Invalidity
* * Future inclusion: Novelty metric (possibly based on templates) 

"""
import os
import pickle

import numpy as np
from collections import defaultdict, Counter
from rdkit import Chem
from rxnfp.transformer_fingerprints import RXNBERTFingerprintGenerator

from src.utils.sc_score import SCScorer 
from src.utils.utilities import *
from src.utils.fwd_mdls import gcn_forward, init_fwd


project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_path = os.path.join(project_root, 'data')

# Enable use of separate .dll for scscore and rxnfp
os.environ['KMP_DUPLICATE_LIB_OK']='True'

def eval_scscore(alg:object, args):
    """ 
     Calculation of ScScore proposed by Coley
     """

    k_pred_retro = args.k_back
    eval_scscore.name = "SCScore"
    eval_scscore.index = "SCScore"
    eval_scscore.k = args.k_back
    cleaned_data = alg.check_smi

    model = SCScorer()
    model.restore(os.path.join(project_root, 'src', 'sc_models', 'full_reaxys_model_2048bool', 'model.ckpt-10654.as_numpy.json.gz'), FP_len=2048)
    
    diff_scores = defaultdict()
    # Getting one prediction from test set at a time
    for k ,(smiles_tar, smiles_rxn) in enumerate(alg.get_data("scscore")):
        
        # Getting ScScore for target in question
        scores = defaultdict(dict)
        _, scores["target"] = model.get_score_from_smi(smiles_tar) 

        # Check if uncleaned dataset is used
        if not cleaned_data:
            smiles_rxn = adjust_smiles(smiles_rxn[:k_pred_retro+1], 'sc')
        # Getting ScScores for all predicted reaction sets
        for i, smi_list in enumerate(smiles_rxn[:k_pred_retro+1]):
            scores_rxn = []
            for smi in smi_list: 
                try:
                    _, x = model.get_score_from_smi(smi)
                    scores_rxn.append(x)
                except: 
                    # Log something here
                    print("Exception")
                    scores_rxn.append(0)
            
            scores["reactants"][i+1] = scores_rxn
        
        diff_scores[f"{k}_{smiles_tar}"] = calculate_diff_scscore(scores) # Maybe add id_rxn to dictionary

    return diff_scores 


def round_trip(alg, args):
    """ 
     Calculation of round-trip accuracy and coverage:
     This is defined as % predicted product = target
    """
    
    k_pred_retro, k_pred_forw, fwd_model = args.k_back, args.k_forward, args.fwd_model

    round_trip.name = "Round-trip"
    round_trip.index = "acc_mean"
    round_trip.k = args.k_back
    rt_scores = defaultdict(dict)
    cleaned_data = alg.check_smi
    model_init = init_fwd(k_pred_forw, fwd_model.__name__)

    for k, (smiles_tar, reactants) in enumerate(alg.get_data("rt")):
        
        if not cleaned_data:
            reactants = adjust_smiles(reactants[:k_pred_retro], 'trial')
        if alg.remove_stereo:
            smiles_tar, reactants = alg.stereo(smiles_tar, reactants)
        target = smiles_tar
        reactants = reactants[:k_pred_retro]
        fwd_predictions = fwd_model(reactants, *model_init)
        
        rt_acc, top_k_acc = calculate_rtacc(fwd_predictions, target)
        rt_cov, top_k_cov = calculate_rtcov(fwd_predictions, target, k_pred_retro)
        metrics = {"cov_total":rt_cov, **top_k_cov,"acc_mean":rt_acc, **top_k_acc}
        rt_scores[f"{k}_{smiles_tar}"] = metrics

    return rt_scores


def diversity(alg, args):
    """ 
    Calculation of class diversity as proposed by Schwaller et. al.
    The metric is normalized to 0-1, with 1 being the most diverse i.e. all 10 reaction classes
    """
    diversity.name = "Diversity"
    diversity.index = "No_classes"
    diversity.k = args.k_back
    cleaned_data = alg.check_smi
    
    k_pred_retro = args.k_back
    model, tokenizer = load_model_tokenizer("bert_50k_best")
    rxnfp_generator = RXNBERTFingerprintGenerator(model, tokenizer)
    # Loading in classifer
    path = os.path.join(project_root,'src', 'rxnfp', 'lr_cls.pkl') 
    lr_cls = pickle.load(open(path, 'rb'))
    div_scores = defaultdict(dict)

    for i, (smiles_tar, rxn) in enumerate(alg.get_data("div")):
        # Generate fingerprints
        if not cleaned_data:
            rxn = adjust_smiles(rxn[:k_pred_retro], 'div')
        rxn = rxn[:k_pred_retro]
        fps = rxnfp_generator.convert_batch(rxn)
        rxn_classes = lr_cls.predict(fps)
        classes_count = Counter(rxn_classes)
        classes, counts = zip(*classes_count.items())
        classes, counts = tuple(classes), tuple(counts)
        rxn_classes = len(set(rxn_classes))/10 # Normalizing to 0-1
        div_scores[f"{i}_{smiles_tar}"] = {"No_classes":rxn_classes, "Classes":classes, "Counts":counts}
    return div_scores


def duplicates(alg, args):
    """ 
    Calculation of average number of duplicates in a reaction set
    This metric is normalized from 0-1 where 0 is no duplicates
    """
    duplicates.name = "Duplicates"
    duplicates.index = "dup"
    duplicates.k = args.k_back

    dup = defaultdict(dict)

    for i, (smiles_tar, reactants) in enumerate(alg.get_data("dup")):
        check_dup = args.dup
        # Add rxn to counter
        reactants= reactants[:check_dup]
        counter = Counter(reactants)
        # Get length of keys and rxn
        n_keys = len(counter.keys())
        n_rxn = len(reactants)
        try:
            dup[f'{i}_{smiles_tar}']= {'dup': (n_keys-1)/(n_rxn-1)}
        except ZeroDivisionError:
            dup[f'{i}_{smiles_tar}']= {'dup': 0}
    return dup


def valsmiles(alg, args):
    """
    Calculation of percentage of invalid / valid smiles in dataset.
    Val_k = 1 - Inv_k
    """

    valsmiles.name = "InvSmiles"
    valsmiles.index = "invsmi"
    valsmiles.k = args.invsmiles

    k_pred_retro = args.invsmiles
    top_k_invalid = []

    for _, reactants in alg.get_data("invsmi"):
        dict_invalid = {f'top_{k+1}': 0 for k in range(k_pred_retro)}
        reactants = reactants[:k_pred_retro]
        k_invalid = 0
        for k, rxn in enumerate(reactants):
            try: 
                Chem.MolToSmiles(Chem.MolFromSmiles(rxn), kekuleSmiles=True)
            except Exception:
                k_invalid += 1
                
            # Format results to 3 decimal places
            dict_invalid[f'top_{k+1}'] = k_invalid / (k+1)
        
        top_k_invalid.append(dict_invalid)

    # Average over each top_k to get final percentage for each k
    inv_smiles = []
    for k in range(k_pred_retro):
        top_k = [d[f'top_{k+1}'] for d in top_k_invalid]
        inv_smiles.append(np.round(np.mean(top_k),3))

    inv_smiles = {f'top_{k+1}': inv_smiles[k] for k in range(k_pred_retro)}    
    return inv_smiles


def top_k(alg, args):
    """ 
     Calculation of classic top-k accuracy for sanity checks
    """
    top_k.name = "Top-k"
    top_k.index = "top_k"
    top_k.k = args.k_back

    k_pred_retro = args.k_back

    k_dict = {'1': 0, '3': 0, '5': 0, '10': 0, '20':0, '50':0}
    keys = k_dict.keys()
    for i, (gt, reactants) in enumerate(alg.get_data("topk")):
        acc, k = top_k_accuracy(gt, reactants[:k_pred_retro])
        keys_to_update = [key for key in keys if int(key) >= k]
        for key in keys_to_update:
            k_dict[key] += acc
    for key in keys:
        k_dict[key] /= i

    filtered_k_dict = {key: value for key, value in k_dict.items() if int(key) <= k_pred_retro} 

    return {"top-k accuracy":filtered_k_dict}


# ** Single metrics found below ** 

def eval_scscore_single(target_smile, reactant_smiles):

    model_version = "full_reaxys_model_2048bool"
    model = SCScorer()

    model.restore(os.path.join(project_root, 'src', 'sc_models', model_version, 'model.ckpt-10654.as_numpy.json.gz'), FP_len=2048)
    
    # Getting ScScore for target in question
    scores = defaultdict(dict)
    _, scores["target"] = model.get_score_from_smi(target_smile) 
    reactant_smiles = [reactant_smile.split(".") for reactant_smile in reactant_smiles]
    # Getting ScScores for all predicted reaction sets
    for i, smiles in enumerate(reactant_smiles): 
        scores_rxn = []
        for smi in smiles:
            try:
                _, x = model.get_score_from_smi(smi)
                scores_rxn.append(x)
            except: 
                        # Log something here
                        print("Exception")
                        scores_rxn.append(0)
            
        scores["reactants"][str(i+1)] = scores_rxn
        
        diff_score = calculate_diff_scscore(scores) # Maybe add id_rxn to dictionary

    return diff_score 


def round_trip_single(target_smile, reactant_smiles, args=[2,gcn_forward]):
    """ 
    Calculation of round_trip for single example target and for testing
    """
    k_pred_forw, fwd_model = args[0], args[1]
    reactant_smiles = list(reactant_smiles)
    target = Chem.MolToSmiles(Chem.MolFromSmiles(target_smile), canonical=True, kekuleSmiles=True)
    fwd_predictions = fwd_model(reactant_smiles, k_pred_forw)
    rt_acc, top_k_acc = calculate_rtacc(fwd_predictions, target)
    rt_cov, top_k_cov = calculate_rtcov(fwd_predictions, target, 1)
    metrics = {"cov":rt_cov, **top_k_cov,"acc":rt_acc, **top_k_acc}
    return metrics, fwd_predictions

def diversity_single(target_smile, reactant_smiles):
    """ 
    Calculation of class diversity for single example target and for testing 
    """
    target = Chem.MolToSmiles(Chem.MolFromSmiles(target_smile), canonical=True, kekuleSmiles=True)
    path = os.path.join(project_root, 'src', 'rxnfp', 'lr_cls.pkl')
    # Concatenate strings into right format
    rxn = [reactant + ">>" + target for reactant in reactant_smiles]
    model, tokenizer = load_model_tokenizer("bert_50k_best")
    rxnfp_generator = RXNBERTFingerprintGenerator(model, tokenizer)
    rxn = rxnfp_generator.convert_batch(rxn)
    # The array will be of shape (n_rxns, n_rxnfp)
    rxn = np.array(rxn)
    # Loading in classifer
    lr_cls = pickle.load(open(path, 'rb'))
    pred = lr_cls.predict(rxn)
    return pred, set(pred)

def duplicates_single(target_smile, reactant_smiles):
    """  
    Calculation of duplicates for single example target and for testing
    """
    counter = Counter(reactant_smiles)
    # Get length of keys and rxn
    n_keys = len(counter.keys())
    n_rxn = len(reactant_smiles)
    return ((n_keys-1)/(1-n_rxn), n_keys)