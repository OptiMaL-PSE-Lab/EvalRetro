""" 
This package was taken from: https://github.com/connorcoley/rexgen_direct
Full credits go to the authors.
"""
import os
import logging
import warnings
from collections import defaultdict
from rdkit import Chem

# Supress FutureWarnings from tf library raised through rexgen_direct
warnings.simplefilter(action='ignore', category=FutureWarning)

from src.rexgen_direct.rank_diff_wln.directcandranker import DirectCandRanker
from src.rexgen_direct.core_wln_global.directcorefinder import DirectCoreFinder


# Disable tensorflow warning 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def init_fwd(k_eval):
    directcorefinder = DirectCoreFinder()
    directcorefinder.load_model()
    directcandranker = DirectCandRanker(TOPK=k_eval)
    directcandranker.load_model()

    return directcorefinder, directcandranker

def gcn_forward(reactants, directcorefinder:DirectCoreFinder, directcandranker:DirectCandRanker):
    """  
    Implementation of gcn forward model for all predicted reactants for single target 
    """
    predictions = defaultdict()

    for i,react in enumerate(reactants):
        pred_k = []
        
        try: 
            (react, bond_preds, bond_scores, cur_att_score) = directcorefinder.predict(react)
            outcomes = directcandranker.predict(react, bond_preds, bond_scores)
            # Canonicalize smile from outcome prediction
            for outcome in outcomes: 
                smiles_can = [Chem.MolToSmiles(Chem.MolFromSmiles(smi), canonical=True, kekuleSmiles=True) for smi in outcome['smiles']]
                pred_k += smiles_can
        except Exception as e:
            # Log something here to warn user 
            print(f"{e}")
            pred_k = []
        finally:
            predictions.update({f"set_{str(i)}": pred_k})
    
    return predictions

    
if __name__ == "__main__":
    react = ["CN(C)c1cc(S(C)(=O)=O)ccc1-n1ncc2c(OCc3ccccc3)ncnc21", "COc1ncnc2c1cnn2-c1ccc(S(C)(=O)=O)cc1N(C)C","C=O.CNc1cc(S(C)(=O)=O)ccc1-n1ncc2c(O)ncnc21"]
    directcorefinder, directcandranker = init_fwd(2)
    predictions = gcn_forward(react, directcorefinder, directcandranker)
    for pred in predictions.values():
        print(pred)
        print("CN(C)c1cc(S(C)(=O)=O)ccc1-n1ncc2c(O)ncnc21")