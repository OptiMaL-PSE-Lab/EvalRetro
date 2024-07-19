""" 
These packages were taken from: https://github.com/connorcoley/rexgen_direct and https://github.com/kaist-amsg/LocalTransform
Full credits go to the authors.
"""
import os
import warnings
from collections import defaultdict
from rdkit import Chem

# Supress FutureWarnings from tf library raised through rexgen_direct
warnings.simplefilter(action='ignore', category=FutureWarning)

from src.rexgen_direct.rank_diff_wln.directcandranker import DirectCandRanker
from src.rexgen_direct.core_wln_global.directcorefinder import DirectCoreFinder
from src.localtransform.synthesis import LocalTransform


# Disable tensorflow warning 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def init_fwd(k_eval, model_type):
    if model_type == "gcn_forward":
        directcorefinder = DirectCoreFinder(TOPK=20)
        directcorefinder.load_model()
        directcandranker = DirectCandRanker(TOPK=k_eval)
        directcandranker.load_model()
        instance_models = [directcorefinder, directcandranker]
    elif model_type == "localt_forward":
        localtransform = LocalTransform(TOPK=k_eval)
        instance_models = [localtransform]
    return instance_models


def localt_forward(reactants, localtransform:LocalTransform):
    """  
    Implementation of local transform model for all predicted reactants for single target

    """
    #* Add a heuristic that len(reactant) > 6 to avoid computational errors in batch inference 
    min_len = min([len(rct) for rct in reactants])
    
    if min_len > 6:
        # do faster batch inference
        try: 
            outcomes = localtransform.predict_product(reactants)
            for key, val in outcomes.items():
                outcomes[key] = [Chem.MolToSmiles(Chem.MolFromSmiles(smi), canonical=True, kekuleSmiles=True) for smi in val]
        except Exception as e:
            print(e)      
            outcomes = {}
        return outcomes
    else: 
        predictions = defaultdict()
        # do slower individual inference 
        for i, rct in enumerate(reactants): 
            try: 
                outcome = localtransform.predict_product(rct)["set_0"]
                pred_k = [Chem.MolToSmiles(Chem.MolFromSmiles(smi), canonical=True, kekuleSmiles=True) for smi in outcome]
            except Exception as e: 
                print(e)
                pred_k = []
            finally:
                predictions.update({f"set_{str(i)}": pred_k})
        return predictions


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
            print(e)
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