import os
import pickle

import logging
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from rdkit import Chem
from tqdm import tqdm


project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) 
results_path = os.path.join(project_root, 'results')

# Add logging for info to terminal
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class Alg(ABC):
    def __init__(self, algname:str, check_invalid_smi:bool, skip_blank_lines:bool, remove_stereo:bool, data_dir):
        self._name = algname    
        self.data_dir = os.path.join(data_dir, f'{self._name}', f'{self._name}_processed.csv')
        self.data_dir_cleaned = os.path.join(data_dir, f'{self._name}', f'{self._name}_cleaned.csv')
        self.file_inv_smiles = os.path.join(data_dir, f'{self._name}', f'{self._name}_inv_smiles.csv')
            
        self.result_dir = os.path.join(results_path, f'{self._name}')

        self._skip = skip_blank_lines
        self.remove_stereo = remove_stereo
        self.check_smi = check_invalid_smi
        
        # Canonicalize all smiles within dataset
        self.canonicalize()

        #Checking for invalid smiles 
        if os.path.exists(self.data_dir_cleaned):
            loaded_data = pd.read_csv(self.file_inv_smiles)
            self.invalid_smi = loaded_data.to_dict()
        elif self.check_smi:
            self.invalid_smi = self.detect_inv_smiles()
        else: 
            self.invalid_smi = "Not checked"


    @abstractmethod
    def get_data(self):
        """ 
        Yielding predictions for targets to be used for eval metrics, this depends on datafile for alg
        """
        pass

    def evaluate(self, metric, args): 
        """ 
        Evaluating algorithm on particular metric
        """
        try:
            results = pd.DataFrame.from_dict(metric(self, args), orient='index')
            print_results = self._quick_eval(results, metric.index)
            results.to_csv(os.path.join(results_path, f'{self._name}', f'{metric.name}.csv'))
            return print_results
        except OSError:
            results = pd.DataFrame.from_dict(metric(self, args), orient='index')
            os.makedirs(os.path.join(results_path, f'{self._name}'))
            results.to_csv(os.path.join(results_path, f'{self._name}', f'{metric.name}.csv'))
            return print_results
        except TypeError:
            logger.exception(f"Something went wrong with {metric.name} evaluation for {self._name} dataset.")
            return None    

    def detect_inv_smiles(self):
        """
        Detects invalid smiles in processed.csv file 
        Removes invalid smiles and saves data in cleaned.csv file
        Saves invalid smiles in inv_smiles.csv file
        Returns % invalid smiles & (smile + row index in original csv)
        """
        # Helper function to catch invalid smiles
        def catch(func, smi, idx):
            try: 
                x = func()
                if x is None:
                    raise
                return (1,None)
            except Exception:
                return (smi,idx)

        # Add logger info: 
        logger.info(f"Checking for invalid smiles in {self._name} dataset.\n Results are written to {self.file_inv_smiles} and {self.data_dir_cleaned}")

        df_alg = pd.read_csv(self.data_dir, skip_blank_lines=self._skip)
        all_reactants = df_alg["reactants"].tolist()
        invalid_smiles = []
        total_no_reactants = sum(1 if isinstance(item, str) else 0 for item in all_reactants)
        for idx, react_set in enumerate(tqdm(all_reactants)):
            if isinstance(react_set,str):
                check_smiles = catch(lambda: Chem.MolToSmiles(Chem.MolFromSmiles(react_set), kekuleSmiles=True),react_set, idx)
                if check_smiles != (1,None):
                    invalid_smiles.append(check_smiles)
            else:
                continue    

        indeces = [index for _,index in invalid_smiles]
        df_alg_cleaned = df_alg.drop(indeces)
        df_alg_cleaned = df_alg_cleaned.iloc[:,-3:]
        # reset indeces
        df_alg.reset_index(drop=True, inplace=True)
        
        dict_inv_smiles = {"Index": [index for _,index in invalid_smiles], "Smile": [smi for smi,_ in invalid_smiles]}
        df_inv_smiles = pd.DataFrame(dict_inv_smiles)

        df_inv_smiles.to_csv(self.file_inv_smiles)
        df_alg_cleaned.to_csv(self.data_dir_cleaned, index=False)
        smiles_inv_percentage = len(indeces) / total_no_reactants
        # Save percentage of invalid smiles in pickle file
        dir_results = os.path.join(results_path, f'{self._name}')
        os.makedirs(dir_results, exist_ok=True)
        with open(os.path.join(dir_results, 'Inv_smi.pickle'), 'wb') as handle:
            pickle.dump(smiles_inv_percentage, handle)

        return dict_inv_smiles

    def canonicalize(self):
        """ 
        Canonicalizes all smiles in dataset for fair evaluation 
        If necessary, removes stereo information too (fwd model may not incorporate stereo information)
        """
        # Add logging info

        logger.info(f"Canonicalizing smile strings from {self._name} dataset.\n Results are written to {self.data_dir}")

        df_alg = pd.read_csv(self.data_dir,skip_blank_lines=self._skip)
        reactants = df_alg["reactants"].tolist()
        targets = df_alg["target"].tolist()
        
        for index, (targ, react) in enumerate(tqdm(zip(targets,reactants), total=len(reactants))):
            if isinstance(react,str):
                try:
                    react = react.split(".")
                    react.sort(key=len)
                    react = ".".join(react)
                    mol_rct = Chem.MolFromSmiles(react)
                    mol_tar = Chem.MolFromSmiles(targ)
                    df_alg.at[index, "reactants"] = Chem.MolToSmiles(mol_rct, kekuleSmiles=True, canonical=True)
                    df_alg.at[index, "target"] = Chem.MolToSmiles(mol_tar, kekuleSmiles=True, canonical=True)
                except Exception:
                    continue
                    # Will throw errors from invalid smiles
            else:
                if isinstance(targ,str):
                    df_alg.at[index, "reactants"] = "Failed"

        df_alg = df_alg.iloc[:,-3:]
        df_alg.to_csv(self.data_dir, index=False)

    def stereo(self,target_smile, reactants):
        """
        Removes stereo information from smiles
        """
        mol_target = Chem.MolFromSmiles(target_smile)
        Chem.RemoveStereochemistry(mol_target)
        target_smile = Chem.MolToSmiles(mol_target, kekuleSmiles=True)
        for index, react in enumerate(reactants):
            mol_rct = Chem.MolFromSmiles(react)
            Chem.RemoveStereochemistry(mol_rct)
            reactants[index] = Chem.MolToSmiles(mol_rct, kekuleSmiles=True)
        
        return target_smile, reactants
    
    def _quick_eval(self, results:pd.DataFrame, index_name:str) -> float:
        """
        Allows for the presentation of results in quick fashion
        """
        if index_name == 'top_k':
            print_results = results.iat[-1,-1]
        elif index_name == 'invsmi':
            print_results = 1 - results.iat[-1,-1] # convert invalidity into validity metric
        else:
            print_results = results[index_name].mean()    
        return print_results

class LineSeparated(Alg):
    """ 
    Class for algorithms that have a line-separated datafile for predictions
    """
    def __init__(self, algname, check_invalid_smi, skip_lines=False, check_stereo=True, data_dir=None):
        super().__init__(algname, check_invalid_smi, skip_lines, check_stereo, data_dir)
    
    def get_data(self, metric_name:str):
        uncleaned_metrics = ['invsmi', 'topk']

        if self.check_smi and metric_name not in uncleaned_metrics:
            df_alg = pd.read_csv(self.data_dir_cleaned, skip_blank_lines=self._skip, index_col=0)
        else:
            df_alg = pd.read_csv(self.data_dir, skip_blank_lines=self._skip,index_col=0)
        
        rows = pd.isnull(df_alg).any(axis=1)
        rows = rows.to_numpy().nonzero()[0]
        old_row = -1

        for row in tqdm(rows):
        # Ensuring that there is at least 1 prediction for each target
            if row-old_row != 2:
                df_new = df_alg.iloc[old_row+1:row]

                if metric_name == "scscore":
                    reactants = df_new.iloc[0:]["reactants"].str.split(".").tolist()
                    yield (df_new.iloc[0]["target"], reactants)

                elif metric_name in ['dup','rt']:
                    yield (df_new.iloc[0]["target"], df_new.iloc[1:]["reactants"].tolist())

                elif metric_name in ['topk', 'invsmi']:
                    gt = df_new.iloc[0]["reactants"]
                    reactants = df_new.iloc[1:]["reactants"].tolist()
                    yield gt, reactants

                elif metric_name == 'div':
                    # Return data in format reactants>>product
                    target = df_new.iloc[0]["target"]
                    reactants = df_new.iloc[1:]["reactants"].tolist()
                    rxn = [reactant + ">>" + target for reactant in reactants]
                    yield target, rxn
            old_row = row


class IndexSeparated(Alg):
    """ 
    Class for algorithms that have an index-separated datafile for predictions
    """
    def __init__(self, algname, check_invalid_smi, skip_lines=True, check_stereo=True, data_dir=None):
        super().__init__(algname, check_invalid_smi, skip_lines, check_stereo, data_dir)

    def get_data(self, metric_name:str):
        uncleaned_metrics = ['invsmi', 'topk']

        if self.check_smi and metric_name not in uncleaned_metrics:
            df_alg = pd.read_csv(self.data_dir_cleaned, skip_blank_lines=self._skip, index_col=0)
        else:
            df_alg = pd.read_csv(self.data_dir, skip_blank_lines=self._skip, index_col=0)
        
        indeces = df_alg.index.tolist()
        rows = self.find_indeces(indeces)
        rows.append(df_alg.shape[0]+1)
        old_row = 0
        for row in tqdm(rows):
            # Ensuring that there is at least 1 prediction for each target
            if row-old_row != 1:
                df_new = df_alg.iloc[old_row:row]

                if metric_name == "scscore":
                    reactants = df_new.iloc[0:]["reactants"].str.split(".").tolist()
                    yield (df_new.iloc[0]["target"], reactants)

                elif metric_name in ['dup','rt']:
                    yield (df_new.iloc[0]["target"], df_new.iloc[1:]["reactants"].tolist())

                elif metric_name in ['topk', 'invsmi']:
                    
                    gt = df_new.iloc[0]["reactants"]
                    reactants = df_new.iloc[1:]["reactants"].tolist()
                    yield gt, reactants

                elif metric_name == 'div':
                    # Return data in format reactants>>product
                    target = df_new.iloc[0]["target"]
                    reactants = df_new.iloc[1:]["reactants"].tolist()
                    rxn = [reactant + ">>" + target for reactant in reactants]
                    yield target, rxn
            old_row = row



    def find_indeces(self, indeces_list):
        idx_list = []
        for i in range(len(indeces_list)):
            if i == 0:
                continue
            if indeces_list[i] != indeces_list[i-1]:
                idx_list.append(i)
        return idx_list