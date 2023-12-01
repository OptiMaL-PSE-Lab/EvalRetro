"""
Preprocess functions which are used as wrappers for data extraction 
"""

import numpy as np
import pandas as pd
from rdkit import Chem
from tqdm import tqdm 


def megan_preprocess(obj):
    df_old = obj.df_txt

    rows = pd.isnull(df_old).any(axis=1)
    rows = rows.to_numpy().nonzero()[0]
    old_row = -1
    
    for row in tqdm(rows):
        target_list = []
        pred_length = row - old_row -1
        mol = Chem.MolFromSmiles(df_old.iloc[old_row+1]["target"])
        [a.SetAtomMapNum(0) for a in mol.GetAtoms()]
        target = Chem.MolToSmiles(mol)
        targets = np.tile(target, pred_length)
        target_list = targets.tolist() + [""]
        df_old.iloc[old_row+1:row+1]["target"].values[:] = target_list
        old_row = row
    
def other_algs(obj):
      pass