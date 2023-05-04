""" 
Create a logisitic regression classifer for reaction fingerprints
Please run this script from this directory
"""

import numpy as np
import pandas as pd
import torch
import os
import pickle
from sklearn.linear_model import LogisticRegression
from rdkit import Chem
from transformers import BertModel

from tqdm import tqdm
from rxnfp.transformer_fingerprints import RXNBERTFingerprintGenerator
from rxnfp.tokenization import SmilesTokenizer

path = os.path.abspath(os.path.dirname(__file__))

def model_tokenizer(model, force_no_cuda=False, project_path=path):
    
    model_path =  os.path.join(project_path, model)

    tokenizer_vocab_path = os.path.join(project_path, model, 'vocab.txt')
    
    device = torch.device("cuda" if (torch.cuda.is_available() and not force_no_cuda) else "cpu")

    model = BertModel.from_pretrained(model_path)
    model = model.eval()
    model.to(device)

    tokenizer = SmilesTokenizer(
        tokenizer_vocab_path
    )
    return model, tokenizer

def generate_fingerprints(model_name):
    model, tokenizer = model_tokenizer(model_name)
    rxnfp_generator = RXNBERTFingerprintGenerator(model, tokenizer)
    return rxnfp_generator

def main():
    # Load in model and ft fingerprints
    schneider_df = pd.read_csv(os.path.join(path, "schneider50.csv"), index_col=0)
    rxn = schneider_df['rxn'].values.tolist()
    generator = generate_fingerprints(model_name='bert_50k_best')
    fps = []
    for i in tqdm(range(0, len(rxn), 200)):
        fp = generator.convert_batch(rxn[i:i+200])
        fps.extend(fp)

    schneider_df['ft_full'] = fps

    # Build prediction model

    lr_cls = LogisticRegression(max_iter=1000, multi_class='multinomial', solver='lbfgs')

    train_df = schneider_df[(schneider_df['split'] == 'train') | (schneider_df['split'] == 'val')]
    test_df = schneider_df[schneider_df['split'] == 'test']

    lr_cls.fit(train_df['ft_full'].values.tolist(), train_df['class'].values.tolist())
    predicted = lr_cls.predict(test_df['ft_full'].values.tolist())
    # Calculate accuracy
    accuracy = np.sum(predicted == test_df['class'].values.tolist()) / len(test_df)
    print(accuracy)
    print(predicted[0:5])
    # Save model to file
    pickle.dump(lr_cls, open(os.path.join(os.path.dirname(os.path.abspath(__file__)),'lr_cls.pkl'), 'wb'))

if __name__ == "__main__":
    main()
