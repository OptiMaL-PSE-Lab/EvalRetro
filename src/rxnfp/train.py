""" 
Building the BERT classifier for the reaction fingerprinting
Please run this script from this directory if necessary and create a separate env solely with rxnfp
"""

import pandas as pd
import torch
import logging
import pkg_resources
import sklearn

from rxnfp.models import SmilesClassificationModel
logger = logging.getLogger(__name__)

df = pd.read_csv('schneider50.csv', index_col=0)
labels = df['class'].values.tolist()
labels = [int(label)-1 for label in labels]
df['class'] = labels
df_train = df[(df['split'] == 'train') | (df['split'] == 'val')]
df_test = df[df['split'] == 'test']

train_df = df_train[['rxn', 'class']]
train_df.columns = ['text', 'labels']

eval_df = df_train[['rxn', 'class']]
eval_df.columns = ['text', 'labels']
model_args = {
    'wandb_project': 'uspto50k', 'num_train_epochs': 10, 'overwrite_output_dir': True,
    'learning_rate': 2e-5, 'gradient_accumulation_steps': 1,
    'regression': False, "num_labels":  10, "fp16": False,
    "evaluate_during_training": True, 'manual_seed': 42,
    "max_seq_length": 512, "train_batch_size": 16,"warmup_ratio": 0.00,
    'output_dir': 'bert_50k_best', 
    'thread_count': 8,
    }

model_path =  pkg_resources.resource_filename("rxnfp", "models/transformers/bert_pretrained")
model = SmilesClassificationModel("bert", model_path, args=model_args, num_labels=10, use_cuda=torch.cuda.is_available())
model.train_model(train_df, eval_df=eval_df, acc=sklearn.metrics.accuracy_score, mcc=sklearn.metrics.matthews_corrcoef)