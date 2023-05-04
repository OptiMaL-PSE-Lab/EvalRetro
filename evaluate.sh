#!bin/bash

conda activate benm
export DATAPATH=./data
export CONFIGPATH=./config

python main.py --k_back 10 --k_forward 2 --invsmiles 20 --fwd_model 'gcn'
