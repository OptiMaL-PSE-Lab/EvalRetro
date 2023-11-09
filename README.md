# evalretro
[![Python 3.6](https://img.shields.io/badge/python-3.6-blue.svg)](https://www.python.org/downloads/release/python-360/)

A repository for evaluating single-step retrosynthesis algorithms.

The datafiles related to all benchmarked algorithms can be found below:
https://www.dropbox.com/sh/vuiksmg6p2hr8ie/AAAR9pW5TALhmM9mtUNvwF4ja?dl=0

## Environment
Set up a new environment by running the following line in your terminal: 

``` 
conda create -n evalretro -f environment.yml 
```
## File Structure
To test the predictions, the file must follow one of the two following structures:

1. **Line-Separated** file: _N_ retrosynthesis predictions per _target_ are separated by an empty line (example: [TiedTransformer](https://www.dropbox.com/home/data_retroalgs/tiedtransformer?preview=tiedtransformer_pred.csv))
2. **Index-Separated** file: _N_ retrosynthesis predictions per _target_ are separated by different indices (example: [G<sup>2</sup>Retro](https://www.dropbox.com/home/data_retroalgs/g2retro?preview=g2retro_pred.csv))

The data file should contain the following columns: ["index", "target", "reactants"]

The configuration for the benchmarked algorithm is shown in [the config directory](./config/raw_data.json). Specifying the configuration is important so that the data file is processed correctly. 
The structure is in json format and structured as follows: 
```
{
"key":{
    "file":"file_name.csv",       # In ./data/"key" directory
    "preprocess":bool,            # Keep this as false if data file has the correct structure
    "class":"LineSeparated",      # One of: ["LineSeparated", "IndexSeparated"]
    "skip":bool,                  # false if LineSeparated, true if IndexSeparated
    "delimiter":"comma",          # Delimiter of file. One of: ["comma", " "]
    "colnames": null,             # null unless data file has different header to ["idx", "target", "reactants"]
    "type": "tmplate",            # Retrosynthesis category, One of: ["tmplate", "semi", "tfree"]
    "name": "your_name"           # Name of the algorithm
},
```
To test your own algorithm, replace the example in [the example config directory](./config/new_config.json) with your own configuration data.

## Pre-processing Data
Put the file containing your predictions into the ./data directory.
To ensure that the file has the correct structure and information in the config file, run the following line of code: 
```
conda activate evalretro
python data_import.py --config_name new_config.json
```
If no error is logged, the algorithm can be tested.

## Testing Algorithm
To test your own algorithms, run:
```
source evaluate.sh  
```
Note: Adjust --config_name to 'new_config.json' 

Within the script, the following hyperparameters can be adjusted: 
- k_back: Evaluation includes _k_ retrosynthesis predictions per target
- k_forward: Forward model includes _k_ target predictions per reactant set.
- fwd_model: Type of forward reaction prediction model. So far, only _gcn_ is included.
- config_name: Name of the config file to be used
- quick_eval: Boolean - prints the results (averages) for evaluation metrics directly to the terminal.

## Reproducibility
To reproduce results in paper, follow the steps below: 
1. Download all data files from dropbox and place inside ./data directory
2. Run the code in for data pre-processing with --config_name raw_data.json
3. Run `source evaluate.sh`
4. Run `python plotting.py` to generate figures and tables

# Interpretability Study
The code related to the interpretability study is found in [the interpretability folder](./interpret).

## Environment
The environment can be set-up running the following lines of code: 

```
conda create -n rxn_exp python=3.10
conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
conda install pyg -c pyg
conda install scikit-learn -c conda-forge
conda install tqdm matplotlib pandas
pip install rdkit
```

## Data Files
Install both folders from the following link and place them into the ./interpret directory:
https://www.dropbox.com/sh/h5jnlmc4caebe3u/AADwIUVvKPg52oeGQWAMjogUa?dl=0

## Reproducibility
Pre-trained models are provided in the dropbox. However, models can be retrained by running: 
```
conda activate rxn_exp
cd interpret
python train.py --model_type DMPNN
```
The model_type can be chosen from: DMPNN, EGAT and GCN.

To test the trained models (i.e. EGAT and DMPNN) and create the plots as in the paper, run `python inference.py`.

![Example of interpretability case study](/interpret/example_interpret.png)

