# evalretro
A repository for evaluation single step retrosynthesis algorithms.

The datafiles related to all benchmarked algorithms can be found below:
https://www.dropbox.com/sh/vuiksmg6p2hr8ie/AAAR9pW5TALhmM9mtUNvwF4ja?dl=0

## Environment
Set up a new environment by running the following command: 

``` 
conda env create -n evalretro -f environment.yml 
```
## File Structure
To test the predictions, the file must follow one of the two following structures:

1. **Line-Separated** file: _N_ retrosynthesis predictions per _target_ are separated by an empty line (example: [TiedTransformer](https://www.dropbox.com/home/data_retroalgs/tiedtransformer?preview=tiedtransformer_pred.csv))
2. **Index-Separated** file: _N_ retrosynthesis predictions per _target_ are separated by different indicies (example: [G<sup>2</sup>Retro](https://www.dropbox.com/home/data_retroalgs/g2retro?preview=g2retro_pred.csv))

The data file should contain the following columns: ["index", "target", "reactants"]

The configuration for the benchmarked algorithm is shown in [the config directory](./config/raw_data.json). Specifying the configuration is important so that the data file is processed correctly. 
The structure is in json format and structured as follows: 
```
{
"key":{
    "file":"file_name.csv",       # In ./data/key directory
    "preprocess":bool,            # Keep this as false if data file has the correct structure
    "skip":bool,                  # false if LineSeparated, true if IndexSeparated
    "class":"LineSeparated",      # One of: ["LineSeparated", "IndexSeparated"]
    "delimiter":"comma",          # Delimiter of file. One of: ["comma", " "]
    "colnames": null,             # null unless data file has different header to ["idx", "target", "reactants"]
    "type": "tmplate",            # Retrosynthesis category, One of: ["tmplate", "semi", "tfree"]
    "name": "your_name"           # Name of the algorithm
},
```
To test your own algorithm, replace the example in [the example config directory](./config/new_config.json) with your own configuration data.

## Pre-processing Data
Put the file containing into the ./data directory.
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

## Reproducibility
To reproduce results in paper, follow the steps below: 
1. Download all data files from dropbox and place inside ./data directory
2. Run the code in for data pre-processing with --config_name raw_data.json
3. Run `source evaluate.sh`
4. Run `python plotting.py` to generate figures and tables

# Interpretability Study

