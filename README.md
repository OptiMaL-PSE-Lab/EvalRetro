<img src="https://avatars.githubusercontent.com/u/81195336?s=200&v=4" alt="Optimal PSE logo" title="OptiMLPSE" align="right" height="150" />
</a>

# EvalRetro
[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A repository for evaluating single-step retrosynthesis algorithms.

This code was tested for Linux (Ubuntu), Windows and Mac OS.

## Environment
Set up a new environment by running the following line in your terminal: 

``` 
conda env create -n evalretro --file environment.yml 
pip install rxnfp --no-deps
```
For MacOS, replace the environment.yml file with:
``` 
conda env create -n evalretro --file environment_mac.yml 
```

## Testing your own algorithm
To test your own retrosynthetic prediction on the test dataset (e.g. USPTO-50k), follow the steps below: 
1. Place the file containing the predictions per molecular target in the ./data/"key" directory ("key" as defined in config file - step 2.) <br />
    > Please ensure the correct layout of your prediction file as shown in [File Structure](#File-Structure)
2. Enter the configuration details in the config .json under [the example config directory](./config/new_config.json) by replacing the example provided <br />
    > Please refer to [Config](#Configuration-File) for the configuration layout
3. To ensure that the file has the correct structure, run the following line of code: 
    ```
    conda activate evalretro
    python data_import.py --config_name new_config.json 
    ```
4. If no error is logged in step 3, the algorithm can be evaluated with: 
    ```
    python main.py --k_back 10 --k_forward 2 --invsmiles 20 --fwd_model 'gcn' --config_name 'new_config.json' --quick_eval True  
    ```
    Within the script, the following hyperparameters can be adjusted: 
    - k_back: Evaluation includes _k_ retrosynthesis predictions per target
    - k_forward: Forward model includes _k_ target predictions per reactant set.
    - fwd_model: Type of forward reaction prediction model. So far, only _gcn_ is included.
    - config_name: Name of the config file to be used
    - quick_eval: Boolean - prints the results (averages) for evaluation metrics directly to the terminal.
    - data_path: The path to the folder that contains your file, default = ./data
      
> [!TIP]   
> For futher help, look at the example Jupyter notebook provided in [the examples directory](./examples/evaluate_algorithm.ipynb)

### File Structure
The file should follow **one** of the following two formats with the **first row entry per target molecule being the ground truth reaction** i.e. N+1 predictions per target:

1. **Line-Separated** file: _N+1_ retrosynthesis predictions per _molecular target_ are separated by an empty line (example: [TiedTransformer](https://www.dropbox.com/home/data_retroalgs/tiedtransformer?preview=tiedtransformer_pred.csv))
2. **Index-Separated** file: _N+1_ retrosynthesis predictions per _molecular target_ are separated by different indices (example: [G<sup>2</sup>Retro](https://www.dropbox.com/home/data_retroalgs/g2retro?preview=g2retro_pred.csv))

The headers within the file should contain the following columns: ["index", "target", "reactants"]

### Configuration File
The configuration for the benchmarked algorithm is shown in [the config directory](./config/raw_data.json). Specifying the configuration is important so that the data file is processed correctly by the code. 
The structure is in .json format and should contain: 
```
"key":{
    "file":"file_name.csv",       # The name of the prediction file in ./data/"key" directory
    "class":"LineSeparated",      # One of: ["LineSeparated", "IndexSeparated"]
    "skip":bool,                  # false if LineSeparated; true if IndexSeparated
    "delimiter":"comma",          # Delimiter of file. One of: ["comma", " "]
    "colnames": null,             # null - unless data file has different header to ["idx", "target", "reactants"]
    "preprocess":bool,            # false - in most cases
}
```

## Reproducibility
To reproduce results in paper, follow the steps below: 
1. Download all data files from dropbox and place inside ./data directory <br />
    > The datafiles related to all benchmarked algorithms can be found below:
    > https://www.dropbox.com/sh/vuiksmg6p2hr8ie/AAAR9pW5TALhmM9mtUNvwF4ja?dl=0 
2. Run the following lines of code within your terminal:
   ```
   conda activate evalretro
   python data_import.py --config_name raw_data.json
   python main.py --k_back 10 --k_forward 2 --invsmiles 20 --fwd_model 'gcn' --config_name 'raw_data.json' --quick_eval False
   ```
3. Run `python plotting.py` to generate figures and tables

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

To test the trained models (i.e. EGAT and DMPNN) and create the plots as in the paper, run:  
```
conda activate rxn_exp
python inference.py
```
**Note**: The plots for the GNN models may slightly differ compared to the paper due to the stochastic nature of GNNExplainer.
![Example of interpretability case study](/examples/example_interpret.png)
