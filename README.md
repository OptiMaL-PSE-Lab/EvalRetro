# evalretro
A repository for evaluation single step retrosynthesis algorithms.

The datafiles related to all benchmarked algorithms can be found below:
https://www.dropbox.com/sh/vuiksmg6p2hr8ie/AAAR9pW5TALhmM9mtUNvwF4ja?dl=0

## Environment:
Set up a new environment by running the following command: 

``` 
conda env create -n evalretro -f environment.yml 
```
## File Structure: 
To test the predictions, the file must follow one of the two following structures:

1. **Line-Separated** file: _N_ retrosynthesis predictions per _target_ are separated by an empty line (example: [TiedTransformer](https://www.dropbox.com/home/data_retroalgs/tiedtransformer?preview=tiedtransformer_pred.csv))
2. **Index-Separated** file: _N_ retrosynthesis predictions per _target_ are separated by different indicies (example: [G<sup>2</sup>Retro](https://www.dropbox.com/home/data_retroalgs/g2retro?preview=g2retro_pred.csv))

The configuration for a specific algorithm is defined in 

## Testing Algorithm: 
Put the file containing into the ./data directory 

