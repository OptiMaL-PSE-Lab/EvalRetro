
import logging
import shutil
import os
import argparse 
import json
import numpy as np # type: ignore
# Disable tensorflow warning if no GPU found
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from src.alg_classes import LineSeparated, IndexSeparated
from src.evaluation_metrics import eval_scscore, round_trip, diversity, duplicates, valsmiles, top_k
from src.utils.fwd_mdls import gcn_forward, localt_forward
from src.utils.utilities import str2bool

parser = argparse.ArgumentParser(description='Evaluate retrosynthesis algorithms')
parser.add_argument('--k_back', type=int, help='Number of predictions made per target for retrosynthesis', default=10)
parser.add_argument('--k_forward', type=int, help='Number of predictions made per reactant set for forward synthesis', default=2)
parser.add_argument('--fwd_model', type=str, help='Name of forward model to use. Choose from ["gcn" and "lct"]',default='gcn')
parser.add_argument('--invsmiles', type=int, help='Number of predictions check for invalid smiles per target', default=20)
parser.add_argument('--dup', type=int, help='Number of predictions check for duplicates', default=20)
parser.add_argument('--stereo', type=str2bool, help='Whether to remove stereochemistry for fwd model: [True for gcn, False for lct]', default=True)
parser.add_argument('--check', type=str2bool, help='Remove invalid smiles from files', default=True)
parser.add_argument('--config_name', type=str, help='Name of config file to use', required=True)
parser.add_argument('--quick_eval', type=str2bool, help='Whether to evaluate results on the fly', default=True)
parser.add_argument('--data_path',  type=str, help='Location of data files', default='data')
parser.add_argument('--config_path',  type=str, help='Location of config files', default='config')

args = parser.parse_args()
# Calling env variable
cwd = os.getcwd()
with open(os.path.join(cwd, args.config_path, args.config_name), 'r') as f:
    config = json.load(f)

# Create dict for fwd models
fwd_models = {"gcn": gcn_forward, "lct": localt_forward}
args.fwd_model = fwd_models[args.fwd_model]

logging.basicConfig(filename='logs/main.log',level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.INFO)
f = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
stream_handler.setFormatter(f)
logging.getLogger().addHandler(stream_handler)

# Get terminal width
terminal_width = shutil.get_terminal_size().columns
num_chars = terminal_width - 1  # subtract 1 to account for newline character

eval_metrics = [diversity, duplicates, valsmiles, top_k, eval_scscore, round_trip]
# List foldernames under Data directory 
algorithms = config.keys()
alg_type = {"LineSeparated": LineSeparated, "IndexSeparated": IndexSeparated}

def constructor():
    
    for retro_alg in algorithms:
        summary_stats = []
        # Log algorithm name about to evaluate and directory for results
        print('-'*num_chars)
        logging.info(f"Evaluating {retro_alg}:\n Results will be saved in {os.path.join(os.getcwd(), 'results', retro_alg)}")
        print('-'*num_chars)
        try:
            info = config[retro_alg]
        except NameError:
            logging.error(f"Specify {retro_alg} in config file")
        alg = alg_type[info["class"]]

        alg_instance = alg(retro_alg, args.check, info["skip"], args.stereo, args.data_path)
        for metric in eval_metrics:
            # Log metric name about to evaluate
            logging.info(f"Evaluating {metric.__name__}")
            print_results = alg_instance.evaluate(metric, args)
            summary_stats.append(print_results)

        # Print results to terminal:
        if args.quick_eval:
            print('Metrics scaled between [0,1] with 1 being the best!')
            print('-'*num_chars)
            print(f"Results for {retro_alg}:")
            for i,metric in enumerate(eval_metrics):
                print(f"Top-{metric.k} {metric.__name__}: {np.round(summary_stats[i], 2)}\n")
            print('-'*num_chars)

if __name__ == "__main__":
    constructor()