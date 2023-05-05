
import logging
import shutil
import os
import argparse 
import json

from src.alg_classes import LineSeparated, IndexSeparated
from src.evaluation_metrics import eval_scscore, round_trip, diversity, duplicates, invsmiles, top_k
from src.utils.fwd_mdls import gcn_forward 

parser = argparse.ArgumentParser(description='Evaluate retrosynthesis algorithms')
parser.add_argument('--k_back', type=int, help='Number of predictions made per target for retrosynthesis', default=10)
parser.add_argument('--k_forward', type=int, help='Number of predictions made per reactant set for forward synthesis', default=2)
parser.add_argument('--scmodel', type=str, help='Name of scscore model to use', default="2048bool")
parser.add_argument('--fwd_model', type=str, help='Name of forward model to use', default='gcn')
parser.add_argument('--invsmiles', type=int, help='Number of predictions check for invalid smiles per target', default=20)
parser.add_argument('--dup', type=int, help='Number of predictions check for duplicates', default=20)
parser.add_argument('--stereo', type=bool, help='Whether to remove stereochemistry for fwd model', default=True)
parser.add_argument('--check', type=bool, help='Remove invalid smiles from files', default=True)

args = parser.parse_args()
# Calling env variable
data_path = os.environ['DATAPATH']
config_path = os.environ['CONFIGPATH']
with open(os.path.join(config_path,"raw_data.json"), 'r') as f:
    config = json.load(f)

# Create dict for fwd models
fwd_models = {"gcn": gcn_forward}
args.fwd_model = fwd_models[args.fwd_model]

# Disable tensorflow warning if no GPU found
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

logging.basicConfig(filename='logs/main.log',level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.INFO)
f = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
stream_handler.setFormatter(f)
logging.getLogger().addHandler(stream_handler)

# Get terminal width
terminal_width = shutil.get_terminal_size().columns
num_chars = terminal_width - 1  # subtract 1 to account for newline character

eval_metrics = [round_trip, eval_scscore, diversity, duplicates, invsmiles, top_k]
# List foldernames under Data directory 
algorithms = [f for f in os.listdir(data_path)]
alg_type = {"LineSeparated": LineSeparated, "IndexSeparated": IndexSeparated}

def constructor():
    
    for retro_alg in algorithms:
        # Log algorithm name about to evaluate and directory for results
        print('-'*num_chars)
        logging.info(f"Evaluating {retro_alg}:\n Results will be saved in {os.path.join(os.getcwd(), 'results', retro_alg)}")
        print('-'*num_chars)
        try:
            info = config[retro_alg]
        except NameError:
            logging.error(f"Specify {retro_alg} in config file")
        alg = alg_type[info["class"]]

        alg_instance = alg(retro_alg, args.check, info["skip"], args.stereo)
        for metric in eval_metrics:
            # Log metric name about to evaluate
            logging.info(f"Evaluating {metric.__name__}")
            alg_instance.evaluate(metric, args)

if __name__ == "__main__":
    constructor()