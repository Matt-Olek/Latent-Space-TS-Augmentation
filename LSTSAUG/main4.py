from pipeline import pipeline
from utils import save_logs, get_default_device
from config import config
import time
import os
import itertools
from tqdm import tqdm

import warnings
warnings.filterwarnings("ignore", message=".*cudnnException.*CUDNN_STATUS_NOT_SUPPORTED.*", category=UserWarning, module="torch")

log_error = True

RECON_WEIGHT_GRID = [1, 3, 5, 10]
KL_WEIGHT_GRID = [1, 3, 5, 10]
CLASSIFIER_WEIGHT_GRID = [1, 3, 5, 10]
CONTRASTIVE_WEIGHT_GRID = [1, 3, 5, 10]

grid = list(itertools.product(RECON_WEIGHT_GRID, KL_WEIGHT_GRID, CLASSIFIER_WEIGHT_GRID, CONTRASTIVE_WEIGHT_GRID))

def main():
    classifier_Types = ["FCN", "Resnet"]
    for classifier_Type in classifier_Types:
        config["CLASSIFIER"] = classifier_Type
        # datasets_names = open("data/datasets_names.txt", "r").read().split("\n")
        # Get the datasets names by listing the subdirectories in the "UCRArchive_2018" directory
        datasets_names = os.listdir("data/UCRArchive_2018")
        # datasets_names = [datasets_names]
        current_time = time.strftime("%Y-%m-%d_%H-%M-%S")
        config["RESULTS_DIR"] = config["RESULTS_DIR_ROOT"] + current_time
        config["WITH_AUG"] = True
        # Print the pytorch device
        print(f"Device: {get_default_device()}")
        for dataset_name in datasets_names:
            for i in range(1):
                config["SEED"] = i
                config["DATASET"] = dataset_name
                if log_error:
                    try:
                        logs = pipeline(config)
                        save_logs(logs, config)
                        print(f"{dataset_name} done!")
                    except Exception as e:
                        print(f"{dataset_name} failed!")
                        save_logs({"error": str(e)}, config)
                        print(e)
                        continue
                else:
                    logs = pipeline(config)
                    save_logs(logs, config)
                    print(f"{dataset_name} done!")

if __name__ == "__main__":
    
    print(f'Grid search with {len(grid)} configurations')
    
    for i in tqdm(range(len(grid))):
        
        # Normalizing the weights so that they sum to 1
        total = sum(grid[i])
        grid[i] = [x / total for x in grid[i]]
        
        config["RECON_WEIGHT"] = grid[i][0]
        config["KL_WEIGHT"] = grid[i][1]
        config["CLASSIFIER_WEIGHT"] = grid[i][2]
        config["CONTRASTIVE_WEIGHT"] = grid[i][3]
        main()