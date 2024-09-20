from pipeline import pipeline
from utils import save_logs
from config import config
import time

if __name__ == "__main__":
    datasets_names = open("data/datasets_names.txt", "r").read().split("\n")
    current_time = time.strftime("%Y-%m-%d_%H-%M-%S")
    config["RESULTS_DIR"] = config["RESULTS_DIR"] + current_time
    for dataset_name in datasets_names:
        for i in range(1):
            config["SEED"] = i
            config["DATASET"] = dataset_name
            try:
                logs = pipeline(config)
                save_logs(logs, config)
                print(f"{dataset_name} done!")
            except Exception as e:
                print(f"{dataset_name} failed!")
                save_logs({"error": str(e)}, config)
                print(e)
                continue
