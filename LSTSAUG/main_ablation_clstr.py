from pipeline_ablation_clstr import pipeline
from utils import save_logs, get_default_device
from config import config
import time
import pandas as pd

if __name__ == "__main__":
    classifier_Types = ["Resnet"]  # "FCN",
    datasets_names = open("data/datasets_names.txt", "r").read().split("\n")
    selected_datasets = pd.read_csv("data/selected_datasets.csv")
    selected_datasets = selected_datasets["dataset"].values
    print("Selected datasets:", selected_datasets)
    current_time = time.strftime("%Y-%m-%d_%H-%M-%S")
    for classifier_Type in classifier_Types:
        config["CLASSIFIER"] = classifier_Type
        config["RESULTS_DIR"] = config["RESULTS_DIR"] + current_time

        current_time = time.strftime("%Y-%m-%d_%H-%M-%S")
        # Print the pytorch device
        print(f"Device: {get_default_device()}")
        for dataset_name in datasets_names:
            # if dataset_name in selected_datasets:
            #     visualizations = True
            # else:
            #     visualizations = False
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
