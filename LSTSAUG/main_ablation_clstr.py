from pipeline_ablation_clstr import pipeline
from utils import save_logs, get_default_device
from config import config
import time
import pandas as pd


datasets_names_benchmark = pd.read_csv("baselines.csv", sep=";", skiprows=1).iloc[:, 0].tolist()


if __name__ == "__main__":
    classifier_Types = ["Resnet"]  # "FCN",
    datasets_names = datasets_names_benchmark
    selected_datasets = datasets_names
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
