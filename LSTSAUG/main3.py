from pipeline import pipeline
from utils import save_logs
from config import config
import pandas as pd
import numpy as np


if __name__ == "__main__":
    selected_datasets = pd.read_csv("data/selected_datasets.csv")
    classifiers = ["FCN", "Resnet"]
    config["RESULTS_DIR"] = "results/alpha_study"
    for dataset_name in selected_datasets["dataset"]:
        config["DATASET"] = dataset_name
        for classifier in classifiers:
            config["CLASSIFIER"] = classifier
            for i in np.arange(1.0, 2.0, 0.2):
                if i > 1.0:
                    config["ALPHA"] = i
                    try:
                        logs = pipeline(config)
                        logs["alpha"] = i
                        save_logs(logs, config)
                        print(f"{dataset_name} done!")
                    except Exception as e:
                        print(f"{dataset_name} failed!")
                        print(e)
                        continue
