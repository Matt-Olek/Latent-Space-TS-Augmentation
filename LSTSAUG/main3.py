from pipeline import pipeline
from utils import save_logs
from config import config

if __name__ == "__main__":
    config["RESULTS_DIR"] = "results/alpha_study_bis"
    dataset_name = "MedicalImages"
    for i in range(1, 10):
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
