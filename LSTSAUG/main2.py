from pipeline import pipeline
from utils import save_logs
from config import config

if __name__ == '__main__':
    datasets_names = open('datasets_names.txt', 'r').read().split('\n')
    for dataset_name in datasets_names:
        for i in range(1):
            config["SEED"] = i
            config["DATASET"] = dataset_name
            try :
                logs = pipeline(config)
                save_logs(logs, config)
                print(f'{dataset_name} done!')
            except Exception as e:
                print(f'{dataset_name} failed!')
                print(e)
                continue
    
    