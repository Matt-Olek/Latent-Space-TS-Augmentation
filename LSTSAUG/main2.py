from pipeline import pipeline
from utils import save_logs
from config import config

if __name__ == '__main__':
    logs = pipeline()
    save_logs(logs, config)