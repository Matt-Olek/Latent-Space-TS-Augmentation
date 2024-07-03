import os
import csv
import torch

# ------------------------------ Utils ------------------------------ #

def get_default_device():
    """Pick GPU if available, else CPU"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')
    
def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list,tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)

def to_default_device(data):
    """Move tensor(s) to default device"""
    return to_device(data, get_default_device())

def custom_collate(batch, device):
    if isinstance(batch[0], tuple):
        return [(item[0].to(device, non_blocking=True), item[1].to(device, non_blocking=True)) for item in batch]
    else:
        return [item.to(device, non_blocking=True) for item in batch]
    
def save_logs(logs, config):
    with open(os.path.join(config["RESULTS_DIR"], 'logs.csv'), 'a', newline='') as f:
        writer = csv.writer(f)
        if f.tell() == 0:
            writer.writerow(logs.keys())
        writer.writerow(logs.values())
        
def get_model_path(config, name='vae'):
    return os.path.join(config["MODEL_DIR"], f'{config["DATASET"]}_{name}.pth')