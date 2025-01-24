import json
import os
import csv
import numpy as np
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
    
    # If the key 'class_trusts' exists in the logs, we save it as a separate csv file
    if "class_trusts" in logs:
        # Check if tge file exists
        if not os.path.exists(os.path.join(config["RESULTS_DIR"], 'class_trusts.json')):
            with open(os.path.join(config["RESULTS_DIR"], 'class_trusts.json'), 'w') as f:
                json.dump({}, f)
        # Read as a json file
        current_results = {}
        with open(os.path.join(config["RESULTS_DIR"], 'class_trusts.json'), 'r') as f:
            # Extract the current json file
            current_results = json.load(f)
        with open(os.path.join(config["RESULTS_DIR"], 'class_trusts.json'), 'w') as f:
            # Format the class trusts as a dictionary
            class_trusts = {f"class_{i}": {f"step_{j}": float(trust) for j, trust in enumerate(trusts)} for i, trusts in enumerate(np.array(logs["class_trusts"]).T)}
            
            # Add the class trusts under the dataset key to the json file
            current_results[config["DATASET"]] = class_trusts
            
            # Save the updated json file
            json.dump(current_results, f)
        
    
    with open(os.path.join(config["RESULTS_DIR"], 'logs.csv'), 'a', newline='') as f:
        writer = csv.writer(f)
        if f.tell() == 0:
            writer.writerow(logs.keys())
        writer.writerow(logs.values())
        
    # Save the config in a json file as well for reproducibility
    
    with open(os.path.join(config["RESULTS_DIR"], 'config.json'), 'w') as f:
        json.dump(config, f)
        
def get_model_path(config, name='vae'):
    return os.path.join(config["MODEL_DIR"], f'{config["DATASET"]}_{name}.pth')