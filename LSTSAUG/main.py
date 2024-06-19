#---------------------------------- Imports ----------------------------------#

import os
import wandb
import torch    
import tqdm
import numpy as np
from VAE import VAE
from visualization import plot_latent_space_neighbors
from ClassifierModel import Classifier_RESNET
from loader import getUCRLoader, augment_loader
from utils import to_default_device
from config import config
import csv

#---------------------------------- Main ----------------------------------#

def main(config=config):
    torch.manual_seed(config["SEED"])
    
    os.makedirs(config["MODEL_DIR"], exist_ok=True)
    os.makedirs(config["RESULTS_DIR"], exist_ok=True)
    train_loader, test_dataset, nb_classes, scaler = getUCRLoader(config["DATA_DIR"], config["DATASET"], config["BATCH_SIZE"])
    
    input_dim = train_loader.dataset[0][0].shape[0]
    print("Detected input dimension: ", input_dim)
    vae = to_default_device(VAE(input_dim, nb_classes, latent_dim=input_dim, hidden_dim=10000, hidden_dim_classifier=1000))
    
    num_samples = int(len(train_loader.dataset)/nb_classes)*config["NUM_SAMPLES"]
    print("Number of samples produced per class: ", num_samples)
    logs = {}
    logs['dataset'] = config["DATASET"]
    logs['num_classes'] = nb_classes
    logs['num_train_samples'] = len(train_loader.dataset)
    logs['num_test_samples'] = len(test_dataset[0])
    
    # ---------------------------- VAE Training ---------------------------- #
    model_path = os.path.join(config["MODEL_DIR"], f'{config["DATASET"]}_vae-{config["BATCH_SIZE"]}bs-{config["LATENT_DIM"]}ld-{config["VAE_NUM_EPOCHS"]}e.pth')

    if not config["USE_TRAINED"] or not os.path.exists(model_path):
        if config["WANDB"]:
            wandb.init(project='lstsaug_no_val',
                        config=config,
                        tags=['vae'],
                        name=f'{config["DATASET"]}_vae-{config["BATCH_SIZE"]}bs-{config["LATENT_DIM"]}ld-{config["VAE_NUM_EPOCHS"]}e')
            wandb.watch(vae)
        
        # Train 
        print('#'*50)
        print('Training the VAE model...')
        for epoch in tqdm.tqdm(range(config["VAE_NUM_EPOCHS"])):
            train_loss, train_recon_loss, train_kl_div, train_class_loss = vae.train_epoch(train_loader)
            
            # Test the model
            acc, f1 = vae.validate(test_dataset)
            if config["WANDB"]:
                wandb.log({ 'train_loss': train_loss,
                            'train_recon_loss': train_recon_loss,
                            'train_kl_div': train_kl_div,
                            'train_class_loss': train_class_loss,
                            'lr': float(vae.scheduler.get_last_lr()[0]),
                            'test_accuracy': acc,
                            'test_f1': f1})
                
        # Save the trained model
        model_path = os.path.join(config["MODEL_DIR"], f'{config["DATASET"]}_vae-{config["BATCH_SIZE"]}bs-{config["LATENT_DIM"]}ld-{config["VAE_NUM_EPOCHS"]}e.pth')
        torch.save(vae.state_dict(), model_path)
        print(f'Model saved at {model_path}')  
        
        if config["WANDB"]:
            wandb.finish()
    
    else:
        model_path = os.path.join(config["MODEL_DIR"], f'{config["DATASET"]}_vae-{config["BATCH_SIZE"]}bs-{config["LATENT_DIM"]}ld-{config["VAE_NUM_EPOCHS"]}e.pth')
        vae.load_state_dict(torch.load(model_path))
        vae.eval()
        
    # ---------------------------- VAE Testing ---------------------------- #
    
    if config["AUGMENT_PLOT"]:
        print('#'*50)
        print('Plotting the augmented data points...')
        # Load the trained model
        model_path = os.path.join(config["MODEL_DIR"], f'{config["DATASET"]}_vae-{config["BATCH_SIZE"]}bs-{config["LATENT_DIM"]}ld-{config["VAE_NUM_EPOCHS"]}e.pth')
        vae.load_state_dict(torch.load(model_path))
        vae.eval()
        
        plot_latent_space_neighbors(vae, test_dataset, num_neighbors=5, distance=config["NOISE"], num_classes=nb_classes)
        
    # ---------------------------- Classifier Training ---------------------------- #
    
    if config["TEST_AUGMENT"]:
        # Baseline classifier training
        if config["BASELINE"]:
            classifier = to_default_device(Classifier_RESNET(input_dim, nb_classes,lr=config["LEARNING_RATE"], weight_decay=config["WEIGHT_DECAY"]))
            if config["WANDB"]:
                wandb.init(project='lstsaug_no_val',
                            config=config,
                            tags=['baseline'],
                            name=f'{config["DATASET"]}_classifier_validate-{config["BATCH_SIZE"]}bs-{config["NUM_EPOCHS"]}e')
                wandb.watch(classifier)
            print('#'*50)
            print('Training the classifier model and testing on the validation dataset...')
            best_acc = 0
            best_f1 = 0
            for epoch in tqdm.tqdm(range(config["NUM_EPOCHS"])):
                train_loss, train_acc = classifier.train_epoch(train_loader)
                # val_acc, val_f1 = classifier.validate(validation_dataset)
                test_acc, test_f1 = classifier.validate(test_dataset)
                if test_acc > best_acc:
                    best_acc = test_acc
                    best_f1 = test_f1
                if config["WANDB"]:
                    wandb.log({ 'train_loss': train_loss,
                                'train_accuracy': train_acc,
                                # 'val_accuracy': val_acc,
                                # 'val_f1': val_f1,
                                'test_accuracy': test_acc,
                                'test_f1': test_f1})
            print('Best test accuracy:', best_acc)
            print('Best test f1:', best_f1)

            logs['baseline_best_acc'] = best_acc
            logs['baseline_best_f1'] = best_f1
            logs['baseline_final_acc'] = test_acc
            logs['baseline_final_f1'] = test_f1
            
            if config["WANDB"]:
                wandb.finish()
        
        # Augment the dataset
        print('#'*50)
        print('Augmenting the dataset...')
        augmented_loader_baseline = augment_loader(train_loader, vae, num_samples=num_samples, distance=config["NOISE"], scaler=scaler, num_classes=nb_classes, alpha=config["ALPHA"])
        print('Done!')
        # augmented_loader = tw_loader(train_loader, config["NUM_SAMPLES"])
        
        # Train the classifier on the augmented dataset
        classifier_augmented = to_default_device(Classifier_RESNET(input_dim, nb_classes,lr=config["LEARNING_RATE"], weight_decay=config["WEIGHT_DECAY"]))
        if config["WANDB"]:
            wandb.init(project='lstsaug_no_val',
                        config=config,
                        tags=['augmented'],
                        name=f'{config["DATASET"]}_classifier_augmented-{config["BATCH_SIZE"]}bs-{config["NUM_EPOCHS"]}e-{config["NUM_SAMPLES"]}ns-{config["NOISE"]}n')
            wandb.watch(classifier_augmented)
        
        print('#'*50)
        
        print('Training the classifier model on the augmented dataset...')
        print('Augmented dataset size:', len(augmented_loader_baseline.dataset))
        logs['augmented_train_samples'] = len(augmented_loader_baseline.dataset)
        best_acc = 0
        best_f1 = 0
        for epoch in tqdm.tqdm(range(config["NUM_EPOCHS"])):
            train_loss, train_acc = classifier_augmented.train_epoch(augmented_loader_baseline)
            test_acc, test_f1 = classifier_augmented.validate(test_dataset)
            # val_acc, val_f1 = classifier_augmented.validate(validation_dataset)
            if test_acc > best_acc:
                best_acc = test_acc
                best_f1 = test_f1
            if config["WANDB"]:
                wandb.log({ 'train_loss': train_loss,
                            'train_accuracy': train_acc,
                            'test_accuracy': test_acc,
                            'test_f1': test_f1,
                            # 'val_accuracy': val_acc,
                            # 'val_f1': val_f1
                            })
        print('Best test accuracy:', best_acc)
        print('Best test f1:', best_f1)
        
        logs['augmented_best_acc'] = best_acc
        logs['augmented_best_f1'] = best_f1
        logs['augmented_final_acc'] = test_acc
        logs['augmented_final_f1'] = test_f1
        
        if config["WANDB"]:
            wandb.finish()     
        
    print('#'*50)
    
    # Save the logs
    with open(os.path.join(config["RESULTS_DIR"], 'logs.csv'), 'a', newline='') as f:
        writer = csv.writer(f)
        if f.tell() == 0:  # Check if file is empty
            writer.writerow(logs.keys())  # Write column names
        writer.writerow(logs.values())
            
if __name__ == '__main__':
    datasets_names = open('datasets_names.txt', 'r').read().split('\n')
    for dataset_name in datasets_names:
        for i in range(3):
            if i == 0:
                config["WANDB"] = True
            else:
                config["WANDB"] = False
            config["SEED"] = i
            config["DATASET"] = dataset_name
            main(config)
            print(f'{dataset_name} done!')