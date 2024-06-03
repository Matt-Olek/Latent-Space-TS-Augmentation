#---------------------------------- Imports ----------------------------------#

import os
import wandb
import argparse
import torch    
import tqdm
import numpy as np
from VAE import VAE
from visualization import plot_latent_space_neighbors
from ClassifierModel import Classifier_RESNET
from loader import getUCRLoader, augment_loader, tw_loader
from utils import to_default_device

#---------------------------------- Parser ----------------------------------#  

def get_parser():
    parser = argparse.ArgumentParser(description='LSTSAUG')
    parser.add_argument('--data_dir', type=str, default='../../FastAutoAugment-Time-Series/data',
                        help='Directory containing the dataset')
    parser.add_argument('--dataset', type=str, default='Worms',
                        help='Directory containing the dataset')
    parser.add_argument('--model_dir', type=str, default='models',
                        help='Directory to save models')
    parser.add_argument('--results_dir', type=str, default='results',
                        help='Directory to save results')
    parser.add_argument('--num_epochs', type=int, default=100,
                        help='Number of epochs to train the model')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size for training')
    parser.add_argument('--latent_dim', type=int, default=100,
                        help='Dimension of the latent space')
    parser.add_argument('--learning_rate', type=float, default=0.01,
                        help='Learning rate for training')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--wandb', action='store_true', default=False,
                        help='Use Weights and Biases for experiment tracking')
    parser.add_argument('--augment_plot', action='store_true', default=False,
                        help='Augment the dataset using the trained VAE')
    parser.add_argument('--test_augment', action='store_true', default=False,
                        help='Test the augmentation using the trained VAE')
    parser.add_argument('--use_trained', action='store_true', default=False,
                        help='Use a trained model for augmentation')    
    parser.add_argument('--baseline', action='store_true', default=False,
                        help='Train a baseline classifier')
    parser.add_argument('--num_samples', type=int, default=125,
                        help='Number of synthetic data samples to generate for each input sample')
    parser.add_argument('--noise', type=float, default=0.01,
                        help='Noise level for the latent space perturbation')
    parser.add_argument('--incremental_aug', action='store_true', default=False,
                        help='Incremental augmentation')
    return parser
    
#---------------------------------- Main ----------------------------------#

def main() :
    # Parse command line arguments
    parser = get_parser()
    args = parser.parse_args()
    
    # Set random seed for reproducibility
    # torch.manual_seed(args.seed)
    
    # Create directories for saving models and results
    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)
    
    # Create a data loader for training
    train_loader, test_dataset, nb_classes, scaler = getUCRLoader(args.data_dir, args.dataset, args.batch_size)
    
    # # Split the test dataset into test_dataset and validation_dataset
    # test_size = len(test_dataset[0])
    # val_size = int(0.2 * test_size)
    # indices = np.random.permutation(test_size)
    # val_indices = indices[:val_size]
    # test_indices = indices[val_size:]
    
    # X_val, y_val = test_dataset[0][val_indices], test_dataset[1][val_indices]
    # validation_dataset = [X_val, y_val]
    # test_dataset = [test_dataset[0][test_indices], test_dataset[1][test_indices]]
    
    # Initialize the VAE model
    input_dim = train_loader.dataset[0][0].shape[0]
    print("Detected input dimension: ", input_dim)
    vae = to_default_device(VAE(input_dim, nb_classes, latent_dim=args.latent_dim))
    
    # ---------------------------- VAE Training ---------------------------- #
    
    if not args.use_trained:
        if args.wandb:
            wandb.init(project='lstsaug_no_val',
                        config=vars(args),
                        tags=['vae'],
                        name=f'{args.dataset}_vae-{args.batch_size}bs-{args.latent_dim}ld-{args.num_epochs}e')
            wandb.watch(vae)
        
        # Train 
        print('Training the VAE model...')
        for epoch in tqdm.tqdm(range(args.num_epochs)):
            train_loss, train_recon_loss, train_kl_div, train_class_loss = vae.train_epoch(train_loader)
            
            # Test the model
            acc, f1 = vae.validate(test_dataset)
            if args.wandb:
                wandb.log({ 'train_loss': train_loss,
                            'train_recon_loss': train_recon_loss,
                            'train_kl_div': train_kl_div,
                            'train_class_loss': train_class_loss,
                            'lr': float(vae.scheduler.get_last_lr()[0]),
                            'test_accuracy': acc,
                            'test_f1': f1})
                
        # Save the trained model
        model_path = os.path.join(args.model_dir, f'{args.dataset}_vae-{args.batch_size}bs-{args.latent_dim}ld-{args.num_epochs}e.pth')
        torch.save(vae.state_dict(), model_path)
        print(f'Model saved at {model_path}')  
        
        if args.wandb:
            wandb.finish()
    
    else :
        model_path = os.path.join(args.model_dir, f'{args.dataset}_vae-{args.batch_size}bs-{args.latent_dim}ld-{args.num_epochs}e.pth')
        vae.load_state_dict(torch.load(model_path))
        vae.eval()
    # ---------------------------- VAE Testing ---------------------------- #
    
    if args.augment_plot:
        #load the trained model
        model_path = os.path.join(args.model_dir, f'{args.dataset}_vae-{args.batch_size}bs-{args.latent_dim}ld-{args.num_epochs}e.pth')
        vae.load_state_dict(torch.load(model_path))
        vae.eval()
        
        plot_latent_space_neighbors(vae, test_dataset, num_neighbors=5, distance=args.noise, num_classes=nb_classes)
    
    # ---------------------------- Classifier Training ---------------------------- #
    
    if args.test_augment :
        # Baseline classifier training
        if args.baseline:
            classifier = to_default_device(Classifier_RESNET(input_dim, nb_classes))
            if args.wandb:
                wandb.init(project='lstsaug_no_val',
                            config=vars(args),
                            tags = ['baseline'],
                            name=f'{args.dataset}_classifier_validate-{args.batch_size}bs-{args.num_epochs}e')
                wandb.watch(classifier)
            
            print('Training the classifier model and testing on the validation dataset...')
            best_acc = 0
            best_f1 = 0
            for epoch in tqdm.tqdm(range(args.num_epochs)):
                train_loss, train_acc = classifier.train_epoch(train_loader)
                # val_acc, val_f1 = classifier.validate(validation_dataset)
                test_acc, test_f1 = classifier.validate(test_dataset)
                if args.wandb:
                    wandb.log({ 'train_loss': train_loss,
                                'train_accuracy': train_acc,
                                # 'val_accuracy': val_acc,
                                # 'val_f1': val_f1,
                                'test_accuracy': test_acc,
                                'test_f1': test_f1})
                    

            if args.wandb:
                wandb.finish()
            
        # Augment the dataset
        augmented_loader_baseline = augment_loader(train_loader, vae, args.num_samples, distance=args.noise, scaler=scaler, num_classes=nb_classes)
        # augmented_loader = tw_loader(train_loader, args.num_samples)
        
        # Train the classifier on the augmented dataset
        classifier_augmented = to_default_device(Classifier_RESNET(input_dim, nb_classes))
        if args.wandb:
            wandb.init(project='lstsaug_no_val',
                        config=vars(args),
                        tags = ['augmented'],
                        name=f'{args.dataset}_classifier_augmented-{args.batch_size}bs-{args.num_epochs}e-{args.num_samples}ns-{args.noise}n')
            wandb.watch(classifier_augmented)
            
        print('Training the classifier model on the augmented dataset...')
        print('Augmented dataset size:', len(augmented_loader_baseline.dataset))
        best_acc = 0
        best_f1 = 0
        for epoch in tqdm.tqdm(range(args.num_epochs)):
            train_loss, train_acc = classifier_augmented.train_epoch(augmented_loader_baseline)
            test_acc, test_f1 = classifier_augmented.validate(test_dataset)
            # val_acc, val_f1 = classifier_augmented.validate(validation_dataset)
            if args.wandb:
                wandb.log({ 'train_loss': train_loss,
                            'train_accuracy': train_acc,
                            'test_accuracy': test_acc,
                            'test_f1': test_f1,
                            # 'val_accuracy': val_acc,
                            # 'val_f1': val_f1
                            })
                
        if args.wandb:
            wandb.finish()     
        
    # ---------------------------- Incremental Augmentation ---------------------------- #
    
    if args.incremental_aug:
        print('Incremental Augmentation')
        rounds = 100
        num_aug_samples = 1
        
        # Quickly train a baseline to get the initial best accuracy and f1
        classifier = to_default_device(Classifier_RESNET(input_dim, nb_classes))
        
        if args.wandb:
            wandb.init(project='lstsaug_no_val',
                        config=vars(args),
                        tags = ['baseline'],
                        name=f'{args.dataset}_classifier_baseline-{args.batch_size}bs-{args.num_epochs}e')
            wandb.watch(classifier)
        best_acc_baseline = 0
        best_f1_baseline = 0
        for epoch in tqdm.tqdm(range(args.num_epochs)):
            train_loss, train_acc = classifier.train_epoch(train_loader)
            # val_acc, val_f1 = classifier.validate(validation_dataset)
            if val_acc > best_acc_baseline:
                best_acc_baseline = val_acc
            if val_f1 > best_f1_baseline:
                best_f1_baseline = val_f1
            if args.wandb:
                wandb.log({ 'train_loss': train_loss,
                            'train_accuracy': train_acc,
                            'val_accuracy': val_acc,
                            'val_f1': val_f1,
                            'best_accuracy': best_acc_baseline,
                            'best_f1': best_f1_baseline})
                
        if args.wandb:
            wandb.finish()
                
        print(f'Baseline best accuracy: {best_acc_baseline}, best f1: {best_f1_baseline}')
        
        for _ in range(rounds):
            print('Round', _+1)
            n_choices =5 
            idx = np.random.choice(len(train_loader.dataset), n_choices, replace=False)
            
            # Augment the dataset using the trained VAE on the selected sample
            augmented_loader = augment_loader(train_loader, vae, num_aug_samples, distance=args.noise, sample_idx=idx, scaler=scaler)
            # Train the classifier on the augmented dataset and track for best accuracy and f1
            
            classifier_augmented = to_default_device(Classifier_RESNET(input_dim, nb_classes))
            if args.wandb:
                wandb.init(project='lstsaug_bis',
                            config=vars(args),
                            tags = ['incremental'],
                            name=f'{args.dataset}_classifier_incremental-{args.batch_size}bs-{args.num_epochs}e-{args.num_samples}ns-{args.noise}n')
                wandb.watch(classifier_augmented)
                
            print('Training the classifier model on the augmented dataset...')
            print('Augmented dataset size:', len(augmented_loader.dataset))
            best_acc = 0
            best_f1 = 0
            for epoch in tqdm.tqdm(range(args.num_epochs)):
                train_loss, train_acc = classifier_augmented.train_epoch(augmented_loader)
                # val_acc, val_f1 = classifier_augmented.validate(validation_dataset)
                test_acc, test_f1 = classifier_augmented.validate(test_dataset)
                if val_acc > best_acc:
                    best_acc = val_acc
                if val_f1 > best_f1:
                    best_f1 = val_f1
                if args.wandb:
                    wandb.log({ 'train_loss': train_loss,
                                'train_accuracy': train_acc,
                                'test_accuracy': test_acc,
                                'test_f1': test_f1,
                                'val_accuracy': val_acc,
                                'val_f1': val_f1,
                                'best_accuracy': best_acc,
                                'best_f1': best_f1})
                    
            if args.wandb:
                wandb.finish()
                
            # Update the training dataset
            if best_acc > best_acc_baseline:
                print(f'Round {_+1} - Improvement')
                train_loader = augmented_loader
                
                best_acc_baseline = best_acc
                best_f1_baseline = best_f1
                print(f'Round {_+1} - Best accuracy: {best_acc_baseline}, Best f1: {best_f1_baseline}')
            else:
                print(f'Round {_+1} - No improvement')
            
if __name__ == '__main__':
    main()  