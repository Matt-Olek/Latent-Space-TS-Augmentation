#---------------------------------- Imports ----------------------------------#

import os
import wandb
import argparse
import torch    
import tqdm
import numpy as np
from VAE import VAE
from loader import getUCRLoader, augment_loader
import matplotlib.pyplot as plt
from ClassifierModel import Classifier_RESNET

#---------------------------------- Device ----------------------------------#

global torch_device 
torch_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#---------------------------------- Parser ----------------------------------#  

def get_parser():
    parser = argparse.ArgumentParser(description='LSTSAUG')
    parser.add_argument('--dataset', type=str, default='ECG5000',
                        help='Directory containing the dataset')
    parser.add_argument('--model_dir', type=str, default='models',
                        help='Directory to save models')
    parser.add_argument('--results_dir', type=str, default='results',
                        help='Directory to save results')
    parser.add_argument('--num_epochs', type=int, default=250,
                        help='Number of epochs to train the model')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size for training')
    parser.add_argument('--latent_dim', type=int, default=64,
                        help='Dimension of the latent space')
    parser.add_argument('--learning_rate', type=float, default=0.001,
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
    return parser

#---------------------------------- Plotting ----------------------------------#

def plot_latent_space_neighbors(vae, test_dataset, num_neighbors=5):
    # Set the model to evaluation mode
    vae.eval()
    
    # Randomly sample a data point from the test dataset
    idx = np.random.randint(len(test_dataset))
    x= test_dataset[0][idx]
    x = x.to(torch_device).unsqueeze(0)
    
    # Get the latent space representation
    with torch.no_grad():
        encoded = vae.encoder(x)
        mu, log_var = torch.chunk(encoded, 2, dim=1)
        z = vae.reparameterize(mu, log_var)
    
    # Generate neighbors by adding small random noise to the latent vector
    neighbors = [z + torch.randn_like(z) * 1 for _ in range(num_neighbors)]
    neighbors.append(z)  # Include the original point for reference
    neighbors = torch.cat(neighbors, dim=0)
    
    # Decode the neighbors
    with torch.no_grad():
        decoded_neighbors = vae.decoder(neighbors)
    
    # Convert tensors to numpy for plotting
    original_sample = x.squeeze().cpu().numpy()
    decoded_neighbors = decoded_neighbors.cpu().numpy()
    
    # Plot the original and neighboring points
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
    
    # Plot original sample
    ax1.plot(original_sample, label='Original Sample', color='black')
    ax1.set_title('Original Sample')
    
    # Plot decoded neighbors
    for i, decoded in enumerate(decoded_neighbors):
        ax2.plot(decoded, label=f'Neighbor {i+1}')
    ax2.set_title('Latent Space Neighbors')
    
    # Add legends and show the plots
    ax1.legend()
    ax2.legend()
    plt.show()
    
#---------------------------------- Main ----------------------------------#

def main() :
    # Parse command line arguments
    parser = get_parser()
    args = parser.parse_args()
    
    # Set random seed for reproducibility
    torch.manual_seed(args.seed)
    
    # Create directories for saving models and results
    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)
    
    # Create a data loader for training
    train_loader, test_dataset, nb_classes = getUCRLoader(args.dataset, args.batch_size)
    
    # Initialize the VAE model
    input_dim = train_loader.dataset[0][0].shape[0]
    print("Detected input dimension: ", input_dim)
    vae = VAE(input_dim, args.latent_dim, nb_classes).to(torch_device)
    
    
    if not args.use_trained:
        if args.wandb:
            wandb.init(project='lstsaug',
                        config=vars(args),
                        name=f'{args.dataset}_vae-{args.batch_size}bs-{args.latent_dim}ld-{args.num_epochs}e')
            wandb.watch(vae)
        
        # Train 
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
          
    
    if args.augment_plot:
        #load the trained model
        model_path = os.path.join(args.model_dir, f'{args.dataset}_vae-{args.batch_size}bs-{args.latent_dim}ld-{args.num_epochs}e.pth')
        vae.load_state_dict(torch.load(model_path))
        vae.eval()
        
        plot_latent_space_neighbors(vae, test_dataset)
    
    if args.test_augment :
        # Baseline classifier training
        # classifier = Classifier_RESNET(input_dim, nb_classes).to(torch_device)
        # if args.wandb:
        #     wandb.init(project='lstsaug',
        #                 config=vars(args),
        #                 name=f'{args.dataset}_classifier-{args.batch_size}bs-{args.num_epochs}e')
        #     wandb.watch(classifier)
        
        # for epoch in tqdm.tqdm(range(args.num_epochs)):
        #     train_loss, train_acc = classifier.train_epoch(train_loader)
        #     # test_acc, test_f1 = classifier.validate(test_dataset)
        #     if epoch % 10 == 0:
        #         test_acc, test_f1 = classifier.validate(test_dataset)
        #     if args.wandb:
        #         wandb.log({ 'train_loss': train_loss,
        #                     'train_accuracy': train_acc,
        #                     'test_accuracy': test_acc,
        #                     'test_f1': test_f1})
                

        # if args.wandb:
        #     wandb.finish()
            
        # Augment the dataset
        num_samples = 5
        augmented_loader = augment_loader(train_loader, vae, num_samples)
        
        # Train the classifier on the augmented dataset
        classifier_augmented = Classifier_RESNET(input_dim, nb_classes).to(torch_device)
        if args.wandb:
            wandb.init(project='lstsaug',
                        config=vars(args),
                        name=f'{args.dataset}_classifier_augmented-{args.batch_size}bs-{args.num_epochs}e')
            wandb.watch(classifier_augmented)
            
        for epoch in tqdm.tqdm(range(args.num_epochs)):
            train_loss, train_acc = classifier_augmented.train_epoch(augmented_loader)
            test_acc, test_f1 = classifier_augmented.validate(test_dataset)
            if args.wandb:
                wandb.log({ 'train_loss': train_loss,
                            'train_accuracy': train_acc,
                            'test_accuracy': test_acc,
                            'test_f1': test_f1})
                
        if args.wandb:
            wandb.finish()     
    

if __name__ == '__main__':
    main()