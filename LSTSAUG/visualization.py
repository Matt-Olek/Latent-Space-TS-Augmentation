import torch
import numpy as np
from utils import to_default_device
import matplotlib.pyplot as plt

#---------------------------------- Plotting ----------------------------------#

def plot_latent_space_neighbors(vae, test_dataset, num_neighbors=5, distance=1, num_classes=6):
    # Set the model to evaluation mode
    vae.eval()
    
    X_test, y_test = test_dataset
    X_test = to_default_device(X_test)
    y_test = to_default_device(y_test)
    
    fig, axes = plt.subplots(num_classes, 1, figsize=(10, 4 * num_classes))
    
    for class_idx in range(num_classes):
        class_samples = X_test[y_test.argmax(dim=1) == class_idx]
        others = np.random.randint(len(class_samples), size=5)
        others_x = [class_samples[i].unsqueeze(0) for i in others]
        if len(class_samples) == 0:
            continue
        neighbors = []
        # Compute the mean and variance of the latent space for the current class
        with torch.no_grad():
            for neighbor_idx in range(num_neighbors):
                # Randomly sample a data point from the current class
                idx = np.random.randint(len(class_samples))
                x = class_samples[idx].unsqueeze(0)
                
                # Get the latent space representation
                mu, log_var = vae.encode(x)
                z = vae.reparameterize(mu, log_var)
                
                neighbors.append(z)
        neighbors = torch.cat(neighbors, dim=0)
        with torch.no_grad():
            decoded_neighbors = vae.decode(neighbors)
            
        # Convert tensors to numpy for plotting
        decoded_neighbors = decoded_neighbors.cpu().numpy()
        others_samples = [x.squeeze().cpu().numpy() for x in others_x]
        
        # Plot the original and neighboring points for the current class
        ax = axes[class_idx]
        
        for i, other in enumerate(others_samples):
            ax.plot(other, label=f'Class Sample {i+1}', color='gray')
            
        for i, decoded in enumerate(decoded_neighbors):
            ax.plot(decoded, label=f'Neighbor {i+1}', color='green', alpha=(i+1)/(num_neighbors+1))
        
        ax.set_title(f'Class {class_idx}')
        ax.legend()
    
    plt.tight_layout()
    plt.savefig('results/latent_space_neighbors.png')
    plt.close()

def plot_latent_space_viz(vae, train_loader, test_dataset, num_classes=6):
    '''
    Plot the latent space representation of the test dataset by classes on different dimensions.
    '''
    # Set the model to evaluation mode
    vae.eval()
    train_dataset = train_loader.dataset
    X_train_list = []
    y_train_list = []

    # Iterate through the entire data loader and accumulate the batches
    for X_batch, y_batch in train_loader:
        X_train_list.append(X_batch)
        y_train_list.append(y_batch)

    # Concatenate all the accumulated batches
    X_train = torch.cat(X_train_list, dim=0)
    y_train = torch.cat(y_train_list, dim=0)
    
    X_train = to_default_device(X_train)
    y_train = to_default_device(y_train)
    
    X_test, y_test = test_dataset
    X_test = to_default_device(X_test)
    y_test = to_default_device(y_test)
    
    fig, axes = plt.subplots(1, 3, figsize=(30, 10))
    
    colors = ['blue', 'red', 'green', 'purple', 'orange', 'brown', 'pink', 'gray', 'olive', 'cyan', 'magenta', 'yellow', 'black', 'lime', 'teal', 'aqua', 'navy', 'maroon', 'silver', 'gold'] * 10
    for class_idx in range(num_classes):
        class_samples_train = X_train[y_train.argmax(dim=1) == class_idx]
        class_samples_test = X_test[y_test.argmax(dim=1) == class_idx]
        if len(class_samples_train) == 0 and len(class_samples_test) == 0:
            continue
        # Compute the mean and variance of the latent space for the current class
        with torch.no_grad():
            mu_train, log_var_train = vae.encode(class_samples_train)
            z_train = vae.reparameterize(mu_train, log_var_train)
            
            mu_test, log_var_test = vae.encode(class_samples_test)
            z_test = vae.reparameterize(mu_test, log_var_test)
        
        # Convert tensors to numpy for plotting
        z_train = z_train.cpu().numpy()
        z_test = z_test.cpu().numpy()
        
        # Plot the latent space representation of the current class
        ax = axes[0]
        ax.scatter(z_train[:, 0], z_train[:, 1], label=f'Train Class {class_idx}', color=colors[class_idx], alpha=0.5)
        ax.scatter(z_test[:, 0], z_test[:, 1], label=f'Test Class {class_idx}', color=colors[class_idx], alpha=1)
        ax.set_title('Latent Space Visualization (Dim 0 vs Dim 1)')
        
        ax = axes[1]
        ax.scatter(z_train[:, 1], z_train[:, 2], label=f'Train Class {class_idx}', color=colors[class_idx], alpha=0.5)
        ax.scatter(z_test[:, 1], z_test[:, 2], label=f'Test Class {class_idx}', color=colors[class_idx], alpha=1)
        ax.set_title('Latent Space Visualization (Dim 1 vs Dim 2)')
        
        ax = axes[2]
        ax.scatter(z_train[:, 2], z_train[:, 3], label=f'Train Class {class_idx}', color=colors[class_idx], alpha=0.5)
        ax.scatter(z_test[:, 2], z_test[:, 3], label=f'Test Class {class_idx}', color=colors[class_idx], alpha=1)
        ax.set_title('Latent Space Visualization (Dim 2 vs Dim 3)')
    
    plt.tight_layout()
    plt.savefig('results/latent_space_viz.png')
    # print('Saved latent space visualization to results/latent_space_viz.png')
    plt.close()