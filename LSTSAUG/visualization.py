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
    plt.show()



def plot_latent_space_neighbors_old(vae, test_dataset, num_neighbors=5, distance=1, num_classes=6):
    # Set the model to evaluation mode
    vae.eval()
    
    X_test, y_test = test_dataset
    X_test = to_default_device(X_test)
    y_test = to_default_device(y_test)
    
    fig, axes = plt.subplots(num_classes, 1, figsize=(10, 4 * num_classes))
    
    for class_idx in range(num_classes):
        class_samples = X_test[y_test.argmax(dim=1) == class_idx]
        
        if len(class_samples) == 0:
            continue
        
        # Randomly sample a data point from the current class
        idx = np.random.randint(len(class_samples))
        others = np.random.randint(len(class_samples), size=4)
        others_x = [class_samples[i].unsqueeze(0) for i in others]
        x = class_samples[idx].unsqueeze(0)
        
        # Get the latent space representation
        with torch.no_grad():
            mu, log_var = vae.encode(x)
            z = vae.reparameterize(mu, log_var)
        
        # Generate neighbors by adding small random noise to the latent vector
        neighbors = [z + torch.randn_like(z) * distance for _ in range(num_neighbors)]
        neighbors.append(z)  # Include the original point for reference
        neighbors = torch.cat(neighbors, dim=0)
        
        # Decode the neighbors
        with torch.no_grad():
            decoded_neighbors = vae.decode(neighbors)
        
        # Convert tensors to numpy for plotting
        original_sample = x.squeeze().cpu().numpy()
        others_samples = [x.squeeze().cpu().numpy() for x in others_x]
        decoded_neighbors = decoded_neighbors.cpu().numpy()
        
        # Plot the original and neighboring points for the current class
        ax = axes[class_idx]
        
        ax.plot(original_sample, label='Original Sample', color='black')
        for i, other in enumerate(others_samples):
            ax.plot(other, label=f'Other Sample {i+1}', color='gray')
            
        for i, decoded in enumerate(decoded_neighbors):
            ax.plot(decoded, label=f'Neighbor {i+1}', color='green', alpha=(i+1)/(num_neighbors+1))
        
        ax.set_title(f'Class {class_idx}')
        ax.legend()
    
    plt.tight_layout()
    plt.savefig('results/latent_space_neighbors.png')
    plt.show()