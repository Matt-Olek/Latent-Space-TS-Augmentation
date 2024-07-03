import os 
import torch
from PIL import Image
import numpy as np
from utils import to_default_device
from loader import augment_loader
import matplotlib.pyplot as plt

#---------------------------------- Plotting ----------------------------------#

def plot_latent_space_neighbors(vae, data_loader, num_neighbors=5, alpha=1, num_classes=6):
    
    augmented_loader = augment_loader(data_loader, vae, num_neighbors*10, num_classes = num_classes, alpha = alpha, return_augmented_only=True)

    fig, axes = plt.subplots(num_classes, 1, figsize=(10, 4 * num_classes))
    
    X_ground_truth = []
    y_ground_truth = []
    X_augmented = []
    y_augmented = []
    
    for X, y in data_loader:
        X_ground_truth.append(X)
        y_ground_truth.append(y)
    for X, y in augmented_loader:
        X_augmented.append(X)
        y_augmented.append(y)
        
    X_ground_truth = torch.cat(X_ground_truth, dim=0)
    y_ground_truth = torch.cat(y_ground_truth, dim=0)
    
    X_augmented = torch.cat(X_augmented, dim=0)
    y_augmented = torch.cat(y_augmented, dim=0)
    
    for class_idx in range(num_classes):
        class_ground_truth = X_ground_truth[y_ground_truth.argmax(dim=1) == class_idx]
        class_augmented = X_augmented[y_augmented.argmax(dim=1) == class_idx]
        
        # plot 5 samples from the class or less if there are less than 5
        num_true_samples = len(class_ground_truth)
        num_augmented_samples = len(class_augmented)
        num_samples = min(num_true_samples, num_augmented_samples, 5)
        if num_samples == 0:
            continue
        for i in range(num_samples):
            ax = axes[class_idx]
            ax.plot(class_ground_truth[i].cpu().numpy(), label=f'Class Sample {i+1}', color='gray')
            ax.plot(class_augmented[i].detach().cpu().numpy(), label=f'Augmented Sample {i+1}', color='green', alpha=0.5)
        ax.set_title(f'Class {class_idx}')
        ax.legend()
        
    plt.tight_layout()
    plt.savefig('results/visualization/neighbors.png')
    plt.close()

def plot_latent_space_neighbors_old(vae, test_dataset, num_neighbors=5, distance=1, num_classes=6):
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
                logvar = log_var * distance
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

def plot_latent_space_viz(vae, train_loader, test_dataset, num_classes=6, type='3d', id=0):
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
    
    colors = ['blue', 'red', 'green', 'purple', 'orange', 'brown', 'pink', 'gray', 'olive', 'cyan', 'magenta', 'yellow', 'black', 'lime', 'teal', 'aqua', 'navy', 'maroon', 'silver', 'gold'] * 10
    
    if type == '2d':
        
        fig, axes = plt.subplots(1, 3, figsize=(30, 10))
        
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
    
    elif type == '3d':
        fig = plt.figure(figsize=(15, 15))
        ax = fig.add_subplot(111, projection='3d')
        
        for class_idx in range(num_classes):
            class_samples_train = X_train[y_train.argmax(dim=1) == class_idx]
            class_samples_test = X_test[y_test.argmax(dim=1) == class_idx]
            if len(class_samples_train) == 0 and len(class_samples_test) == 0:
                continue
            # Compute the mean and variance of the latent space for the current class
            with torch.no_grad():
                mu_train, log_var_train = vae.encode(class_samples_train)
                z_train = vae.reparameterize(mu_train, log_var_train)
                # mu_train_mean = mu_train.mean(dim=0)
                # print(f"Class {class_idx} Train Mean: {mu_train_mean}")
                
                mu_test, log_var_test = vae.encode(class_samples_test)
                z_test = vae.reparameterize(mu_test, log_var_test)
            
            # Convert tensors to numpy for plotting
            z_train = z_train.cpu().numpy()
            z_test = z_test.cpu().numpy()
            
            # Plot the latent space representation of the current class
            ax.scatter(z_train[:, 0], z_train[:, 1], z_train[:, 2], label=f'Train Class {class_idx}', color=colors[class_idx], alpha=0.5)
            ax.scatter(z_test[:, 0], z_test[:, 1], z_test[:, 2], label=f'Test Class {class_idx}', color=colors[class_idx], alpha=1)
            ax.set_title('Latent Space Visualization (Dim 0 vs Dim 1 vs Dim 2)')
            ax.set_xlim([-0.3, 0.3])
            ax.set_ylim([-0.3, 0.3])
            ax.set_zlim([-0.3, 0.3])
            angle_x = 30
            angle_y = id*4
            ax.view_init(angle_x, angle_y)
    
    plt.tight_layout()
    plt.savefig('results/visualization/latent_space_viz.png')
    plt.savefig('results/visualization/gif/viz_{}.png'.format(id))
    # print('Saved latent space visualization to results/latent_space_viz.png')
    plt.close()
    
def build_gif(folder_path='results/visualization/gif', output_path='results/visualization/latent_space_viz.gif', duration=200):
    images = [img for img in os.listdir(folder_path) if img.endswith((".png", ".jpg", ".jpeg", ".bmp", ".gif"))]
    
    # # Sort images to maintain order
    # images.sort()
    
    # Load images into a list
    frames = [Image.open(os.path.join(folder_path, image)) for image in images]
    
    # Save as GIF
    if frames:
        frames[0].save(output_path, format='GIF', append_images=frames[1:], save_all=True, duration=duration, loop=0)