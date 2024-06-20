#---------------------------------- Imports ----------------------------------#

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import f1_score
from utils import to_default_device

#---------------------------------- VAE Model ----------------------------------#

class VAE(nn.Module):
    def __init__(self, input_dim, num_classes, hidden_dim=1000, hidden_dim_classifier=1000, latent_dim=720, learning_rate=1e-4):
        super(VAE, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes

        # Encoder layers
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        # Decoder layers
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, input_dim),
            nn.Tanh()
        )
        
        # Mean and log variance layers
        
        self.mean_layer = nn.Linear(hidden_dim, latent_dim)
        self.log_var_layer = nn.Linear(hidden_dim, latent_dim)

        # Classifier layer
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim_classifier),
            nn.ReLU(),
            nn.Linear(hidden_dim_classifier, num_classes)
        )
        self.beta = nn.Parameter(torch.tensor(0.1).float())

        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate, weight_decay=1e-5)
        self.loss_function = nn.CrossEntropyLoss()
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.1, patience=100, verbose=True)

    def encode(self, x):
        h = self.encoder(x)
        return self.mean_layer(h), self.log_var_layer(h)
    
    def decode(self, z):
        return self.decoder(z)
    
    def reparameterize(self, mean, var):
        eps = torch.randn_like(var)
        return mean + var * eps 

    def forward(self, x):
        mean, log_var = self.encode(x)
        z = self.reparameterize(mean, log_var)
        x_hat = self.decode(z)
        y_pred = self.classifier(z)

        return x_hat, mean, log_var, y_pred
    
    def calculate_dispersion_loss(self, mu, target):
        loss= 0
        target = target.argmax(dim=1)
        for i in range(self.num_classes):
            indices = (target == i).nonzero(as_tuple=True)[0]
            mu_i = mu[indices]
            loss += torch.sum(torch.norm(mu_i - mu_i.mean(dim=0), dim=1))
        return loss/len(target)
    
    def calculate_kl_divergence(self, mu, log_var, target):
        return -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    
    def calculate_overlap_loss(self, mu, target):
        # Convert one-hot encoded target to class indices
        target = target.argmax(dim=1)
        
        # Initialize the loss
        loss = 0
        
        # Create a mask for each class
        masks = [target == i for i in range(self.num_classes)]
        
        for i in range(self.num_classes):
            for j in range(i + 1, self.num_classes):  # Start from i+1 to avoid redundant computations
                indices_i = masks[i].nonzero(as_tuple=True)[0]
                indices_j = masks[j].nonzero(as_tuple=True)[0]
                
                if len(indices_i) == 0 or len(indices_j) == 0:
                    continue
                
                mu_i = mu[indices_i]
                mu_j = mu[indices_j]
                
                # Compute the pairwise distances using broadcasting
                dist = torch.cdist(mu_i, mu_j, p=5)  
                
                # Sum the distances and add to the loss
                loss += dist.sum()
        
        # Add a penalty for classes that are too far away from the origin
        origin = torch.zeros_like(mu[0])
        for i in range(self.num_classes):
            indices = masks[i].nonzero(as_tuple=True)[0]
            mu_i = mu[indices]
            dist_origin = torch.norm(mu_i - origin, dim=1)
            loss += dist_origin.sum()
        
        return -loss/len(target)

    def train_epoch(self, data_loader):
        self.train()
        train_loss = 0
        train_recon_loss = 0
        train_kl_div = 0
        train_class_loss = 0
        train_dispersion_loss = 0
        train_overlap_loss = 0  
        for i, (batch, target) in enumerate(data_loader):
            self.optimizer.zero_grad()
            x = batch.to(next(self.parameters()).device)
            target = target.to(next(self.parameters()).device)
            x_hat, mu, log_var, y_pred = self(x)
            recon_loss = nn.functional.mse_loss(x_hat, x, reduction='sum')
            kl_div = self.calculate_kl_divergence(mu, log_var, target)
            class_loss = self.loss_function(y_pred, target.argmax(dim=1))
            dispersion_loss = self.calculate_dispersion_loss(mu, target)
            overlap_loss = self.calculate_overlap_loss(mu, target)
            loss =  3 * class_loss + dispersion_loss + overlap_loss + kl_div #recon_loss
            loss.backward()
            self.optimizer.step()
            train_loss += loss.item()
            train_recon_loss += recon_loss.item()
            train_kl_div += kl_div.item()
            train_class_loss += class_loss.item()
            train_dispersion_loss += dispersion_loss.item()
            train_overlap_loss += overlap_loss
        self.scheduler.step(train_loss)
        
        return train_loss, train_recon_loss, train_kl_div, train_class_loss, train_dispersion_loss, train_overlap_loss

    def generate(self, num_samples):
        self.eval()
        with torch.no_grad():
            z = torch.randn(num_samples, self.hidden_dim)
            return self.decoder(z).cpu().numpy()
        
    def validate(self, test_data):
        self.eval()
        with torch.no_grad():
            x, y = test_data
            x = to_default_device(x)
            y = to_default_device(y)
            x_hat, mu, log_var, y_pred = self(x)
            accuracy = (y_pred.argmax(dim=1) == y.argmax(dim=1)).float().mean().item()
            f1 = f1_score(y.argmax(dim=1).cpu().numpy(), y_pred.argmax(dim=1).cpu().numpy(), average='weighted')
            return accuracy, f1
    
    def augment(self, x, num_samples):
        '''
        Generate new samples by sampling from the neighborhood of the input samples in the latent space.
        '''
        self.eval()
        with torch.no_grad():
            encoded = self.encoder(x)
            mu, log_var = torch.chunk(encoded, 2, dim=1)
            z = self.reparameterize(mu, log_var)
            z = z.unsqueeze(1).expand(-1, num_samples, -1).reshape(-1, self.hidden_dim)
            return self.decoder(z).cpu().numpy()