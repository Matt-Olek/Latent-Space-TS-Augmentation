#---------------------------------- Imports ----------------------------------#

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import f1_score
from utils import to_default_device

#---------------------------------- VAE Model ----------------------------------#

class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim, num_classes):
        super(VAE, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.num_classes = num_classes

        # Encoder layers
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, latent_dim * 2)  # Output mean and log variance
        )

        # Decoder layers
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, input_dim),
            nn.Sigmoid()  # Output in range [0, 1] for time series data
        )

        # Classifier layer
        self.classifier = nn.Linear(latent_dim, num_classes)

        self.optimizer = optim.Adam(self.parameters(), lr=0.01)
        self.loss_function = nn.CrossEntropyLoss()
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.1, patience=150)

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        encoded = self.encoder(x)
        mu, log_var = torch.chunk(encoded, 2, dim=1)
        z = self.reparameterize(mu, log_var)
        decoded = self.decoder(z)
        y_pred = self.classifier(z)

        return decoded, mu, log_var, y_pred

    def train_epoch(self, data_loader):
        self.train()
        train_loss = 0
        train_recon_loss = 0
        train_kl_div = 0
        train_class_loss = 0
        for i, (batch, target) in enumerate(data_loader):
            self.optimizer.zero_grad()
            x = to_default_device(batch)
            target = to_default_device(target)
            x_hat, mu, log_var, y_pred = self(x)
            recon_loss = nn.functional.mse_loss(x_hat, x, reduction='sum')
            kl_div = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
            class_loss = self.loss_function(y_pred.float(), target.argmax(dim=1))
            loss = recon_loss + kl_div + class_loss
            loss.backward()
            self.optimizer.step()
            train_loss += loss.item()
            train_recon_loss += recon_loss.item()
            train_kl_div += kl_div.item()
            train_class_loss += class_loss.item()
        self.scheduler.step(train_loss)
        train_loss /= len(data_loader.dataset)
        train_recon_loss /= len(data_loader.dataset)
        train_kl_div /= len(data_loader.dataset)
        
        return train_loss, train_recon_loss, train_kl_div, train_class_loss

    def generate(self, num_samples):
        self.eval()
        with torch.no_grad():
            z = torch.randn(num_samples, self.latent_dim)
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
            z = z.unsqueeze(1).expand(-1, num_samples, -1).reshape(-1, self.latent_dim)
            return self.decoder(z).cpu().numpy()