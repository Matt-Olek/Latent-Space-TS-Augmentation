#---------------------------------- Imports ----------------------------------#

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import f1_score
from utils import to_default_device

#---------------------------------- VAE Model ----------------------------------#

class VAE(nn.Module):
    def __init__(self, input_dim, num_classes, hidden_dim=10000, hidden_dim_classifier=1000, latent_dim=720):
        super(VAE, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes

        # Encoder layers
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
        )

        # Decoder layers
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, input_dim),
            nn.Sigmoid()
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

        self.optimizer = optim.Adam(self.parameters(), lr=1e-3)
        self.loss_function = nn.CrossEntropyLoss()
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.1, patience=1000)

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

    def train_epoch(self, data_loader):
        self.train()
        train_loss = 0
        train_recon_loss = 0
        train_kl_div = 0
        train_class_loss = 0
        for i, (batch, target) in enumerate(data_loader):
            self.optimizer.zero_grad()
            x = batch.to(next(self.parameters()).device)
            target = target.to(next(self.parameters()).device)
            x_hat, mu, log_var, y_pred = self(x)
            recon_loss = nn.functional.mse_loss(x_hat, x, reduction='sum')
            kl_div = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
            class_loss = self.loss_function(y_pred, target.argmax(dim=1))
            loss = recon_loss + class_loss + kl_div
            loss.backward()
            self.optimizer.step()
            train_loss += loss.item()
            train_recon_loss += recon_loss.item()
            train_kl_div += kl_div.item()
            train_class_loss += class_loss.item()
        self.scheduler.step(train_loss)
        
        return train_loss, train_recon_loss, train_kl_div, train_class_loss

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