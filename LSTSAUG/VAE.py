#---------------------------------- Imports ----------------------------------#

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import f1_score
from sklearn.neighbors import KNeighborsClassifier
from utils import to_default_device

#---------------------------------- VAE Model ----------------------------------#

class VAE(nn.Module):
    def __init__(self, input_dim, num_classes, hidden_dim=1000, hidden_dim_classifier=1000, latent_dim=720, learning_rate=1e-4):
        super(VAE, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.latent_dim = latent_dim
        self.alpha = nn.Parameter(torch.tensor(0.5).float())
        self.beta = nn.Parameter(torch.tensor(0.1).float())

        self.encoder = self.build_encoder(input_dim, hidden_dim)
        self.decoder = self.build_decoder(latent_dim, input_dim)
        self.mean_layer = nn.Linear(hidden_dim, latent_dim)
        self.log_var_layer = nn.Linear(hidden_dim, latent_dim)
        self.classifier = self.build_classifier(latent_dim, hidden_dim_classifier, num_classes)
        self.beta = nn.Parameter(torch.tensor(0.1).float())
        
        self.knn = KNeighborsClassifier(n_neighbors=3)

        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate, weight_decay=1e-5)
        self.loss_function = nn.CrossEntropyLoss()
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.1, patience=100, verbose=True)

    def build_encoder(self, input_dim, hidden_dim):
        return nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
    
    def build_decoder(self, latent_dim, output_dim):
        return nn.Sequential(
            nn.Linear(latent_dim, output_dim),
            nn.Tanh()
        )

    def build_classifier(self, latent_dim, hidden_dim, num_classes):
        return nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes)          
        )

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
    
    def contrastive_loss(self, z, labels, margin=1.0):
        pairwise_distances = torch.cdist(z, z, p=2)
        labels = labels.argmax(dim=1)
        positive_mask = labels.unsqueeze(0) == labels.unsqueeze(1)
        negative_mask = ~positive_mask

        positive_loss = positive_mask * pairwise_distances.pow(2)
        negative_loss = negative_mask * nn.functional.relu(margin - pairwise_distances).pow(2)

        contrastive_loss = (positive_loss + negative_loss).mean()
        return contrastive_loss


    def calculate_dispersion_loss(self, mu, target):
        loss = 0
        target = target.argmax(dim=1)
        for i in range(self.num_classes):
            indices = (target == i).nonzero(as_tuple=True)[0]
            mu_i = mu[indices]
            loss += torch.sum(torch.norm(mu_i - mu_i.mean(dim=0), dim=1).pow(2))
        return loss / len(target)
    
    def calculate_kl_divergence(self, mu, log_var):
        return -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    
    def calculate_overlap_loss(self, mu, target):
        target = target.argmax(dim=1)
        loss = 0
        masks = [target == i for i in range(self.num_classes)]
        
        for i in range(self.num_classes):
            for j in range(i + 1, self.num_classes):
                indices_i = masks[i].nonzero(as_tuple=True)[0]
                indices_j = masks[j].nonzero(as_tuple=True)[0]
                
                if len(indices_i) == 0 or len(indices_j) == 0:
                    continue
                
                mu_i = mu[indices_i]
                mu_j = mu[indices_j]
                dist = torch.cdist(mu_i, mu_j, p=5)  
                loss += dist.sum()
        
        origin = torch.zeros_like(mu[0])
        for i in range(self.num_classes):
            indices = masks[i].nonzero(as_tuple=True)[0]
            mu_i = mu[indices]
            dist_origin = torch.norm(mu_i - origin, dim=1).pow(2)
            loss += dist_origin.sum()
        
        return -loss / len(target)

    def train_epoch(self, data_loader):
        self.train()
        train_loss = 0
        train_recon_loss = 0
        train_kl_div = 0
        train_class_loss = 0
        train_dispersion_loss = 0
        train_overlap_loss = 0  

        for batch, target in data_loader:
            self.optimizer.zero_grad()
            x = batch.to(next(self.parameters()).device)
            target = target.to(next(self.parameters()).device)
            x_hat, mu, log_var, y_pred = self(x)
            
            recon_loss = nn.functional.mse_loss(x_hat, x, reduction='mean')
            kl_div = self.calculate_kl_divergence(mu, log_var)
            class_loss = self.loss_function(y_pred, target.argmax(dim=1))
            fischer_loss = self.contrastive_loss(mu, target)
            # print('Losses:', 'recon_loss:', recon_loss.item(), 'kl_div:', kl_div.item(), 'class_loss:', class_loss.item(), 'fischer_loss:', fischer_loss.item(), 'dispersion_loss:', dispersion_loss.item(), 'overlap_loss:', overlap_loss.item(), 'dist_to_origin:', dist_to_origin.item() )
            loss = class_loss + kl_div + fischer_loss
            loss.backward()
            self.optimizer.step()
            
            train_loss += loss.item()
            train_recon_loss += recon_loss.item()
            train_kl_div += kl_div.item()
            train_class_loss += class_loss.item()

        
        self.scheduler.step(train_loss)
        
        # Pre-train the classifier
        x= []
        y = []
        for batch, target in data_loader:
            x.append(batch)
            y.append(target)
        x = torch.cat(x, dim=0)
        y = torch.cat(y, dim=0)
        x = x.to(next(self.parameters()).device)
        y = y.to(next(self.parameters()).device)
        
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        self.knn.fit(z.detach().cpu().numpy(), y.argmax(dim=1).cpu().numpy())
        
        return train_loss, train_recon_loss, train_kl_div, train_class_loss, 0, 0

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
            x_hat, mu, log_var, _ = self(x)
            y_pred = self.knn.predict(mu.detach().cpu().numpy())
            accuracy = (y_pred == y.argmax(dim=1).cpu().numpy()).mean()
            f1 = f1_score(y.argmax(dim=1).cpu().numpy(), y_pred, average='macro')
            return accuracy, f1
    
    def augment(self, x, num_samples):
        self.eval()
        with torch.no_grad():
            encoded = self.encoder(x)
            mu, log_var = torch.chunk(encoded, 2, dim=1)
            z = self.reparameterize(mu, log_var)
            z = z.unsqueeze(1).expand(-1, num_samples, -1).reshape(-1, self.hidden_dim)
            return self.decoder(z).cpu().numpy()
