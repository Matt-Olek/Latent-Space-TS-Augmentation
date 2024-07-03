#---------------------------------- Imports ----------------------------------#
import tqdm
import wandb
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from utils import to_default_device, get_model_path
from visualization import plot_latent_space_viz, plot_latent_space_neighbors, build_gif
from sklearn.metrics import f1_score
from sklearn.neighbors import KNeighborsClassifier
from utils import to_default_device

#---------------------------------- VAE Model ----------------------------------#

class VAE(nn.Module):
    def __init__(self, input_dim, num_classes, hidden_dim=1000, hidden_dim_classifier=1000, latent_dim=720, learning_rate=1e-4, knn=5):
        super(VAE, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.latent_dim = latent_dim
        
        self.encoder = self.build_encoder(input_dim, hidden_dim)
        self.decoder = self.build_decoder(latent_dim, hidden_dim, input_dim)
        
        self.mean_layer = nn.Linear(hidden_dim, latent_dim)
        self.log_var_layer = nn.Linear(hidden_dim, latent_dim)
        self.classifier = self.build_classifier(latent_dim, hidden_dim_classifier, num_classes)
        
        self.knn = KNeighborsClassifier(n_neighbors=knn)

        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate, weight_decay=1e-5)
        self.loss_function = nn.CrossEntropyLoss()
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.1, patience=100)

    def build_encoder(self, input_dim, hidden_dim):
        return nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
    
    def build_decoder(self, latent_dim, hidden_dim, output_dim):
        return nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
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
    
    def fit_knn(self, data_loader):
        x, y = [], []
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
        
    def contrastive_loss(self, z, labels, margin=1.0):
        pairwise_distances = torch.cdist(z, z, p=2)
        labels = labels.argmax(dim=1)
        positive_mask = labels.unsqueeze(0) == labels.unsqueeze(1)
        negative_mask = ~positive_mask

        positive_loss = positive_mask * pairwise_distances.pow(2)
        negative_loss = negative_mask * nn.functional.relu(margin - pairwise_distances).pow(2)

        contrastive_loss = (positive_loss + negative_loss).mean()
        return contrastive_loss
    
    def calculate_kl_divergence(self, mu, log_var):
        return -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())

    def train_epoch(self, data_loader):
        self.train()
        total_loss = 0
        total_recon_loss = 0
        total_kl_div = 0
        total_class_loss = 0
        total_contrastive_loss = 0

        for batch, target in data_loader:
            self.optimizer.zero_grad()
            x = to_default_device(batch)
            target = to_default_device(target)

            x_hat, mu, log_var, y_pred = self(x)
            
            # Calculate losses
            kl_div = self.calculate_kl_divergence(mu, log_var)
            contrastive_loss = self.contrastive_loss(mu, target)
            class_loss = self.loss_function(y_pred, target.argmax(dim=1))
            recon_loss = nn.functional.mse_loss(x_hat, x, reduction='mean')

            # Combine losses
            loss = class_loss + kl_div + contrastive_loss + recon_loss

            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Update running totals
            total_loss += loss.item()
            total_recon_loss += recon_loss.item()
            total_kl_div += kl_div.item()
            total_class_loss += class_loss.item()
            total_contrastive_loss += contrastive_loss.item()

        # Adjust learning rate based on total loss
        self.scheduler.step(total_loss)

        # Fit KNN
        self.fit_knn(data_loader)

        # Average the losses over all batches
        num_batches = len(data_loader)
        total_loss /= num_batches
        total_recon_loss /= num_batches
        total_kl_div /= num_batches
        total_class_loss /= num_batches
        total_contrastive_loss /= num_batches

        return total_loss, total_recon_loss, total_kl_div, total_class_loss, total_contrastive_loss

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
            x_hat, mu, log_var, _ = self(x)
            y_pred = self.knn.predict(mu.detach().cpu().numpy())
            accuracy = (y_pred == y.argmax(dim=1).cpu().numpy()).mean()
            f1 = f1_score(y.argmax(dim=1).cpu().numpy(), y_pred, average='weighted')
            return accuracy, f1
    
    def augment(self, x, num_samples):
        self.eval()
        with torch.no_grad():
            encoded = self.encoder(x)
            mu, log_var = torch.chunk(encoded, 2, dim=1)
            z = self.reparameterize(mu, log_var)
            z = z.unsqueeze(1).expand(-1, num_samples, -1).reshape(-1, self.latent_dim)
            return self.decoder(z).cpu().numpy()

    def train_vae(self, train_loader, test_dataset, nb_classes, config, logs, name='vae'):
        if config["WANDB"]:
            wandb.init(project=config["WANDB_PROJECT"],
                        config=config,
                        tags=['train',config["DATASET"], name],
                        name=f'{config["DATASET"]} {name}')     
            wandb.watch(self)
            
        best_acc = 0
        best_f1 = 0
        early_stop_counter = 0
        early_stop_patience = config["EARLY_STOP_PATIENCE"]
        for epoch in tqdm.tqdm(range(config["VAE_NUM_EPOCHS"])):
            train_loss, train_recon_loss, train_kl_div, train_class_loss, train_contrastive_loss = self.train_epoch(train_loader)
            
            # Test the model
            acc, f1 = self.validate(test_dataset)
            if config["WANDB"]:
                wandb.log({ 'train_loss': train_loss,
                            'train_recon_loss': train_recon_loss,
                            'train_kl_div': train_kl_div,
                            'train_class_loss': train_class_loss,
                            'train_contrastive_loss': train_contrastive_loss,
                            'lr': float(self.scheduler.get_last_lr()[0]),
                            'test_accuracy': acc,
                            'test_f1': f1})
            if epoch % 100 == 0 and config["AUGMENT_PLOT"]:
                plot_latent_space_viz(self, train_loader, test_dataset, num_classes=nb_classes, type='3d', id=epoch)
                plot_latent_space_neighbors(self,train_loader, num_neighbors=5, alpha=config["ALPHA"], num_classes=nb_classes)
            
            if acc > best_acc:
                best_acc = acc
                best_f1 = f1
                early_stop_counter = 0
            else:
                early_stop_counter += 1
                if early_stop_counter >= early_stop_patience:
                    print('Early stopping triggered.')
                    break
        print('Best test accuracy:', best_acc)
                
        # Save the trained model
        if config["SAVE_VAE"]:
            model_path = get_model_path(config, name)
            torch.save(self.state_dict(), model_path)
            print(f'Model saved at {model_path}')  
            
        build_gif()
            
        # Save the logs
        logs['vae_best_acc'] = best_acc
        logs['vae_best_f1'] = best_f1
        
        
        if config["WANDB"]:
            wandb.finish()
            
        return self, logs