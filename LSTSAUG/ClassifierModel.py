import torch 
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score
from utils import to_default_device

# ------------------------------ ResNet ------------------------------ #

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size//2)                  # Convolution 1
        self.bn1 = nn.BatchNorm1d(out_channels)                                                                 # Batch Normalization 1
        self.relu = nn.ReLU()                                                                                   # ReLU 1
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=kernel_size//2)                 # Convolution 2  
        self.bn2 = nn.BatchNorm1d(out_channels)                                                                 # Batch Normalization 2

        if in_channels != out_channels:
            self.shortcut = nn.Sequential(                                                                      # Shortcut connection
                nn.Conv1d(in_channels, out_channels, kernel_size=1),
                nn.BatchNorm1d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()
            
        self.initialize_weights()

    def forward(self, x):
        residual = x
        out = self.conv1(x)                     # Convolution 1
        out = self.bn1(out)                     # Batch Normalization 1
        out = self.relu(out)                    # ReLU 1
        out = self.conv2(out)                   # Convolution 2
        out = self.bn2(out)                     # Batch Normalization 2
        out += self.shortcut(residual)          # Residual connection
        out = self.relu(out)                    # ReLU 2
        return out
    
    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.xavier_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

class Classifier_RESNET(nn.Module):
    def __init__(self, input_shape, nb_classes, lr=0.001, weight_decay=0.0001):
        super(Classifier_RESNET, self).__init__()

        n_feature_maps = 64
        self.conv1 = nn.Conv1d(1, n_feature_maps, kernel_size=8, padding=4)
        self.bn1 = nn.BatchNorm1d(n_feature_maps)
        self.relu = nn.ReLU()

        self.residual_block1 = ResidualBlock(n_feature_maps, n_feature_maps, kernel_size=3)                 # Residual Block 1
        self.residual_block2 = ResidualBlock(n_feature_maps, n_feature_maps * 2, kernel_size=3)             # Residual Block 2
        self.residual_block3 = ResidualBlock(n_feature_maps * 2, n_feature_maps * 2, kernel_size=3)         # Residual Block 3

        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(n_feature_maps * 2, nb_classes)
        
        self.initialize_weights()
        
        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=lr, weight_decay=weight_decay)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.1, patience=250 , verbose=True)


    def forward(self, x):
        out = self.conv1(x)                                     # Convolution 1
        out = self.bn1(out)                                     # Batch Normalization 1
        out = self.relu(out)                                    # ReLU 1
        out = self.residual_block1(out)                         # Residual Block 1
        out = self.residual_block2(out)                         # Residual Block 2
        out = self.residual_block3(out)                         # Residual Block 3
        out = self.global_avg_pool(out)                         # Global Average Pooling
        out = torch.flatten(out, 1)                             # Flatten
        out = self.fc(out)                                      # Fully Connected
        return out
    
    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.xavier_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)
    
    def train_epoch(self, train_loader):
        self.train()
        train_loss = 0
        correct = 0
        total = 0
        for batch_idx, (X, y) in enumerate(train_loader):
            self.optimizer.zero_grad()
            X = X.unsqueeze(1)
            X = to_default_device(X)
            y = to_default_device(y)
            output = self(X)
            loss = self.criterion(output.float(), y.float())
            loss.backward()
            self.optimizer.step()
            acc = torch.sum(torch.argmax(output, dim=1) == torch.argmax(y, dim=1))
            
            train_loss += loss.item()
            correct += acc.item()
            total += len(y)
        epoch_loss = train_loss / total
        self.scheduler.step(epoch_loss)
        epoch_acc = correct / total
        return epoch_loss, epoch_acc
    
    def validate(self, test_data):
        self.eval()
        with torch.no_grad():
            x, y = test_data
            x = x.unsqueeze(1)
            x = to_default_device(x)
            y = to_default_device(y)
            y_pred = self(x)
            accuracy = torch.sum(torch.argmax(y_pred, dim=1) == torch.argmax(y, dim=1)).item() / len(y)
            f1 = f1_score(y.argmax(dim=1).cpu().numpy(), y_pred.argmax(dim=1).cpu().numpy(), average='weighted')
            return accuracy, f1