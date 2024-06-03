#---------------------------------- Imports ----------------------------------#

import torch
from torch.utils.data import TensorDataset, DataLoader
from utils import to_default_device
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import tsaug

#---------------------------------- Data Loader ----------------------------------#

def getUCRLoader(data_dir, dataset_name, batch_size, transform=None):
    path = data_dir + '/UCRArchive_2018/{}/'.format(dataset_name)

    train_file = path + '{}_TRAIN.tsv'.format(dataset_name)
    test_file = path + '{}_TEST.tsv'.format(dataset_name)

    train_data = pd.read_csv(train_file, sep='\t', header=None)
    test_data = pd.read_csv(test_file, sep='\t', header=None)

    nb_classes = len(train_data[0].unique())
    min_class = train_data[0].min()

    if min_class != 0:  # Re-index classes
        train_data[0] = train_data[0] - min_class
        test_data[0] = test_data[0] - min_class

    print('Number of detected classes : {}'.format(nb_classes))
    print('Number of detected samples in the training set : {}'.format(len(train_data)))
    print('Number of detected samples in the test set : {}'.format(len(test_data)))

    train_np = train_data.to_numpy()
    test_np = test_data.to_numpy()
    train = train_np.reshape(np.shape(train_np)[0], 1, np.shape(train_np)[1])
    test = test_np.reshape(np.shape(test_np)[0], 1, np.shape(test_np)[1])

    y_train = train[:, 0, 0]
    y_test = test[:, 0, 0]
    X_train = train[:, 0, 1:]
    X_test = test[:, 0, 1:]

    # Scale data to [0, 1]
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.int64)
    y_test = torch.tensor(y_test, dtype=torch.int64)

    # One-hot encoding of the class labels
    y_train = torch.nn.functional.one_hot(y_train, num_classes=nb_classes)
    y_test = torch.nn.functional.one_hot(y_test, num_classes=nb_classes)

    train_dataset = []
    for i in range(len(X_train)):
        train_dataset.append((X_train[i], y_train[i].squeeze(0)))

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataset = [X_test, y_test]

    return train_loader, test_dataset, nb_classes, scaler

def augment_loader(data_loader, model, num_samples, distance=1, scaler=None,num_classes=6):
    '''
    Generate new samples by sampling from the neighborhood of the input samples in the latent space. Returns a data loader.
    '''
    batch_size = data_loader.batch_size
    X_list = []
    y_list = []

    # Iterate through the entire data loader and accumulate the batches
    for X_batch, y_batch in data_loader:
        X_list.append(X_batch)
        y_list.append(y_batch)

    # Concatenate all the accumulated batches
    X = torch.cat(X_list, dim=0)
    y = torch.cat(y_list, dim=0)

    X = to_default_device(X)
    y = to_default_device(y)
    model.eval()
    with torch.no_grad():
      for class_idx in range(num_classes):
            class_samples = X[y.argmax(dim=1) == class_idx]
            if len(class_samples) == 0:
                continue
            else :
                neighbors = []
                for neighbor_idx in range(num_samples):
                    idx = np.random.randint(len(class_samples))
                    x = class_samples[idx].unsqueeze(0)
                    
                    mu, log_var = model.encode(x)
                    z = model.reparameterize(mu, log_var)
                    
                    neighbors.append(z)
                neighbors = torch.cat(neighbors, dim=0)
                X_aug = model.decode(neighbors)
                y_aug = torch.nn.functional.one_hot(torch.tensor([class_idx]*len(neighbors)), num_classes=num_classes)
                if class_idx == 0:
                    X_aug_list = X_aug
                    y_aug_list = y_aug
                else:
                    X_aug_list = torch.cat((X_aug_list, X_aug), dim=0)
                    y_aug_list = torch.cat((y_aug_list, y_aug), dim=0)
    X_aug = X_aug_list
    y_aug = y_aug_list
    
    # To default device
    X_aug = to_default_device(X_aug)
    y_aug = to_default_device(y_aug)
    print(X.shape)
    print(y.shape)
    
    print(X_aug.shape)
    print(y_aug.shape)
    
    batch_size = int(batch_size * (1 + num_samples))
    
    augment_loader = DataLoader(
        TensorDataset(torch.cat((X, X_aug), dim=0), torch.cat((y, y_aug), dim=0)),
        batch_size=batch_size,
        shuffle=True   
    )
    
    return augment_loader

def augment_loader_old(data_loader, model, num_samples, distance=1, sample_idx=None, scaler=None,num_classes=6):
    '''
    Generate new samples by sampling from the neighborhood of the input samples in the latent space. Returns a data loader.
    '''
    batch_size = data_loader.batch_size
    X_list = []
    y_list = []

    # Iterate through the entire data loader and accumulate the batches
    for X_batch, y_batch in data_loader:
        X_list.append(X_batch)
        y_list.append(y_batch)

    # Concatenate all the accumulated batches
    X = torch.cat(X_list, dim=0)
    y = torch.cat(y_list, dim=0)

    X = to_default_device(X)
    y = to_default_device(y)
    model.eval()
    with torch.no_grad():
        if sample_idx is None:
            mu, log_var = model.encode(X)
            Z = model.reparameterize(mu, log_var)
            X_aug_list = []
            for i in range(num_samples):
                noise = torch.randn(1, Z.shape[1]) * distance
                noise = to_default_device(noise)
                X_aug = model.decode(Z + noise)
                X_aug_list.append(X_aug)
            X_aug = torch.cat(X_aug_list, dim=0)
            y_aug = y.unsqueeze(1).repeat(1, num_samples, 1).view(-1, y.shape[1])
                    
        else:
            mu, log_var = model.encode(X[sample_idx])
            Z = model.reparameterize(mu, log_var)
            X_aug_list = []
            for i in range(num_samples):
                noise = torch.randn_like(Z) * distance
                noise = to_default_device(noise)
                X_aug = model.decode(Z + noise)
                X_aug_list.append(X_aug)
            X_aug = torch.cat(X_aug_list, dim=0)
            y_aug = y[sample_idx].unsqueeze(1).repeat(1, num_samples, 1).view(-1, y.shape[1])

    # Scale the augmented data using the same scaler
    if scaler is not None:
        X_aug_np = X_aug.cpu().numpy()  # Convert to numpy for scaling
        X_aug_np = scaler.transform(X_aug_np)  # Apply the scaler
        X_aug = torch.tensor(X_aug_np, dtype=torch.float32)
        X_aug = to_default_device(X_aug)

    if sample_idx is not None:
        batch_size = int(batch_size * (1 + len(sample_idx) / len(X)))
    else:
        batch_size = int(batch_size * (1 + num_samples))

    augment_loader = DataLoader(
        TensorDataset(torch.cat((X, X_aug), dim=0), torch.cat((y, y_aug), dim=0)),
        batch_size=batch_size,
        shuffle=True
    )
    return augment_loader

def to_default_device(data):
    if torch.cuda.is_available():
        return data.cuda()
    return data

def tw_loader(data_loader,num_sample):
    '''
    Generate new samples by using time warping on the input samples. Returns a data loader.
    '''
    batch_size = data_loader.batch_size
    X, y = next(iter(data_loader))
    X = to_default_device(X)
    y = to_default_device(y)
    
    # Iterate over the data loader and generate augmented samples
    X_aug, y_aug = [X], [y]

    for i, (batch, target) in enumerate(data_loader):
        x = batch
        x = to_default_device(x)
        target = to_default_device(target)
        for element in time_warp(x,num_sample):
            X_aug.append(to_default_device(torch.tensor(element)))
            y_aug.append(target)
        
    # Convert the augmented samples to PyTorch tensors and create a new data loader
    X_aug = torch.cat(X_aug, dim=0)
    y_aug = torch.cat(y_aug, dim=0)
    augment_loader = torch.utils.data.DataLoader(
        TensorDataset(X_aug, y_aug),
        batch_size=batch_size*(num_sample+1),
        shuffle=True
    )
    
    return augment_loader

def time_warp(x,num_sample):
    '''
    Generate new samples by using time warping on the input samples. Returns a tensor.
    '''
    x = x.cpu().numpy()
    X_aug = []
    for i in range(num_sample):
        x_aug = tsaug.TimeWarp().augment(x)
        X_aug.append(x_aug)

    return X_aug