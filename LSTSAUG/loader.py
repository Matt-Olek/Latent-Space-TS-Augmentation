#---------------------------------- Imports ----------------------------------#

import torch
from torch.utils.data import TensorDataset
from utils import to_default_device
import numpy as np
import pandas as pd
import tsaug

#---------------------------------- Data Loader ----------------------------------#

def getUCRLoader(data_dir,dataset_name, batch_size, transform=None):
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

    return train_loader, test_dataset, nb_classes

def augment_loader(data_loader, model, num_samples, distance=10):
    '''
    Generate new samples by sampling from the neighborhood of the input samples in the latent space. Returns a data loader.
    '''
    batch_size = data_loader.batch_size
    X, y = next(iter(data_loader))
    X = to_default_device(X)
    y = to_default_device(y)
    model.eval()
    with torch.no_grad():
        # Iterate over the data loader and generate augmented samples
        X_aug, y_aug = [X], [y]
        for i, (batch, target) in enumerate(data_loader):
            x = batch
            x = to_default_device(x)
            target = to_default_device(target)
            mu, log_var = torch.chunk(model.encoder(x), 2, dim=1)
            z = model.reparameterize(mu, log_var)

            # Generate num_samples augmented samples for each input sample
            for j in range(num_samples):
                eps = torch.randn_like(z) * distance # Small random noise
                z_aug = z + eps
                x_aug_j = model.decoder(z_aug)
                X_aug.append(x_aug_j)
                y_aug.append(target)

        # Convert the augmented samples to PyTorch tensors and create a new data loader
        X_aug = torch.cat(X_aug, dim=0)
        y_aug = torch.cat(y_aug, dim=0)
        augment_loader = torch.utils.data.DataLoader(
            TensorDataset(X_aug, y_aug),
            batch_size=batch_size*(num_samples+1),
            shuffle=True
        )

    return augment_loader

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