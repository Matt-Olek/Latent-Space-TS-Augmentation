#---------------------------------- Imports ----------------------------------#

import torch
import numpy as np
import pandas as pd

#---------------------------------- Data Loader ----------------------------------#

def getUCRLoader(dataset_name, batch_size, transform=None):
    path = 'data/UCRArchive_2018/{}/'.format(dataset_name)

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

def augment_loader(data_loader, model, num_samples):
    '''
    Generate new samples by sampling from the neighborhood of the input samples in the latent space. Returns a data loader.
    '''
    batch_size = data_loader.batch_size
    model.eval()
    with torch.no_grad():
        X, y = next(iter(data_loader))
        encoded = model.encoder(X)
        mu, log_var = torch.chunk(encoded, 2, dim=1)
        z = model.reparameterize(mu, log_var)
        # Generate neighbors by adding small random noise to the latent vector
        neighbors = [z + torch.randn_like(z) * 1 for _ in range(num_samples)]
        neighbors.append(z)
        neighbors = torch.cat(neighbors, dim=0)
        decoded_neighbors = model.decoder(neighbors)
        decoded_neighbors = decoded_neighbors.cpu().numpy()
        
        augmented_dataset = []
        for i in range(len(decoded_neighbors)):
            augmented_dataset.append((decoded_neighbors[i], y[0]))
            
    augmented_loader = torch.utils.data.DataLoader(augmented_dataset, batch_size=batch_size*num_samples, shuffle=True)
    return augmented_loader