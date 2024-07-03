#---------------------------------- Imports ----------------------------------#

import torch
from torch.utils.data import TensorDataset, DataLoader
from utils import to_default_device
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import tsaug
import pyro 
import pyro.distributions as dist


#---------------------------------- Data Loader ----------------------------------#

def getUCRLoader(data_dir, dataset_name, batch_size, transform=None):
    path = data_dir + '/UCRArchive_2018/{}/'.format(dataset_name)

    train_file = path + '{}_TRAIN.tsv'.format(dataset_name)
    test_file = path + '{}_TEST.tsv'.format(dataset_name)

    train_data = pd.read_csv(train_file, sep='\t', header=None)
    test_data = pd.read_csv(test_file, sep='\t', header=None)

    nb_classes = len(train_data[0].unique())
    min_class = train_data[0].min()

    if min_class == 0:
        pass
    elif min_class > 0:
        train_data[0] = train_data[0] - min_class
        test_data[0] = test_data[0] - min_class
    elif min_class == -1 :
        train_data[0] = train_data[0].replace(-1, 0).replace(1, 1)
        test_data[0] = test_data[0].replace(-1, 0).replace(1, 1)

    print('Building loader for dataset : {}'.format(dataset_name))
    print('Number of detected classes : {}'.format(nb_classes))
    print('Classes : {}'.format(train_data[0].unique()))
    print('Number of detected samples in the training set : {}'.format(len(train_data)))
    print('Number of detected samples in the test set : {}'.format(len(test_data)))

    batch_size = max(batch_size, len(train_data)//10)
    print('Batch size : {}'.format(batch_size))
    train_np = train_data.to_numpy()
    test_np = test_data.to_numpy()
    train = train_np.reshape(np.shape(train_np)[0], 1, np.shape(train_np)[1])
    test = test_np.reshape(np.shape(test_np)[0], 1, np.shape(test_np)[1])

    y_train = train[:, 0, 0]
    y_test = test[:, 0, 0]
    X_train = train[:, 0, 1:]
    X_test = test[:, 0, 1:]

    # Scale data to [-1;1]
    
    scaler = MinMaxScaler(feature_range=(-1, 1))
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

def augment_loader(data_loader, model, num_samples, scaler=None, num_classes=6, alpha=1, return_augmented_only=False):
    """
    Generate new samples by sampling from the neighborhood of the input samples in the latent space. Returns a data loader.
    """
    batch_size = data_loader.batch_size
    sample_dim = data_loader.dataset[0][0].shape[0]
    alpha_increment = 0.01
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

    # Compute latent space representations for all samples
    with torch.no_grad():
        mu, log_var = model.encode(X)
        z = model.reparameterize(mu, log_var)

    # Initialize Gaussian Mixture Models for each class using mu and log_var
    gmns = []
    for class_idx in range(num_classes):
        class_indices = (y.argmax(dim=1) == class_idx).nonzero(as_tuple=True)[0].cpu().numpy()
        class_mu = mu[class_indices].cpu().numpy().squeeze()
        
        mean = class_mu.mean(axis=0)
        var = class_mu.var(axis=0) * alpha
        
        gmn = dist.MultivariateNormal(
            to_default_device(torch.tensor(mean)),
            to_default_device(torch.tensor(np.diag(var)))
        )
        gmns.append(gmn)
        
    X_aug_list = []
    y_aug_list = []
    
    for class_idx in range(num_classes):
        z_samples = gmns[class_idx].sample([num_samples]).detach()
        x_augs = model.decode(z_samples).detach()
        y_aug = torch.nn.functional.one_hot(to_default_device(torch.tensor([class_idx]*len(z_samples))), num_classes=num_classes).detach()
        X_aug_list.append(x_augs)
        y_aug_list.append(y_aug)
        
    X_aug = torch.cat(X_aug_list, dim=0)
    y_aug = torch.cat(y_aug_list, dim=0)
    
    if return_augmented_only:
        augment_loader = DataLoader(
            TensorDataset(X_aug, y_aug),
            batch_size=batch_size,
            shuffle=True
        )
    else:
        augment_loader = DataLoader(
            TensorDataset(torch.cat((X, X_aug), dim=0), torch.cat((y, y_aug), dim=0)),
            batch_size=batch_size,
            shuffle=True
        )
        print('Augmented data size:', len(augment_loader.dataset))
        
    return augment_loader
        
        

def augment_loader_oold(data_loader, model, num_samples, distance=1, scaler=None, num_classes=6, alpha=1, return_augmented_only=False):
    """
    Generate new samples by sampling from the neighborhood of the input samples in the latent space. Returns a data loader.
    """
    batch_size = data_loader.batch_size
    sample_dim = data_loader.dataset[0][0].shape[0]
    print('Sample dimension:', sample_dim)
    alpha_increment = 0.01
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

    # Compute latent space representations for all samples
    with torch.no_grad():
        mu, log_var = model.encode(X)
        z = model.reparameterize(mu, log_var)

    # Initialize Gaussian Mixture Models for each class using mu and log_var
    gmms = []
    for class_idx in range(num_classes):
        class_indices = (y.argmax(dim=1) == class_idx).nonzero(as_tuple=True)[0].cpu().numpy()
        class_mu = mu[class_indices].cpu().numpy().squeeze()
        class_var = torch.exp(log_var[class_indices])
        class_covar = [torch.diag(class_var[i]) * alpha for i in range(len(class_indices))]
        class_covar = torch.stack(class_covar).cpu().numpy()
        class_covar_tensor = torch.tensor(class_covar)
        class_mu_tensor = torch.tensor(class_mu)

        mgm = dist.MixtureSameFamily(
            dist.Categorical(torch.ones(len(class_indices))/len(class_indices)),
            dist.MultivariateNormal(class_mu_tensor, class_covar_tensor)
        )
        gmms.append(mgm)

    X_aug_list = []
    y_aug_list = []
    real_size_of_samples = 0
    alpha_update = False

    while not alpha_update:
        print('Trying alpha = {}'.format(alpha))
        X_aug_list = []
        y_aug_list = []
        real_size_of_samples = 0
        with torch.no_grad():
            for class_idx in range(num_classes):
                print('Class', class_idx)
                z_samples = gmms[class_idx].sample([num_samples])
                real_log_dens = gmms[class_idx].log_prob(z_samples)
                kept_z_samples = z_samples
                for other_class_idx in range(num_classes):
                    if kept_z_samples.shape[0] == 0:
                        break
                    if other_class_idx == class_idx:
                        continue
                    else:
                        log_dens_prob_of_other_class = gmms[other_class_idx].log_prob(kept_z_samples)

                        # prob = real_log_dens - torch.log(torch.tensor(1) + torch.exp(log_dens_prob_of_other_class - real_log_dens))
                        should_be_kept = log_dens_prob_of_other_class < real_log_dens
                        kept_z_samples = kept_z_samples[should_be_kept]
                        print('Kept {} samples'.format(len(kept_z_samples)), 'out of', len(z_samples))
                        real_log_dens = real_log_dens[should_be_kept]

                z_samples = kept_z_samples

                if len(z_samples) > 0:
                    # Did not drop all samples
                    z_tensors = torch.tensor(z_samples, device=X.device, dtype=torch.float32)
                    x_augs = model.decode(z_tensors)
                    y_aug = torch.nn.functional.one_hot(torch.tensor([class_idx]*len(z_tensors)), num_classes=num_classes)
                    X_aug_list.append(x_augs)
                    y_aug_list.append(y_aug)
                    real_size_of_samples += len(z_samples)

                    if z_samples.shape[0] < num_samples:
                        alpha_update = True
                        print('Dropped {} samples'.format(num_samples - len(z_samples)))
                    else:
                        alpha_update = True
                else :
                    alpha_update = True
                    print('Dropped {} samples'.format(num_samples))

        # If no samples were dropped, increment alpha and try again
        if not alpha_update:
            print('No samples dropped')
            alpha += alpha_increment
            gmms = []
            for class_idx in range(num_classes):
                class_indices = (y.argmax(dim=1) == class_idx).nonzero(as_tuple=True)[0]
                class_mu = mu[class_indices].cpu().numpy().squeeze()
                class_var = torch.exp(log_var[class_indices]).cpu().numpy()
                class_covar = [torch.diag(torch.exp(log_var[class_indices][i])) * alpha for i in range(len(class_indices))]
                class_covar = torch.stack(class_covar).cpu().numpy()
                class_covar_tensor = torch.tensor(class_covar)
                class_mu_tensor = torch.tensor(class_mu)

                mgm = dist.MixtureSameFamily(
                    dist.Categorical(torch.ones(len(class_indices))),
                    dist.MultivariateNormal(class_mu_tensor, class_covar_tensor)
                )
                gmms.append(mgm)

    if X_aug_list:
        X_aug = torch.cat(X_aug_list, dim=0)
        y_aug = torch.cat(y_aug_list, dim=0)
    else:
        X_aug = torch.tensor([]).to(X.device)
        y_aug = torch.tensor([]).to(X.device)

    X_aug = to_default_device(X_aug)
    y_aug = to_default_device(y_aug)

    if return_augmented_only:
        augment_loader = DataLoader(
            TensorDataset(X_aug, y_aug),
            batch_size=batch_size,
            shuffle=True
        )
    else:
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
