# ---------------------------------- Imports ----------------------------------#

from expansion_eval import eval_gmms
import torch
from torch.utils.data import TensorDataset, DataLoader
from utils import to_default_device
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import tsaug
import pyro
import pyro.distributions as dist
from gmmx import GaussianMixtureModelJax, EMFitter

# ---------------------------------- Data Loader ----------------------------------#


def malwareLoader(data_dir, batch_size, transform=None, plot=True):
    # Define paths to training and testing data
    data_dir = data_dir + "UCI_HAR_Dataset/"
    train_data_path = data_dir + "train/X_train.txt"
    train_labels_path = data_dir + "train/y_train.txt"
    test_data_path = data_dir + "test/X_test.txt"
    test_labels_path = data_dir + "test/y_test.txt"

    # Load the training and testing data
    X_train = pd.read_csv(train_data_path, delim_whitespace=True, header=None).values
    y_train = pd.read_csv(
        train_labels_path, delim_whitespace=True, header=None
    ).values.flatten()
    X_test = pd.read_csv(test_data_path, delim_whitespace=True, header=None).values
    y_test = pd.read_csv(
        test_labels_path, delim_whitespace=True, header=None
    ).values.flatten()

    batch_size = max(batch_size, len(X_train) // 10)
    # Get the number of classes
    nb_classes = len(np.unique(y_train))

    y_train = y_train - min(y_train)
    y_test = y_test - min(y_test)

    # Scale the data to the range [-1, 1]
    scaler = MinMaxScaler(feature_range=(-1, 1))
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Convert the data and labels to PyTorch tensors
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.int64)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.int64)

    # One-hot encode the class labels
    y_train = torch.nn.functional.one_hot(y_train, num_classes=nb_classes)
    y_test = torch.nn.functional.one_hot(y_test, num_classes=nb_classes)

    # Create TensorDataset for training and testing data
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = [X_test, y_test]

    # Create DataLoader for training data
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Return the train loader, test dataset, number of classes, and the scaler
    return train_loader, test_dataset, nb_classes, scaler


def getUCRLoader(data_dir, dataset_name, batch_size, transform=None, plot=True):
    path_train = data_dir + "/UCR/{}/".format(dataset_name)
    path_test = data_dir + "/UCR/{}/".format(dataset_name)

    train_file = path_train + "{}_TRAIN.tsv".format(dataset_name)
    test_file = path_test + "{}_TEST.tsv".format(dataset_name)

    train_data = pd.read_csv(train_file, sep="\t", header=None)
    test_data = pd.read_csv(test_file, sep="\t", header=None)

    nb_classes = len(train_data[0].unique())
    min_class = train_data[0].min()

    if min_class == 0:
        pass
    elif min_class > 0:
        train_data[0] = train_data[0] - min_class
        test_data[0] = test_data[0] - min_class
    elif min_class == -1:
        train_data[0] = train_data[0].replace(-1, 0).replace(1, 1)
        test_data[0] = test_data[0].replace(-1, 0).replace(1, 1)

    if plot:
        print("Building loader for dataset : {}".format(dataset_name))
        print("Number of detected classes : {}".format(nb_classes))
        print("Classes : {}".format(train_data[0].unique()))
        print(
            "Number of detected samples in the training set : {}".format(
                len(train_data)
            )
        )
        print("Number of detected samples in the test set : {}".format(len(test_data)))

    batch_size = max(batch_size, len(train_data) // 10)
    if plot:
        print("Batch size : {}".format(batch_size))
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

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    test_dataset = [X_test, y_test]

    return train_loader, test_dataset, nb_classes, scaler


from sklearn.mixture import GaussianMixture
import torch
from torch.utils.data import DataLoader, TensorDataset


def augment_loader(
    data_loader,
    model,
    num_samples,
    num_classes=10,
    alpha=1.0,
    return_augmented_only=False,
    config=None,
    logs=None,
    step=None,
    scaler=None,
):
    batch_size = data_loader.batch_size

    X_list, y_list = [], []
    for X_batch, y_batch in data_loader:
        X_list.append(X_batch)
        y_list.append(y_batch)

    X = torch.cat(X_list, dim=0).float()
    y = torch.cat(y_list, dim=0).float()

    device = next(model.parameters()).device
    X, y = X.to(device), y.to(device)

    model.eval()
    with torch.no_grad():
        mu, log_var = model.encode(X)

    X_aug_list, y_aug_list = [], []

    for class_idx in range(num_classes):
        class_mask = (y.argmax(dim=1) == class_idx)
        if class_mask.sum() == 0:
            continue

        mu_class = mu[class_mask]
        log_var_class = log_var[class_mask]

        
        
        # cov = centered.T @ centered / (mu_class.size(0) - 1)
        # cov = torch.diag(torch.exp(log_var_class).mean(dim=0)) * alpha
        # cov_e = np.cov(mu_class.T.cpu(), bias=False)  # shape: (D, D)
        # to cuda
        # cov_e = torch.tensor(cov_e, dtype=torch.float32, device=device)

        eps = 1e-4  # ou 1e-6 selon stabilité

        mean = mu_class.mean(dim=0)
        cov = torch.cov(mu_class.T)  # shape: (D, D)
        cov += eps * torch.eye(cov.shape[0], device=device)  # régularisation
        cov = cov * alpha  # extension


        # print(f"[Class {class_idx}] cov shape: {cov.shape}, min diag: {cov.diag().min()}, max diag: {cov.diag().max()}")

        mvn = dist.MultivariateNormal(mean, covariance_matrix=cov)
        z_samples = mvn.sample((num_samples,)).to(device)

        # log-prob filtering
        # log_prob_class = mvn.log_prob(z_samples)
        # keep_mask = torch.ones(num_samples, dtype=torch.bool, device=device)
        

# Log prob de la vraie classe pour tous les samples
        log_prob_class = mvn.log_prob(z_samples)

        # On stocke tous les log_probs des autres classes
        log_probs_other = []
        # print(mu.shape)
        # print(f'==== class idx : {class_idx} ====')
        for other_idx in range(num_classes):
            if other_idx == class_idx:
                continue
            # print(f'other_idx : {other_idx}')
            other_mask = (y.argmax(dim=1) == other_idx)
            # print(other_mask.shape)
            # print(f'other_idx : {other_idx}, sum : {other_mask.sum()}')
            if other_mask.sum() == 0:
                continue


            
            mu_other = mu[other_mask]
            mean_o = mu_other.mean(dim=0)
            cov_o = torch.cov(mu_other.T) + eps * torch.eye(mu_other.size(1), device=device)
            cov_o = cov_o * alpha

            mvn_o = dist.MultivariateNormal(mean_o, covariance_matrix=cov_o)
            log_prob_other = mvn_o.log_prob(z_samples)
            log_probs_other.append(log_prob_other)

        # Calcul du log-prob de la classe cible (déjà fait)
        log_prob_target = log_prob_class
        # print('de la classe')
        # print(log_prob_target.shape)

        # On empile les log-probs des autres classes (skip index 0 qui est la classe cible)
        if num_classes > 2:
            
            log_probs_others = torch.stack(log_probs_other[1:], dim=0)
        else:
            log_probs_others = torch.stack(log_probs_other, dim=0)
        # print('des autres classes')
        # print(log_probs_others.shape)
        # Calcul de la meilleure log-prob parmi les autres classes (worst-case filtering)
        max_logprob_other, _ = torch.max(log_probs_others, dim=0)
        # print(max_logprob_other.shape)
        # Filtrage via log-ratio : on garde uniquement les échantillons plus probables pour la classe cible
        margin = 0.5
        keep_mask = log_prob_target > (max_logprob_other + margin)

        z_samples = z_samples[keep_mask]

        if len(z_samples) == 0:
            continue

        x_aug = model.decode(z_samples).detach()
        y_aug = torch.nn.functional.one_hot(
            torch.tensor([class_idx] * len(z_samples), device=device),
            num_classes=num_classes
        ).float()

        X_aug_list.append(x_aug)
        y_aug_list.append(y_aug)

    if len(X_aug_list) == 0:
        raise ValueError("All augmented samples were dropped. No data to return.")

    X_aug = torch.cat(X_aug_list, dim=0)
    y_aug = torch.cat(y_aug_list, dim=0)
    
    print(f'AUGMENTED DATA : {X_aug.shape}, {y_aug.shape}')

    if return_augmented_only:
        return DataLoader(TensorDataset(X_aug, y_aug), batch_size=batch_size, shuffle=True), logs
    else:
        X_orig = X.detach()
        y_orig = y.detach()
        X_combined = torch.cat([X_orig, X_aug], dim=0)
        y_combined = torch.cat([y_orig, y_aug], dim=0)
        return DataLoader(TensorDataset(X_combined, y_combined), batch_size=batch_size, shuffle=True), logs



def to_default_device(data):
    if torch.cuda.is_available():
        return data.cuda()
    return data


def tw_loader(data_loader, num_sample):
    """
    Generate new samples by using time warping on the input samples. Returns a data loader.
    """
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
        for element in time_warp(x, num_sample):
            X_aug.append(to_default_device(torch.tensor(element)))
            y_aug.append(target)

    # Convert the augmented samples to PyTorch tensors and create a new data loader
    X_aug = torch.cat(X_aug, dim=0)
    y_aug = torch.cat(y_aug, dim=0)
    augment_loader = torch.utils.data.DataLoader(
        TensorDataset(X_aug, y_aug),
        batch_size=batch_size * (num_sample + 1),
        shuffle=True,
    )

    return augment_loader


def time_warp(x, num_sample):
    """
    Generate new samples by using time warping on the input samples. Returns a tensor.
    """
    x = x.cpu().numpy()
    X_aug = []
    for i in range(num_sample):
        x_aug = tsaug.TimeWarp().augment(x)
        X_aug.append(x_aug)

    return X_aug


def simple_augment_loader(data_loader, model, alpha=0.1, return_augmented_only=False):
    """
    Generates new samples by adding random Gaussian noise in the latent space of the model.
    Returns a data loader with augmented data. SIMPLE LATENT AUGMENTATION
    """
    batch_size = data_loader.batch_size
    X_list, y_list = [], []

    # Collect all samples from the data loader
    for X_batch, y_batch in data_loader:
        X_list.append(X_batch)
        y_list.append(y_batch)

    # Concatenate all batches into a single tensor
    X = torch.cat(X_list, dim=0)
    y = torch.cat(y_list, dim=0)

    # Move to the appropriate device
    X = X.to(next(model.parameters()).device)
    y = y.to(next(model.parameters()).device)

    model.eval()
    with torch.no_grad():
        # Encode to the latent space
        mu, log_var = model.encode(X)
        z = model.reparameterize(mu, log_var)

        # Add random Gaussian noise in the latent space
        noise = torch.randn_like(z) * alpha
        z_aug = z + noise

        # Decode the augmented latent vectors back to input space
        X_aug = model.decode(z_aug)

    # Keep the same labels for augmented data
    y_aug = y.clone()

    if return_augmented_only:
        augment_loader = DataLoader(
            TensorDataset(X_aug, y_aug), batch_size=batch_size, shuffle=True
        )
    else:
        X_combined = torch.cat((X, X_aug), dim=0)
        y_combined = torch.cat((y, y_aug), dim=0)
        augment_loader = DataLoader(
            TensorDataset(X_combined, y_combined), batch_size=batch_size, shuffle=True
        )

    return augment_loader
