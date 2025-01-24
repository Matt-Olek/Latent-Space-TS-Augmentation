# ---------------------------------- Imports ----------------------------------#

from LSTSAUG.expansion_eval import eval_gmms
import torch
from torch.utils.data import TensorDataset, DataLoader
from utils import to_default_device
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import tsaug
import pyro
import pyro.distributions as dist


# ---------------------------------- Data Loader ----------------------------------#


def getUCRLoader(data_dir, dataset_name, batch_size, transform=None, plot=True):
    path = data_dir + "/UCRArchive_2018/{}/".format(dataset_name)

    train_file = path + "{}_TRAIN.tsv".format(dataset_name)
    test_file = path + "{}_TEST.tsv".format(dataset_name)

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

    # batch_size = max(batch_size, len(train_data) // 10)
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
    scaler=None,
    num_classes=6,
    alpha=1,
    return_augmented_only=False,
):
    """
    Generate new samples by sampling from the neighborhood of the input samples in the latent space using a Gaussian Mixture Model (GMM).
    Returns a data loader.
    """
    batch_size = data_loader.batch_size
    sample_dim = data_loader.dataset[0][0].shape[0]
    X_list = []
    y_list = []

    # Iterate through the entire data loader and accumulate the batches
    for X_batch, y_batch in data_loader:
        X_list.append(X_batch)
        y_list.append(y_batch)

    # Concatenate all the accumulated batches
    X = torch.cat(X_list, dim=0).float()  # Ensure consistent dtype (float32)
    y = torch.cat(y_list, dim=0).float()  # Ensure consistent dtype (float32)

    X = to_default_device(X)
    y = to_default_device(y)
    model.eval()

    # Compute latent space representations for all samples
    with torch.no_grad():
        mu, log_var = model.encode(X)

    # Fit a GMM for each class
    gmms = []
    for class_idx in range(num_classes):
        class_indices = (
            (y.argmax(dim=1) == class_idx).nonzero(as_tuple=True)[0].cpu().numpy()
        )
        mu_class = mu[class_indices].cpu().numpy()
        log_var_class = log_var[class_indices].cpu().numpy()

        # Fit a GMM with as many components as there are samples in the class
        gmm = GaussianMixture(n_components=len(class_indices), covariance_type="full")
        gmm.fit(mu_class)
        for i in range(gmm.n_components):
            gmm.covariances_[i] = np.diag(np.exp(log_var_class[i]) * alpha)
            gmm.means_[i] = mu_class[i]

        gmms.append(gmm)

    X_aug_list = []
    y_aug_list = []
    
    # Evaluate the trustworthiness of the GMMs
    mean_trust, class_trusts, all_probs = eval_gmms(gmms, num_samples) 

    # Augment data for each class using its GMM
    for class_idx in range(num_classes):
        z_samples, _ = gmms[class_idx].sample(num_samples)
        z_samples = to_default_device(
            torch.tensor(z_samples).float()
        ).detach()  # Ensure float dtype

        # Compute log probabilities for each class and filter samples
        for other_class_idx in range(num_classes):
            if z_samples.shape[0] == 0:
                break
            if other_class_idx == class_idx:
                continue
            else:
                log_prob_class = gmms[class_idx].score_samples(z_samples.cpu().numpy())
                log_prob_other_class = gmms[other_class_idx].score_samples(
                    z_samples.cpu().numpy()
                )

                # Keep samples where the log probability for the current class is higher
                should_be_kept = log_prob_class > log_prob_other_class
                z_samples = z_samples[should_be_kept]
                print(
                    "Class",
                    class_idx,
                    ": Keeping",
                    should_be_kept.sum(),
                    "samples out of",
                    len(should_be_kept),
                )

        # If no samples remain, skip this class
        if len(z_samples) == 0:
            print(f"No samples retained for class {class_idx}")
            continue

        # Decode the remaining latent samples into the input space
        x_augs = model.decode(z_samples).detach().float()  # Ensure float dtype
        y_aug = (
            torch.nn.functional.one_hot(
                to_default_device(torch.tensor([class_idx] * len(z_samples))),
                num_classes=num_classes,
            )
            .detach()
            .float()
        )  # Ensure float dtype

        X_aug_list.append(x_augs)
        y_aug_list.append(y_aug)

    # Check if no augmented data is left
    if len(X_aug_list) == 0 or len(y_aug_list) == 0:
        raise ValueError("All augmented samples were dropped. No data to return.")

    X_aug = torch.cat(X_aug_list, dim=0).float()  # Ensure float dtype
    y_aug = torch.cat(y_aug_list, dim=0).float()  # Ensure float dtype

    if return_augmented_only:
        augment_loader = DataLoader(
            TensorDataset(X_aug, y_aug), batch_size=batch_size, shuffle=True
        )
    else:
        if X_aug.shape[0] == 0:
            augment_loader = DataLoader(
                TensorDataset(X, y), batch_size=batch_size, shuffle=True
            )
        else:
            augment_loader = DataLoader(
                TensorDataset(
                    torch.cat((X, X_aug), dim=0).float(),  # Ensure float dtype
                    torch.cat((y, y_aug), dim=0).float(),  # Ensure float dtype
                ),
                batch_size=batch_size,
                shuffle=True,
            )
        print("Augmented data size:", len(augment_loader.dataset))

    return augment_loader


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
