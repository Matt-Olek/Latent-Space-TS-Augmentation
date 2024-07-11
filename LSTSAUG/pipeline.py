import os
import torch
from utils import get_model_path
from VAE import VAE
from ClassifierModel import Classifier_RESNET

from config import config
from loader import getUCRLoader, augment_loader
from utils import to_default_device

def pipeline(config=config):

    logs = {}
    print('Starting script for dataset : {}'.format(config['DATASET']))
    torch.manual_seed(config["SEED"])
    os.makedirs(config["MODEL_DIR"], exist_ok=True)
    os.makedirs(config["RESULTS_DIR"], exist_ok=True)
    
    train_loader, test_dataset, nb_classes, scaler = getUCRLoader(config["DATA_DIR"], config["DATASET"], config["BATCH_SIZE"])

    input_dim = train_loader.dataset[0][0].shape[0]
    num_samples = int(len(train_loader.dataset)/nb_classes)*config["NUM_SAMPLES"]
    
    print("Detected input dimension: ", input_dim)
    print("Number of samples produced per class: ", num_samples)
    
    logs['dataset'] = config["DATASET"]
    logs['num_classes'] = nb_classes
    logs['num_train_samples'] = len(train_loader.dataset)
    logs['num_test_samples'] = len(test_dataset[0])
    
    # ---------------------------- VAE Training ---------------------------- #
    
    vae = to_default_device(VAE(
        input_dim, 
        nb_classes, 
        latent_dim=config['LATENT_DIM'],
        learning_rate=config['VAE_LEARNING_RATE'], 
        hidden_dim=config['VAE_HIDDEN_DIM'],    
        knn = config['VAE_KNN']
    ))
    
    if not config["USE_TRAINED"] or not os.path.exists(model_path):
        print('#'*50 + '\n' +'Training the VAE model...')
        vae, logs = vae.train_vae(train_loader, test_dataset, nb_classes, config, logs)
    else:
        model_path = get_model_path(config, name='vae')
        vae.load_state_dict(torch.load(model_path))
        vae.eval()
        print(f'Model loaded from {model_path}')
    
    # ---------------------------- VAE Augmentation ---------------------------- #
    
    train_loader_augmented = augment_loader(train_loader, vae, num_samples=num_samples, scaler=scaler, num_classes=nb_classes, alpha=config["ALPHA"])
    print('#'*50 + '\n' +'Augmented dataset size: ', len(train_loader_augmented.dataset))
    if config["TEST_AUGMENT"]:
        print('#'*50 + '\n' +'Augmenting the dataset...')
        print('Augmented dataset size: ', len(train_loader_augmented.dataset))
        
        vae_augmented = to_default_device(VAE(
            input_dim, 
            nb_classes, 
            latent_dim=config['LATENT_DIM'],
            learning_rate=config['VAE_LEARNING_RATE'], 
            hidden_dim=config['VAE_HIDDEN_DIM'],
            knn = config['VAE_KNN']
        ))
        
        print('#'*50 + '\n' +'Training the VAE model on augmented data...')
        vae_augmented, logs = vae_augmented.train_vae(train_loader_augmented, test_dataset, nb_classes, config, logs, name='vae_augmented')
        
    # ---------------------------- AUGMENTED VAE Augmentation ---------------------------- #
     
    train_loader_augmented_augmented = augment_loader(train_loader_augmented, vae_augmented, num_samples=num_samples, scaler=scaler, num_classes=nb_classes, alpha=config["ALPHA"])
    if config["TEST_AUGMENT"]:
        print('#'*50 + '\n' +'Augmenting the dataset...')
        print('Augmented dataset size: ', len(train_loader_augmented_augmented.dataset))
        
        vae_augmented_augmented = to_default_device(VAE(
            input_dim, 
            nb_classes, 
            latent_dim=config['LATENT_DIM'],
            learning_rate=config['VAE_LEARNING_RATE'], 
            hidden_dim=config['VAE_HIDDEN_DIM'],
            knn = config['VAE_KNN']
        ))
        
        print('#'*50 + '\n' +'Training the VAE model on augmented data...')
        vae_augmented_augmented, logs = vae_augmented_augmented.train_vae(train_loader_augmented_augmented, test_dataset, nb_classes, config, logs, name='vae_augmented_augmented')
        
    # ---------------------------- AUGMENTED AUGMENTED VAE Augmentation ---------------------------- #
    
    train_loader_augmented_augmented_augmented = augment_loader(train_loader_augmented_augmented, vae_augmented_augmented, num_samples=num_samples, scaler=scaler, num_classes=nb_classes, alpha=config["ALPHA"])
    if config["TEST_AUGMENT"]:
        print('#'*50 + '\n' +'Augmenting the dataset...')
        print('Augmented dataset size: ', len(train_loader_augmented_augmented_augmented.dataset))
        
        vae_augmented_augmented_augmented = to_default_device(VAE(
            input_dim, 
            nb_classes, 
            latent_dim=config['LATENT_DIM'],
            learning_rate=config['VAE_LEARNING_RATE'], 
            hidden_dim=config['VAE_HIDDEN_DIM'],
            knn = config['VAE_KNN']
        ))
        
        print('#'*50 + '\n' +'Training the VAE model on augmented data...')
        vae_augmented_augmented_augmented, logs = vae_augmented_augmented_augmented.train_vae(train_loader_augmented_augmented_augmented, test_dataset, nb_classes, config, logs, name='vae_augmented_augmented_augmented')
        
        
        
    # ---------------------------- Classifier Training ---------------------------- #
    
    if config["BASELINE"]:
        print('#'*50 + '\n' +'Training the baseline classifier...')
        classifier = to_default_device(Classifier_RESNET(
            input_dim, 
            nb_classes, 
            learning_rate=config['CLASSIFIER_LEARNING_RATE'], 
            weight_decay=config['WEIGHT_DECAY']
        ))
        
        classifier, logs = classifier.train_classifier(train_loader, test_dataset, config, logs, name='classifier')
    
    if config["TEST_AUGMENT"]:
        print('#'*50 + '\n' +'Training the augmented classifier...')
        classifier_augmented = to_default_device(Classifier_RESNET(
            input_dim, 
            nb_classes, 
            learning_rate=config['CLASSIFIER_LEARNING_RATE'], 
            weight_decay=config['WEIGHT_DECAY']
        ))
        
        classifier_augmented, logs = classifier_augmented.train_classifier(train_loader_augmented, test_dataset, config, logs, name='classifier_augmented')
        
        print('#'*50 + '\n' +'Training the augmented augmented classifier...')
        classifier_augmented_augmented = to_default_device(Classifier_RESNET(
            input_dim, 
            nb_classes, 
            learning_rate=config['CLASSIFIER_LEARNING_RATE'], 
            weight_decay=config['WEIGHT_DECAY']
        ))  
        
        classifier_augmented_augmented, logs = classifier_augmented_augmented.train_classifier(train_loader_augmented_augmented, test_dataset, config, logs, name='classifier_augmented_augmented')
        
        print('#'*50 + '\n' +'Training the augmented augmented augmented classifier...')
        
        classifier_augmented_augmented_augmented = to_default_device(Classifier_RESNET(
            input_dim, 
            nb_classes, 
            learning_rate=config['CLASSIFIER_LEARNING_RATE'], 
            weight_decay=config['WEIGHT_DECAY']
        ))
        
        classifier_augmented_augmented_augmented, logs = classifier_augmented_augmented_augmented.train_classifier(train_loader_augmented_augmented_augmented, test_dataset, config, logs, name='classifier_augmented_augmented_augmented')
        
    return logs



        
        
    