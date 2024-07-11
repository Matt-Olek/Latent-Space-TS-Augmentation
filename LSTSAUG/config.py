config = {
    
    # ---------------- DATA PARAMETERS---------------- #
    
    "DATA_DIR": '../FastAutoAugment-Time-Series/data',
    "RESULTS_DIR": 'results',
    "MODEL_DIR": 'models',
    "DATASET": 'MedicalImages',
    
    # ---------------- MODEL PARAMETERS ---------------- #
    
    "SEED": 423,
    "VAE_NUM_EPOCHS": 300,
    "NUM_EPOCHS": 3000,
    "BATCH_SIZE": 8,
    "LATENT_DIM": 4,
    "LEARNING_RATE": 0.00001,
    "CLASSIFIER_LEARNING_RATE": 0.0001,
    "VAE_LEARNING_RATE": 1e-3,
    "EARLY_STOP_PATIENCE": 20,
    "VAE_KNN": 100,
    "VAE_HIDDEN_DIM": 1000,
    "WEIGHT_DECAY": 0.000001,
    'SAVE_VAE': False,
    'SAVE_CLASSIFIER': False,
    
    # ---------------- AUGMENTATION PARAMETERS ---------------- #
    
    "AUGMENT_PLOT": True,
    "TEST_AUGMENT": True,
    "USE_TRAINED": False,
    "BASELINE": True,
    "NUM_SAMPLES": 1,
    "NOISE": 0.01,
    "ALPHA": 0.1,
    
    # ---------------- WANDB PARAMETERS ---------------- #
    
    "WANDB": True,  
    "WANDB_PROJECT": 'lstsaug_CIFAR10',
}
