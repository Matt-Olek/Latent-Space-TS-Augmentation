config = {
    
    # ---------------- DATA PARAMETERS---------------- #
    
    "DATA_DIR": '../../FastAutoAugment-Time-Series/data',
    "RESULTS_DIR": 'results',
    "MODEL_DIR": 'models',
    "DATASET": 'Adiac',
    
    # ---------------- MODEL PARAMETERS ---------------- #
    
    "SEED": 423,
    "VAE_NUM_EPOCHS": 1500,
    "NUM_EPOCHS": 3000,
    "BATCH_SIZE": 8,
    "LATENT_DIM": 7,
    "LEARNING_RATE": 0.00001,
    "CLASSIFIER_LEARNING_RATE": 0.0001,
    "VAE_LEARNING_RATE": 1e-4,
    "EARLY_STOP_PATIENCE": 150,
    "VAE_KNN": 1,
    "VAE_HIDDEN_DIM": 1000,
    "WEIGHT_DECAY": 0.000001,
    'SAVE_VAE': True,
    
    # ---------------- AUGMENTATION PARAMETERS ---------------- #
    
    "AUGMENT_PLOT": False,
    "TEST_AUGMENT": True,
    "USE_TRAINED": False,
    "BASELINE": True,
    "NUM_SAMPLES": 2,
    "NOISE": 0.001,
    "ALPHA": 1,
    
    # ---------------- WANDB PARAMETERS ---------------- #
    
    "WANDB": True,  
    "WANDB_PROJECT": 'lstsaug_reforged',
}
