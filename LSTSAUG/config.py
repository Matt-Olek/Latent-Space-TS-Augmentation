config = {
    
    # ---------------- DATA PARAMETERS---------------- #
    
    "DATA_DIR": '../../FastAutoAugment-Time-Series/data',
    "RESULTS_DIR": 'results',
    "MODEL_DIR": 'models',
    "DATASET": 'ChlorineConcentration',
    
    # ---------------- MODEL PARAMETERS ---------------- #
    
    "SEED": 423,
    "VAE_NUM_EPOCHS": 1500,
    "NUM_EPOCHS": 3000,
    "BATCH_SIZE": 8,
    "LATENT_DIM": 4,
    "LEARNING_RATE": 0.00001,
    "CLASSIFIER_LEARNING_RATE": 0.00001,
    "VAE_LEARNING_RATE": 1e-3,
    "WEIGHT_DECAY": 0.000001,
    
    # ---------------- AUGMENTATION PARAMETERS ---------------- #
    
    "AUGMENT_PLOT": True,
    "TEST_AUGMENT": True,
    "USE_TRAINED": False,
    "BASELINE": True,
    "NUM_SAMPLES": 1,
    "NOISE": 0.01,
    "ALPHA": 1,
    
    # ---------------- WANDB PARAMETERS ---------------- #
    
    "WANDB": True,  
    "WANDB_PROJECT": 'lstsaug_2.0',
}
