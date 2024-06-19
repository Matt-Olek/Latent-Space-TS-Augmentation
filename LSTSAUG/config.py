config = {
    
    # ---------------- DATA ---------------- #
    
    "DATA_DIR": '../../FastAutoAugment-Time-Series/data',
    "RESULTS_DIR": 'results',
    "MODEL_DIR": 'models',
    "DATASET": 'HandOutlines',
    
    # ---------------- MODEL ---------------- #
    
    "SEED": 423,
    "VAE_NUM_EPOCHS": 2000,
    "NUM_EPOCHS": 3000,
    "BATCH_SIZE": 8,
    "LATENT_DIM": 500,
    "LEARNING_RATE": 0.0001,
    "WEIGHT_DECAY": 0.000001,
    
    # ---------------- AUGMENTATION ---------------- #
    
    "AUGMENT_PLOT": False,
    "TEST_AUGMENT": True,
    "USE_TRAINED": True,
    "BASELINE": False,
    "NUM_SAMPLES": 1,
    "NOISE": 0.01,
    "ALPHA": 0.0001,
    "INCREMENTAL_AUG": False,
    
    # ---------------- WANDB ---------------- #
    
    "WANDB": False,  
}
