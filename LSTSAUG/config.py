config = {
    # ---------------- DATA PARAMETERS---------------- #
    "DATA_DIR": "./data/",
    "RESULTS_DIR": "results_hp/latentAug",
    "RESULTS_DIR_ROOT": "results_hp/latentAug",
    "MODEL_DIR": "models",
    "DATASET": "MedicalImages",
    # ---------------- MODEL PARAMETERS ---------------- #
    "SEED": 423,
    "CLASSIFIER": "ResNet",
    "VAE_NUM_EPOCHS": 500,
    "NUM_EPOCHS": 3000,
    "BATCH_SIZE": 8,
    "LATENT_DIM": 50,
    # "LEARNING_RATE": 0.00001,
    "CLASSIFIER_LEARNING_RATE": 0.0001,
    "VAE_LEARNING_RATE": 1e-3,
    "EARLY_STOP_PATIENCE": 150,
    "VAE_KNN": 100,
    "VAE_HIDDEN_DIM": 1000,
    "WEIGHT_DECAY": 1e-6,
    "SAVE_VAE": False,
    "SAVE_CLASSIFIER": False,
    # ---------------- LOSS PARAMETERS ---------------- #
    "RECON_WEIGHT": 1,
    "KL_WEIGHT": 1,
    "CLASSIFIER_WEIGHT": 1,
    "CONTRASTIVE_WEIGHT": 1,
    # ---------------- AUGMENTATION PARAMETERS ---------------- #
    "MAX_AUGMENTATION_STEPS": 3,
    "AUGMENT_PLOT": False,
    "TEST_AUGMENT": True,
    "USE_TRAINED": False,
    "BASELINE": True,
    "NUM_SAMPLES": 1,
    "NOISE": 0.01,
    "ALPHA": 1.5,
    # ---------------- WANDB PARAMETERS ---------------- #
    "WANDB": False,
    "WANDB_PROJECT": "LATENT_AUG-UCR",
}
