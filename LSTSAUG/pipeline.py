import os
import torch
from utils import get_model_path
from VAE import VAE
from ClassifierModel import Classifier_RESNET
from ClassifierModelFCN import Classifier_FCN
from config import config
from loader import getUCRLoader, augment_loader, simple_augment_loader
from utils import to_default_device
import time
from visualization import plot_latent_space_viz


def pipeline(config=config):
    logs = {}
    print("Starting script for dataset : {}".format(config["DATASET"]))
    torch.manual_seed(config["SEED"])
    os.makedirs(config["MODEL_DIR"], exist_ok=True)
    os.makedirs(config["RESULTS_DIR"], exist_ok=True)

    train_loader, test_dataset, nb_classes, scaler = getUCRLoader(
        config["DATA_DIR"], config["DATASET"], config["BATCH_SIZE"]
    )

    input_dim = train_loader.dataset[0][0].shape[0]
    num_samples = int(len(train_loader.dataset) / nb_classes) * config["NUM_SAMPLES"]

    print("Detected input dimension: ", input_dim)
    print("Number of samples produced per class: ", num_samples)

    logs["dataset"] = config["DATASET"]
    logs["num_classes"] = nb_classes
    logs["num_train_samples"] = len(train_loader.dataset)
    logs["num_test_samples"] = len(test_dataset[0])

    # ---------------------------- VAE Training ---------------------------- #

    start_time = time.time()

    vae = to_default_device(
        VAE(
            input_dim,
            nb_classes,
            latent_dim=config["LATENT_DIM"],
            learning_rate=config["VAE_LEARNING_RATE"],
            hidden_dim=config["VAE_HIDDEN_DIM"],
            knn=min(config["VAE_KNN"], len(train_loader.dataset) // 2),
        )
    )

    model_path = get_model_path(config, name="vae")
    if not config["USE_TRAINED"] or not os.path.exists(model_path):
        print("#" * 50 + "\n" + "Training the VAE model...")
        vae, logs = vae.train_vae(train_loader, test_dataset, nb_classes, config, logs)

        # print("#" * 50 + "\n" + "Training the classifier on the original data...")

        # if config["CLASSIFIER"] == "FCN":
        #     classifier = to_default_device(
        #         Classifier_FCN(
        #             input_dim,
        #             nb_classes,
        #             learning_rate=config["CLASSIFIER_LEARNING_RATE"],
        #             weight_decay=config["WEIGHT_DECAY"],
        #         )
        #     )

        # else:
        #     classifier = to_default_device(
        #         Classifier_RESNET(
        #             input_dim,
        #             nb_classes,
        #             learning_rate=config["CLASSIFIER_LEARNING_RATE"],
        #             weight_decay=config["WEIGHT_DECAY"],
        #         )
        #     )

        # classifier, logs = classifier.train_classifier(
        #     train_loader, test_dataset, config, logs, name="classifier"
        # )
    else:
        vae.load_state_dict(torch.load(model_path))
        vae.eval()
        print(f"Model loaded from {model_path}")

    plot_latent_space_viz(
        vae,
        train_loader,
        test_dataset,
        num_classes=nb_classes,
        type="3d",
        id=f"dataset_{config['DATASET']}_original",
    )
    # ---------------------------- VAE and Classifier Augmentation Steps ---------------------------- #

    augmented_train_loader = train_loader
    vae_current = vae
    augmentation_step = 0

    while augmentation_step <= config["MAX_AUGMENTATION_STEPS"]:
        # Perform augmentation
        augmented_train_loader = augment_loader(
            # augmented_train_loader = simple_augment_loader(
            augmented_train_loader,
            vae_current,
            num_samples=num_samples,
            scaler=scaler,
            num_classes=nb_classes,
            alpha=config["ALPHA"],
        )
        # plot_latent_space_viz(
        #     vae_current,
        #     augmented_train_loader,
        #     test_dataset,
        #     num_classes=nb_classes,
        #     type="2d",
        #     id=f"dataset_{config['DATASET']}_before_augmented_step_{augmentation_step}",
        # )
        print(
            "#" * 50 + f"\nAugmented dataset size (step {augmentation_step}): ",
            len(augmented_train_loader.dataset),
        )

        if config["TEST_AUGMENT"]:
            print(
                "#" * 50
                + f"\nTraining the VAE model on augmented data (step {augmentation_step})..."
            )
            vae_current = to_default_device(
                VAE(
                    input_dim,
                    nb_classes,
                    latent_dim=config["LATENT_DIM"],
                    learning_rate=config["VAE_LEARNING_RATE"],
                    hidden_dim=config["VAE_HIDDEN_DIM"],
                    knn=min(
                        config["VAE_KNN"], len(augmented_train_loader.dataset) // 2
                    ),
                )
            )

            vae_current, logs = vae_current.train_vae(
                augmented_train_loader,
                test_dataset,
                nb_classes,
                config,
                logs,
                name=f"vae_augmented_step_{augmentation_step}",
            )

            plot_latent_space_viz(
                vae_current,
                augmented_train_loader,
                test_dataset,
                num_classes=nb_classes,
                type="3d",
                id=f"dataset_{config['DATASET']}_augmented_step_{augmentation_step}",
            )

        # # ---------------------------- Classifier Training ---------------------------- #

        # print(
        #     "#" * 50
        #     + f"\nTraining the classifier on augmented data (step {augmentation_step})..."
        # )
        # if config["CLASSIFIER"] == "FCN":
        #     classifier_current = to_default_device(
        #         Classifier_FCN(
        #             input_dim,
        #             nb_classes,
        #             learning_rate=config["CLASSIFIER_LEARNING_RATE"],
        #             weight_decay=config["WEIGHT_DECAY"],
        #         )
        #     )
        # else:
        #     classifier_current = to_default_device(
        #         Classifier_RESNET(
        #             input_dim,
        #             nb_classes,
        #             learning_rate=config["CLASSIFIER_LEARNING_RATE"],
        #             weight_decay=config["WEIGHT_DECAY"],
        #         )
        #     )

        # classifier_current, logs = classifier_current.train_classifier(
        #     augmented_train_loader,
        #     test_dataset,
        #     config,
        #     logs,
        #     name=f"classifier_augmented_step_{augmentation_step}",
        # )

        augmentation_step += 1

    end_time = time.time()
    logs["execution_time"] = end_time - start_time

    return logs
