# Latent Space TS Augmentation
 
 This repository aims at implementing a simple and efficient method to augment time series data using a latent space. The method is based on the idea of using a variational autoencoder (VAE) to learn a latent space representation of the time series data. The latent space is then used to generate new time series data by sampling from the latent space.

 The method is implemented using :
    - [Python 3.12.3](https://www.python.org/)
    - [PyTorch](https://pytorch.org/)
    - [NumPy](https://numpy.org/)
    - [Pandas](https://pandas.pydata.org/)

## Installation
All required packages are listed in the `requirements.txt` file. You can install them using the following command:
```bash
pip install -r requirements.txt
```

## Usage

This augmentation method leverages a VAE alongside a classification model (ResNet). The VAE, trained on the original time series data, learns a latent space representation. New time series data is generated by evaluating the latent space distribution  with Gaussian Mixture Models (GMMs) and sampling from the latent space. The generated data is then used to train a classifier, which is evaluated on the original and augmented data. ON top of that, we use a loop to retrain the VAE on the augmented data to improve the latent space representation and generate better data and so on.


---
### *VAE Architecture & Latent Space Learning*

```mermaid
flowchart LR
    A[/"Encoder"/] --> B[["Mu (μ)"]] & C[["Sigma (σ)"]]
    D((("Clustered Latent Space"))) --> E[/"Decoder"/] & F[/"Classifier"/] & H(["KL Divergence Loss"]) & I(["Clustering Loss"])
    E --> G(["Reconstruction Loss"])
    F --> J(["Classification Loss"])
    F -.-> D
    G -.-> E
    E -.-> D
    J -.-> F
    H -.-> D
    I -.-> D
    NK[["Epsilon (ε)"]] --- NC((" "))
    C --- NC
    NC --- ND((" "))
    B --- ND
    ND --> D
    D -.-> A
    style H stroke-width:2px,stroke-dasharray: 2,fill:#E1BEE7,stroke:#000000,color:#000000
    style I stroke-width:2px,stroke-dasharray: 2,fill:#E1BEE7,stroke:#000000,color:#000000
    style G stroke-width:2px,stroke-dasharray: 2,fill:#E1BEE7,stroke:#000000,color:#000000
    style J stroke-width:2px,stroke-dasharray: 2,fill:#E1BEE7,stroke:#000000,color:#000000
    style NC stroke:none,fill:#000000
    style ND fill:#000000,stroke:#000000
    linkStyle 9 stroke:#E1BEE7,fill:none
    linkStyle 10 stroke:#E1BEE7,fill:none
    linkStyle 11 stroke:#E1BEE7,fill:none
    linkStyle 12 stroke:#E1BEE7,fill:none
    linkStyle 13 stroke:#E1BEE7,fill:none
    linkStyle 14 stroke:#000000,fill:none
    linkStyle 18 stroke:#000000,fill:none
    linkStyle 19 stroke:#E1BEE7,fill:none




```
***

### *Augmentation Process*

 
```mermaid
flowchart LR
    A["Train Data"] -. Learn .-> D((("Clustered Latent Space"))) & E[/"Decoder"/]
    D --> E
    D -- Fit --> I[/"GMM"/]
    I -. Sample with extended boundaries .-> D
    E --> ns["Synthetic data"]
    ns -- Add data --> A
    style ns fill:#AA00FF,color:#FFFFFF
    linkStyle 2 stroke:#AA00FF,fill:none
    linkStyle 4 stroke:#AA00FF,fill:none
    linkStyle 5 stroke:#AA00FF,fill:none
```


The datasets used in this repository are from the [UCR Time Series Classification Archive](https://www.cs.ucr.edu/~eamonn/time_series_data_2018/). They are stored in the `data/` folder and loaded using the custom 'loader.py' script. 

Feel free to add your **own** datasets to the `data/` folder and load them using and **adapted** version of the `LSTAUG/loader.py` script.

To run the code, you can use the following command:
```bash
python LSTSAUG/main.py
```

### Configuration Parameters

The `config.py` file contains various parameters that are crucial for setting up and running the project. Below is an explanation of each parameter:

#### Data Parameters

- **`DATA_DIR`**: Specifies the directory path where the dataset is stored.
- **`RESULTS_DIR`**: Defines the directory where the results of the experiments will be saved.
- **`MODEL_DIR`**: Indicates the directory where the trained models will be stored.
- **`DATASET`**: The specific dataset to be used for training and evaluation.

#### Model Parameters

- **`SEED`**: A seed value for random number generation to ensure reproducibility.
- **`CLASSIDIER`**: The classifier model to be used for training and evaluation. (e.g., `ResNet`, `FCN` or `LSTM`).
- **`VAE_NUM_EPOCHS`**: Number of epochs for training the Variational Autoencoder (VAE).
- **`NUM_EPOCHS`**: Total number of epochs for the entire training process.
- **`BATCH_SIZE`**: The number of samples per batch during training.
- **`LATENT_DIM`**: The dimensionality of the latent space in the VAE.
- **`CLASSIFIER_LEARNING_RATE`**: The learning rate for the optimizer of the classifier.
- **`VAE_LEARNING_RATE`**: The learning rate for the optimizer of the VAE.
- **`VAE_KNN`**: The number of nearest neighbors to consider for the KNN classifier in the VAE.
- **`VAE_HIDDEN_DIM`**: The hidden dimension of the VAE.
- **`WEIGHT_DECAY`**: The weight decay (L2 regularization) value for the optimizer.
- **`SAVE_VAE`**: A boolean flag to indicate whether to save the VAE model.
- **`SAVE_CLASSIFIER`**: A boolean flag to indicate whether to save the classifier model.

#### Augmentation Parameters

- **`AUGMENT_PLOT`**: A boolean flag to indicate whether to plot the augmentations.
- **`TEST_AUGMENT`**: A boolean flag indicating whether to apply augmentations during testing.
- **`USE_TRAINED`**: A boolean flag to indicate whether to use a pre-trained model.
- **`BASELINE`**: A boolean flag indicating whether to use baseline (non-augmented) data.
- **`NUM_SAMPLES`**: The number of samples to generate for each augmentation.
- **`NOISE`**: The noise level to be added during augmentation.
- **`ALPHA`**: The alpha value for augmentation operations.

#### Weights & Biases (WandB) Parameters

- **`WANDB`**: A boolean flag to indicate whether to use Weights & Biases for experiment tracking.
- **`WANDB_PROJECT`**: The name of the Weights & Biases project.

## Results

The results of the experiments are stored in the `results/` folder. The following plots are generated during the training and evaluation process:

- **Latent Space Neigborhoods** (`results/visualization/neighbors.png`): This plot shows the latent space neighborhoods of the original and augmented data points :

![Results](assets/neighbors_example.png) 

- **Augmentation Plot** (`results/visualization/augmentation.png`): This plot shows the original and augmented time series data points :

![Results](assets/augmentation_example.png)




