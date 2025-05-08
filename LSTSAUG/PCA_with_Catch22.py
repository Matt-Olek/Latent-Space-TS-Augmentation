import pandas as pd
import tqdm
import numpy as np
import pycatch22
from loader import getUCRLoader
from config import config
import os
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import seaborn as sns
import json  # Import json to save the dictionary

data_dir = config["DATA_DIR"]
csv_file_path = "results/catch22.csv"

random_ts = np.random.rand(100)
catch22 = pycatch22.catch22_all(random_ts)
catch22_columns = catch22["names"]

if not os.path.exists(csv_file_path):
    datasets_names = open("data/datasets_names.txt", "r").read().split("\n")

    # Add columns for Num_Classes and Train_Test_Ratio
    catch22_df = pd.DataFrame(
        columns=["dataset"] + catch22_columns + ["train_test_ratio"]
    )

    for dataset in tqdm.tqdm(datasets_names):
        catch22_list = []
        try:
            train_loader, test_dataset, nb_classes, _ = getUCRLoader(
                data_dir, dataset, 1, plot=False
            )
            train_samples = 0
            test_samples = 0
            for i in test_dataset:
                test_samples = len(i)
                # print(test_samples)
            print(test_samples)

            for i, (x, y) in enumerate(train_loader):
                x = x.numpy()
                train_samples += len(x)
                for ts in x:
                    catch22 = pycatch22.catch22_all(ts)["values"]
                    catch22_list.append(catch22)

            catch22_list = np.array(catch22_list)
            mean = np.mean(catch22_list, axis=0)
            train_test_ratio = train_samples / test_samples

            # Add row to dataframe
            new_row = {
                "dataset": dataset,
                **dict(zip(catch22_columns, mean)),
                "train_test_ratio": train_test_ratio,
            }
            catch22_df.loc[len(catch22_df)] = new_row
        except Exception as e:
            print(f"Error processing dataset {dataset}: {str(e)}")
            pass

    catch22_df.to_csv(csv_file_path, index=False)

else:
    catch22_df = pd.read_csv(csv_file_path)

# ------------------------ Feature Importance ------------------------ #
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import json


# Load the logs datasets for LA and VISTA
logs_df_la = pd.read_csv("results/LA/logs.csv")
logs_df_la.drop(columns=["num_classes"], inplace=True)
logs_df_vista = pd.read_csv("results/VISTA/logs.csv")
logs_df_faa = pd.read_csv("results/FAA/logs.csv")


# Function to compute feature importances
def compute_feature_importances(logs_df, dataset_name):
    # Merge the data on the 'Dataset' column
    if dataset_name == "LA":
        catch22_df = pd.read_csv("results/catch22.csv")
        variability_df = pd.read_csv("results/datasets_variability.csv")
        variability_df.drop(
            columns=["Variability_TEST", "Variability_TRAIN"], inplace=True
        )
        catch22_df = pd.merge(catch22_df, variability_df, on="dataset")
        merged_df = pd.merge(logs_df, catch22_df, on="dataset")

        # Define augmentation_ratio for LA
        merged_df["augmentation_ratio"] = (
            (
                merged_df["classifier_augmented_best_acc"]
                - merged_df["classifier_best_acc"]
            )
            / merged_df["classifier_best_acc"]
        ).fillna(0)

        features = merged_df.drop(
            columns=[
                "dataset",
                "augmentation_ratio",
                "num_train_samples",
                "num_test_samples",
                "vae_best_acc",
                "vae_best_f1",
                "classifier_best_acc",
                "classifier_best_f1",
                "classifier_augmented_best_acc",
                "classifier_augmented_best_f1",
                "execution_time",
            ]
        )
    elif dataset_name == "ASCENSION":
        catch22_df = pd.read_csv("results/catch22.csv")
        variability_df = pd.read_csv("results/datasets_variability.csv")
        variability_df.drop(
            columns=["Variability_TEST", "Variability_TRAIN"], inplace=True
        )
        catch22_df = pd.merge(catch22_df, variability_df, on="dataset")

        merged_df = pd.merge(logs_df, catch22_df, on="dataset")
        merged_df["augmentation_ratio"] = (
            (
                merged_df[
                    [
                        "classifier_augmented_step_0_best_acc",
                        "classifier_augmented_step_1_best_acc",
                        "classifier_augmented_step_2_best_acc",
                        "classifier_augmented_step_3_best_acc",
                        "classifier_augmented_step_4_best_acc",
                        "classifier_augmented_step_5_best_acc",
                    ]
                ].max(
                    axis=1
                )  # Compute the max along rows (axis=1)
                - merged_df["classifier_best_acc"]
            )
            / merged_df["classifier_best_acc"]
        ).fillna(0)

        features = merged_df.drop(
            columns=[
                "dataset",
                "num_classes",
                "num_train_samples",
                "num_test_samples",
                "vae_best_acc",
                "vae_best_f1",
                "classifier_best_acc",
                "classifier_best_f1",
                "execution_time",
                "vae_augmented_step_0_best_acc",
                "vae_augmented_step_0_best_f1",
                "vae_augmented_step_1_best_acc",
                "vae_augmented_step_1_best_f1",
                "vae_augmented_step_2_best_acc",
                "vae_augmented_step_2_best_f1",
                "vae_augmented_step_3_best_acc",
                "vae_augmented_step_3_best_f1",
                "vae_augmented_step_4_best_acc",
                "vae_augmented_step_4_best_f1",
                "vae_augmented_step_5_best_acc",
                "vae_augmented_step_5_best_f1",
                "classifier_augmented_step_0_best_acc",
                "classifier_augmented_step_0_best_f1",
                "classifier_augmented_step_1_best_acc",
                "classifier_augmented_step_1_best_f1",
                "classifier_augmented_step_2_best_acc",
                "classifier_augmented_step_2_best_f1",
                "classifier_augmented_step_3_best_acc",
                "classifier_augmented_step_3_best_f1",
                "classifier_augmented_step_4_best_acc",
                "classifier_augmented_step_4_best_f1",
                "classifier_augmented_step_5_best_acc",
                "classifier_augmented_step_5_best_f1",
                "augmentation_ratio",
            ]
        )
    else:
        catch22_df = pd.read_csv("results/catch22.csv")
        variability_df = pd.read_csv("results/datasets_variability.csv")
        variability_df.drop(
            columns=["Variability_TEST", "Variability_TRAIN"], inplace=True
        )
        catch22_df = pd.merge(catch22_df, variability_df, on="dataset")
        merged_df = pd.merge(logs_df, catch22_df, on="dataset")
        merged_df["augmentation_ratio"] = (
            (merged_df["accuracy_mean_augmented"] - merged_df["accuracy_mean_baseline"])
            / merged_df["accuracy_mean_baseline"]
        ).fillna(0)

        features = merged_df.drop(
            columns=[
                "dataset",
                "accuracy_mean_baseline",
                "accuracy_mean_augmented",
                "augmentation_ratio",
            ]
        )

    # Target variable
    target = merged_df["augmentation_ratio"]
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        features, target, test_size=0.1, random_state=423
    )

    # Train Random Forest model
    model = RandomForestRegressor(n_estimators=1000, random_state=42)
    model.fit(X_train, y_train)

    # Get feature importances
    importances = model.feature_importances_
    feature_importances = pd.DataFrame(
        {"Feature": features.columns, "Importance": importances}
    )
    feature_importances["Dataset"] = dataset_name  # Add dataset name for grouping
    return feature_importances


# Compute feature importances for both datasets
feature_importances_la = compute_feature_importances(logs_df_la, "LA")
feature_importances_vista = compute_feature_importances(logs_df_vista, "ASCENSION")
feature_importances_faa = compute_feature_importances(logs_df_faa, "FAA")

# Combine feature importances from different datasets
combined_feature_importances = pd.concat(
    [feature_importances_la, feature_importances_faa, feature_importances_vista]
)

# Generate mapping of feature names to F1, F2, etc.
feature_mapping = {
    name: f"F{i+1}"
    for i, name in enumerate(combined_feature_importances["Feature"].unique())
}

# Update feature names in the combined feature importances DataFrame
combined_feature_importances["Feature"] = combined_feature_importances["Feature"].map(
    feature_mapping
)

# Save the feature mapping to a JSON file
with open("feature_mapping.json", "w") as f:
    json.dump(feature_mapping, f, indent=4)

# Set the style of the visualization
sns.set(style="whitegrid")

# Plot horizontal bar chart of feature importances for both datasets
plt.figure(figsize=(10, 4))
bar_plot = sns.barplot(
    data=combined_feature_importances,
    y="Importance",
    x="Feature",
    hue="Dataset",
    palette="viridis",
)

# Add data labels to the bars
# for p in bar_plot.patches:
#     bar_plot.annotate(
#         f"{p.get_width():.2f}",
#         (p.get_width(), p.get_y() + p.get_height() / 2),
#         ha="left",
#         va="center",
#         fontsize=8,
#         color="black",
#         xytext=(15, 0),
#         textcoords="offset points",
#     )

# plt.title(
#     "Feature Importances for Predicting Augmentation Ratio", fontsize=18, weight="bold"
# )
plt.ylabel("Importance", fontsize=14)
plt.xlabel("Feature", fontsize=14)
# use a log scale for the x-axis
# plt.xscale("log")
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.grid(axis="x", linestyle="--", alpha=0.7)
plt.legend(title="Dataset", fontsize=12, title_fontsize="13")
plt.tight_layout()  # Adjust layout to avoid clipping
plt.savefig(
    "combined_feature_importance_histogram_ICLR.png", bbox_inches="tight"
)  # Save the histogram
plt.show()
