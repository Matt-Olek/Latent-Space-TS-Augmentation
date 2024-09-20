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
    datasets_names = open("datasets_names.txt", "r").read().split("\n")

    # Add columns for Num_Classes and Train_Test_Ratio
    catch22_df = pd.DataFrame(
        columns=["Dataset"] + catch22_columns + ["Num_Classes", "Train_Test_Ratio"]
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
                "Dataset": dataset,
                **dict(zip(catch22_columns, mean)),
                "Num_Classes": nb_classes,
                "Train_Test_Ratio": train_test_ratio,
            }
            catch22_df.loc[len(catch22_df)] = new_row
        except Exception as e:
            print(f"Error processing dataset {dataset}: {str(e)}")
            pass

    catch22_df.to_csv(csv_file_path, index=False)

else:
    catch22_df = pd.read_csv(csv_file_path)

# ------------------------ Feature Importance ------------------------ #
# VAE Augmentation
logs_df_path = "results/logs.csv"
logs_df = pd.read_csv(logs_df_path)

logs_df["Max_augmented_VAE_acc"] = (
    logs_df[
        [
            "vae_augmented_best_acc",
            "vae_augmented_augmented_best_acc",
            "vae_augmented_augmented_augmented_best_acc",
        ]
    ].max(axis=1)
    - logs_df["vae_best_acc"]
)
logs_df["Has_augmented_VAE"] = logs_df["Max_augmented_VAE_acc"] > 0
logs_df_dataset = logs_df["dataset"].unique()
X = catch22_df[catch22_df["Dataset"].isin(logs_df_dataset)].drop(columns=["Dataset"])
y = logs_df["Has_augmented_VAE"]

model = RandomForestRegressor(n_estimators=100, random_state=0, max_depth=10)
model.fit(X, y)
train_acc = model.score(X, y)
feature_importances_vae = pd.Series(
    model.feature_importances_, index=X.columns
).sort_values(ascending=False)

# Classifier Augmentation
logs_df["Max_augmented_CLASS_acc"] = (
    logs_df[
        [
            "classifier_augmented_best_acc",
            "classifier_augmented_augmented_best_acc",
            "classifier_augmented_augmented_augmented_best_acc",
        ]
    ].max(axis=1)
    - logs_df["classifier_best_acc"]
)
logs_df["Has_augmented_CLASS"] = logs_df["Max_augmented_CLASS_acc"] > 0
logs_df_dataset = logs_df["dataset"].unique()
X = catch22_df[catch22_df["Dataset"].isin(logs_df_dataset)].drop(columns=["Dataset"])
y = logs_df["Has_augmented_CLASS"]

model = RandomForestRegressor(n_estimators=1000, random_state=0, max_depth=10)
model.fit(X, y)
train_acc = model.score(X, y)
feature_importances_class = pd.Series(
    model.feature_importances_, index=X.columns
).sort_values(ascending=False)

# Create dictionary for feature IDs
feature_ids = ["F" + str(i + 1) for i in range(len(X.columns))]
feature_name_to_id = dict(zip(X.columns, feature_ids))

# Save the dictionary to a JSON file
with open("results/feature_name_to_id.json", "w") as fp:
    json.dump(feature_name_to_id, fp, indent=4)

# Save the feature importances to a CSV file
feature_importances_vae.to_csv("results/feature_importances_vae.csv")
feature_importances_class.to_csv("results/feature_importances_class.csv")

# Prepare feature names for plotting
feature_names_vae = feature_importances_vae.index.map(feature_name_to_id)
feature_names_class = feature_importances_class.index.map(feature_name_to_id)

# Plotting
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(16, 6))

# Feature importances subplot
axes[0].bar(
    range(len(feature_importances_vae)),
    feature_importances_vae,
    color="blue",
    alpha=0.5,
    label="VAE Augmentation",
    width=0.4,
)
axes[0].bar(
    range(len(feature_importances_class)),
    feature_importances_class,
    color="orange",
    alpha=0.5,
    label="Classifier Augmentation",
    width=0.4,
    align="edge",
)
axes[0].set_xticks(range(len(feature_importances_vae)))
axes[0].set_xticklabels(feature_names_vae, rotation=90)
axes[0].set_title("Feature Importances for predicting accuracy augmentation")
axes[0].set_xlabel("Features")
axes[0].set_ylabel("Importance")
axes[0].legend()

# Box plot subplot for mean and variance
catch22_df_for_plot = catch22_df.drop(columns=["Dataset"])
catch22_df_for_plot.columns = feature_ids  # Rename columns to F1, F2, etc.

sns.boxplot(data=catch22_df_for_plot, ax=axes[1], showfliers=False)
axes[1].set_title("Features values repartition among all datasets")
axes[1].set_xlabel("Features")
axes[1].set_ylabel("Values")

plt.tight_layout()
plt.savefig("results/feature_importances_and_boxplots.png")
plt.show()
