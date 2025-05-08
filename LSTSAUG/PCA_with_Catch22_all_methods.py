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

# Load the ResNet results
resnet_df = pd.read_csv("results/ResNet_all.csv", sep=";")


# Function to compute feature importances
def compute_feature_importances(method_name):
    # Load and prepare catch22 and variability data
    catch22_df = pd.read_csv("results/catch22.csv")
    variability_df = pd.read_csv("results/datasets_variability.csv")
    variability_df.drop(columns=["Variability_TEST", "Variability_TRAIN"], inplace=True)
    catch22_df = pd.merge(catch22_df, variability_df, on="dataset")

    # Calculate augmentation ratio for the specified method
    resnet_df["augmentation_ratio"] = (
        (resnet_df[method_name] - resnet_df["baseline"]) / resnet_df["baseline"]
    ).fillna(0)

    # Merge with features
    merged_df = pd.merge(
        resnet_df[["dataset", "augmentation_ratio"]],
        catch22_df,
        left_on="dataset",
        right_on="dataset",
    )

    # Prepare features and target
    features = merged_df.drop(columns=["dataset", "augmentation_ratio"])
    target = merged_df["augmentation_ratio"]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        features, target, test_size=0.1, random_state=423
    )

    # Train Random Forest model
    model = RandomForestRegressor(
        n_estimators=200,
        random_state=42,
        max_depth=5,
    )
    model.fit(X_train, y_train)

    # Get feature importances
    importances = model.feature_importances_
    feature_importances = pd.DataFrame(
        {"Feature": features.columns, "Importance": importances}
    )
    feature_importances["Method"] = method_name
    return feature_importances


# Compute feature importances for each augmentation method
methods = ["FAA", "LA", "TTSGAN", "Time-DDPM", "VaDE", "ASCENSION"]
feature_importances_list = []

for method in methods:
    feature_importances = compute_feature_importances(method)
    feature_importances_list.append(feature_importances)

# Combine feature importances from different methods
combined_feature_importances = pd.concat(feature_importances_list)

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
with open("feature_mapping_resnet.json", "w") as f:
    json.dump(feature_mapping, f, indent=4)

# Set the style of the visualization
sns.set(style="whitegrid")

# Plot horizontal bar chart of feature importances
plt.figure(figsize=(10, 4))
bar_plot = sns.barplot(
    data=combined_feature_importances,
    y="Importance",
    x="Feature",
    hue="Method",
    palette="viridis",
)

plt.ylabel("Importance", fontsize=14)
plt.xlabel("Feature", fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.grid(axis="x", linestyle="--", alpha=0.7)
plt.legend(title="Method", fontsize=12, title_fontsize="13")
plt.tight_layout()
plt.savefig("combined_feature_importance_resnet.png", bbox_inches="tight")
plt.show()
