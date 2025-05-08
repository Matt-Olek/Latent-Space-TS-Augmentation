import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import json

# Load the data
logs_df_vista = pd.read_csv("results/VISTA/logs.csv")
catch22_df = pd.read_csv("results/catch22.csv")
variability_df = pd.read_csv("results/datasets_variability.csv")

# Prepare the data
variability_df.drop(columns=["Variability_TEST", "Variability_TRAIN"], inplace=True)
catch22_df = pd.merge(catch22_df, variability_df, on="dataset")
merged_df = pd.merge(logs_df_vista, catch22_df, on="dataset")

# Prepare features
feature_columns = catch22_df.columns.tolist()
feature_columns.remove("dataset")
# feature_columns.remove("train_test_ratio")

# Get feature importances for each step
feature_importances_steps = []
step_names = []
for step in range(6):
    # Calculate augmentation ratio for this step
    augmentation_ratio = (
        (
            merged_df[f"classifier_augmented_step_{step}_best_acc"]
            - merged_df["classifier_best_acc"]
        )
        / merged_df["classifier_best_acc"]
    ).fillna(0)

    # Prepare features and target
    X = merged_df[feature_columns]
    y = augmentation_ratio

    # Train Random Forest model
    model = RandomForestRegressor(
        n_estimators=100, random_state=42, max_depth=5, min_samples_split=2
    )
    model.fit(X, y)

    # Store feature importances
    importances = model.feature_importances_
    feature_importances_steps.append(importances)
    step_names.append(f"Step {step}")

# Convert to numpy array for PCA
feature_importance_matrix = np.array(feature_importances_steps)

# Apply PCA to feature importances
pca = PCA(n_components=2)
importance_pca = pca.fit_transform(feature_importance_matrix)

# Create DataFrame for plotting
feature_importances_df = pd.DataFrame(
    feature_importance_matrix, columns=feature_columns, index=step_names
)

# Generate mapping of feature names to F1, F2, etc.
feature_mapping = {name: f"F{i+1}" for i, name in enumerate(feature_columns)}

# Save the feature mapping to a JSON file
with open("feature_mapping_VISTA_steps.json", "w") as f:
    json.dump(feature_mapping, f, indent=4)

# Update feature names in the DataFrame
feature_importances_df.columns = [
    feature_mapping[col] for col in feature_importances_df.columns
]

# Reshape data for plotting
melted_importances = feature_importances_df.reset_index()
melted_importances = pd.melt(
    melted_importances, id_vars="index", var_name="Feature", value_name="Importance"
)
melted_importances = melted_importances.rename(columns={"index": "Step"})

# Set the style of the visualization
sns.set_theme(style="whitegrid")

# Plot horizontal bar chart of feature importances for all steps
plt.figure(figsize=(10, 4))
bar_plot = sns.barplot(
    data=melted_importances,
    y="Importance",
    x="Feature",
    hue="Step",
    palette="viridis",
)

plt.ylabel("Importance", fontsize=14)
plt.xlabel("Feature", fontsize=14)
plt.xticks(fontsize=12, rotation=45, ha="right")
plt.yticks(fontsize=12)
plt.grid(axis="x", linestyle="--", alpha=0.7)
plt.legend(title="Step", fontsize=12, title_fontsize="13")
plt.tight_layout()

plt.savefig("VISTA_feature_importances_bar.png", bbox_inches="tight", dpi=300)
plt.show()

# Print explained variance ratios
print("\nExplained variance ratios:")
for i, ratio in enumerate(pca.explained_variance_ratio_):
    print(f"PC{i+1}: {ratio:.2%}")

# Print feature contributions
feature_weights = pd.DataFrame(
    pca.components_.T, columns=["PC1", "PC2"], index=feature_columns
)
print("\nTop feature contributions:")
print("\nPC1 contributions:")
print(feature_weights["PC1"].sort_values(ascending=False).head())
print("\nPC2 contributions:")
print(feature_weights["PC2"].sort_values(ascending=False).head())
