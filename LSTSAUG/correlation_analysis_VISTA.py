import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
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

# Calculate correlations for each step
correlations_steps = []
for step in range(6):
    # Calculate augmentation ratio for this step
    augmentation_ratio = (
        (
            merged_df[f"classifier_augmented_step_{step}_best_acc"]
            - merged_df["classifier_best_acc"]
        )
        / merged_df["classifier_best_acc"]
    ).fillna(0)

    # Calculate correlations
    correlations = {}
    for feature in feature_columns:
        correlation = np.corrcoef(merged_df[feature], augmentation_ratio)[0, 1]
        correlations[feature] = correlation

    correlations_steps.append(correlations)

# Create DataFrame with correlations
correlation_df = pd.DataFrame(correlations_steps, index=[f"Step {i}" for i in range(6)])

# Load feature mapping
with open("feature_mapping_VISTA_steps.json", "r") as f:
    feature_mapping = json.load(f)

# Rename columns using feature mapping
correlation_df.columns = [feature_mapping[col] for col in correlation_df.columns]

# Create a sign matrix (-1, 0, 1)
sign_matrix = np.zeros_like(correlation_df.values, dtype=int)
sign_matrix[correlation_df > 0.1] = 1
sign_matrix[correlation_df < -0.1] = -1

sign_df = pd.DataFrame(
    sign_matrix, index=correlation_df.index, columns=correlation_df.columns
)

# Plot correlation signs
plt.figure(figsize=(12, 4))
sns.heatmap(
    sign_df,
    cmap="RdBu",
    center=0,
    vmin=-1,
    vmax=1,
    cbar_kws={"label": "Correlation Sign"},
    xticklabels=True,
    yticklabels=True,
)

plt.title("Feature Correlation Signs with Augmentation Success")
plt.xlabel("Features")
plt.ylabel("Steps")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig("VISTA_correlation_signs.png", dpi=300, bbox_inches="tight")
plt.show()

# Save correlation values to CSV
correlation_df.to_csv("feature_correlations_VISTA.csv")

# Print summary of consistent correlations
print("\nFeatures with consistent correlation signs across steps:")
for feature in correlation_df.columns:
    signs = sign_df[feature].unique()
    if len(signs) == 1:
        sign_str = (
            "positive" if signs[0] == 1 else "negative" if signs[0] == -1 else "neutral"
        )
        print(f"{feature}: Consistently {sign_str}")

# Print strongest correlations
print("\nStrongest correlations (absolute value > 0.3):")
for step in correlation_df.index:
    strong_correlations = correlation_df.loc[step][abs(correlation_df.loc[step]) > 0.3]
    if not strong_correlations.empty:
        print(f"\n{step}:")
        for feature, corr in strong_correlations.items():
            print(f"{feature}: {corr:.3f}")
