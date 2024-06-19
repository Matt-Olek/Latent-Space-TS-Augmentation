import pandas as pd
import numpy as np

# Lecture du CSV sous forme de DataFrame
data = pd.read_csv('results/logs.csv')

df = pd.DataFrame(data)

# Fonction pour calculer moyenne et variance et formater les résultats
def mean_var_format(series):
    return f"{series.mean():.4f} ({series.var():.4f})"

# Groupement des données par 'dataset' et calcul des moyennes et variances pour chaque colonne
grouped = df.groupby('dataset').agg({
    'num_classes': 'first',  # Assure que nous avons le nombre de classes
    'num_train_samples': 'first',  # Assure que nous avons le nombre d'échantillons d'entraînement
    'augmented_train_samples': 'first',  # Assure que nous avons le nombre d'échantillons d'entraînement augmentés
    'baseline_final_acc': mean_var_format,
    'baseline_best_acc': mean_var_format,
    'baseline_final_f1': mean_var_format,
    'baseline_best_f1': mean_var_format,
    'augmented_final_acc': mean_var_format,
    'augmented_best_acc': mean_var_format,
    'augmented_final_f1': mean_var_format,
    'augmented_best_f1': mean_var_format
})

# Création du tableau Markdown pour les meilleures valeurs
markdown_table_best = "| Dataset | # Classes | # Train/Aug. Samples | 🔬 Baseline Best Acc (var) | 📚 Augmented Best Acc (var) | 🔬 Baseline Best F1 (var) | 📚 Augmented Best F1 (var) |\n"
markdown_table_best += "|---------|-----------|----------------------|------------------------|---------------------------|------------------------|---------------------------|\n"

# Création du tableau Markdown pour les valeurs finales
markdown_table_final = "| Dataset | # Classes | # Train/Aug. Samples | 🔬 Baseline Final Acc (var) | 📚 Augmented Final Acc (var) | 🔬 Baseline Final F1 (var) | 📚 Augmented Final F1 (var) |\n"
markdown_table_final += "|---------|-----------|----------------------|-------------------------|----------------------------|-------------------------|----------------------------|\n"

for dataset, row in grouped.iterrows():
    is_best_acc_improved = row['baseline_best_acc'].split()[0] <= row['augmented_best_acc'].split()[0]
    is_best_f1_improved = row['baseline_best_f1'].split()[0] <= row['augmented_best_f1'].split()[0]
    is_final_acc_improved = row['baseline_final_acc'].split()[0] <= row['augmented_final_acc'].split()[0]
    is_final_f1_improved = row['baseline_final_f1'].split()[0] <= row['augmented_final_f1'].split()[0]
    
    is_best_acc_improved = "✅" if is_best_acc_improved else "❌"
    is_best_f1_improved = "✅" if is_best_f1_improved else "❌"
    is_final_acc_improved = "✅" if is_final_acc_improved else "❌"
    is_final_f1_improved = "✅" if is_final_f1_improved else "❌"
    
    markdown_table_best += f"| {dataset} | {row['num_classes']} | {row['num_train_samples']} / {row['augmented_train_samples']} | {row['baseline_best_acc']} | {row['augmented_best_acc']} {is_best_acc_improved} | {row['baseline_best_f1']} | {row['augmented_best_f1']} {is_best_f1_improved} |\n"
    markdown_table_final += f"| {dataset} | {row['num_classes']} | {row['num_train_samples']} / {row['augmented_train_samples']} | {row['baseline_final_acc']} | {row['augmented_final_acc']} {is_final_acc_improved} | {row['baseline_final_f1']} | {row['augmented_final_f1']} {is_final_f1_improved} |\n"

# Sauvegarde des tableaux Markdown dans le même fichier avec des titres
with open('results/table.md', 'w') as f:
    f.write("# Best Results\n")
    f.write(markdown_table_best)
    
    f.write("# Final Results\n")
    f.write(markdown_table_final)
