import pandas as pd
import matplotlib.pyplot as plt
def csv_to_grouped_markdown(csv_file_path, md_file_path):
    # Read the CSV file into a pandas DataFrame
    df = pd.read_csv(csv_file_path)
    
    # Round the acc and f1 columns to 1 decimal place
    df['vae_best_acc'] = df['vae_best_acc'].round(2)
    df['vae_augmented_best_acc'] = df['vae_augmented_best_acc'].round(2)
    df['vae_augmented_augmented_best_acc'] = df['vae_augmented_augmented_best_acc'].round(2)
    df['vae_augmented_augmented_augmented_best_acc'] = df['vae_augmented_augmented_augmented_best_acc'].round(2)
    df['vae_best_f1'] = df['vae_best_f1'].round(2)
    df['vae_augmented_best_f1'] = df['vae_augmented_best_f1'].round(2)
    df['vae_augmented_augmented_best_f1'] = df['vae_augmented_augmented_best_f1'].round(2)
    df['vae_augmented_augmented_augmented_best_f1'] = df['vae_augmented_augmented_augmented_best_f1'].round(2)
    df['classifier_best_acc'] = df['classifier_best_acc'].round(2)
    df['classifier_augmented_best_acc'] = df['classifier_augmented_best_acc'].round(2)
    df['classifier_augmented_augmented_best_acc'] = df['classifier_augmented_augmented_best_acc'].round(2)
    df['classifier_augmented_augmented_augmented_best_acc'] = df['classifier_augmented_augmented_augmented_best_acc'].round(2)
    df['classifier_best_f1'] = df['classifier_best_f1'].round(2)
    df['classifier_augmented_best_f1'] = df['classifier_augmented_best_f1'].round(2)
    df['classifier_augmented_augmented_best_f1'] = df['classifier_augmented_augmented_best_f1'].round(2)
    df['classifier_augmented_augmented_augmented_best_f1'] = df['classifier_augmented_augmented_augmented_best_f1'].round(2)
    
    # Group columns by model type and combine them into a single column
    df['VAE acc'] = df[['vae_best_acc', 'vae_augmented_best_acc', 'vae_augmented_augmented_best_acc', 'vae_augmented_augmented_augmented_best_acc']].apply(lambda row: '/'.join(row.values.astype(str)), axis=1)
    df['VAE f1'] = df[['vae_best_f1', 'vae_augmented_best_f1', 'vae_augmented_augmented_best_f1', 'vae_augmented_augmented_augmented_best_f1']].apply(lambda row: '/'.join(row.values.astype(str)), axis=1)
    df['Classifier acc'] = df[['classifier_best_acc', 'classifier_augmented_best_acc', 'classifier_augmented_augmented_best_acc', 'classifier_augmented_augmented_augmented_best_acc']].apply(lambda row: '/'.join(row.values.astype(str)), axis=1)
    df['Classifier f1'] = df[['classifier_best_f1', 'classifier_augmented_best_f1', 'classifier_augmented_augmented_best_f1', 'classifier_augmented_augmented_augmented_best_f1']].apply(lambda row: '/'.join(row.values.astype(str)), axis=1)
    
    # Select the relevant columns
    grouped_df = df[['dataset', 'num_classes', 'num_train_samples', 'num_test_samples', 'VAE acc', 'VAE f1', 'Classifier acc', 'Classifier f1']]
    
    # Convert the grouped DataFrame to a markdown table
    markdown_table = grouped_df.to_markdown(index=False)
    
    # Write the markdown table to a .md file
    with open(md_file_path, 'w') as md_file:
        md_file.write(markdown_table)
        
def plot_metrics(df):
    datasets = df['dataset'].unique()
    
    for metric in ['acc', 'f1']:
        plt.figure(figsize=(10, 6))
        
        for model in ['vae', 'classifier']:
            metric_columns = [f"{model}_best_{metric}", f"{model}_augmented_best_{metric}", f"{model}_augmented_augmented_best_{metric}", f"{model}_augmented_augmented_augmented_best_{metric}"]
            for dataset in datasets:
                subset = df[df['dataset'] == dataset]
                plt.plot(metric_columns, subset[metric_columns].values.flatten(), marker='o', label=f"{model.capitalize()} - {dataset}")
        
        plt.title(f"Evolution of {metric.upper()} during different dataset augmentations")
        plt.xlabel('Augmentation')
        plt.ylabel(f'{metric.upper()}')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"results/{metric}_plot.png")

csv_file_path = 'results/logs.csv'
md_file_path = 'results/outputfile.md'
csv_to_grouped_markdown(csv_file_path, md_file_path)
plot_metrics(pd.read_csv(csv_file_path))
print(f"Markdown table has been written to {md_file_path}")
