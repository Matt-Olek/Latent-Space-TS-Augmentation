import pandas as pd
import tqdm
import numpy as np
import pycatch22
from loader import getUCRLoader 
from config import config 
import os
from sklearn.ensemble import RandomForestRegressor
import pandas as pd

import matplotlib.pyplot as plt

data_dir = config['DATA_DIR']
csv_file_path = 'results/catch22.csv'

random_ts = np.random.rand(100)
catch22 = pycatch22.catch22_all(random_ts)
catch22_columns = catch22['names']

if not os.path.exists(csv_file_path):
    datasets_names = open('datasets_names.txt', 'r').read().split('\n')

    catch22_df = pd.DataFrame(columns=['Dataset'] + catch22_columns)

    for dataset in tqdm.tqdm(datasets_names):
        catch22_list = []
        try:
            train_loader, test_dataset, nb_classes, _ = getUCRLoader(data_dir, dataset, 1,plot=False)
            for i, (x, y) in enumerate(train_loader):
                x = x.numpy()
                for ts in x:
                    catch22 = pycatch22.catch22_all(ts)['values']
                    catch22_list.append(catch22)
            catch22_list = np.array(catch22_list)
            mean = np.mean(catch22_list, axis=0)

            ##Add row to dataframe
            new_row = {'Dataset': dataset, **dict(zip(catch22_columns, mean))}
            catch22_df.loc[len(catch22_df)] = new_row
        except Exception as e:
            print(f"Error processing dataset {dataset}: {str(e)}")
            pass

    catch22_df.to_csv(csv_file_path, index=False)
    
else:
    catch22_df = pd.read_csv(csv_file_path)

# # Plotting the graph
# num_features = len(catch22_columns)
# num_rows = (num_features + 1) // 2
# fig, axes = plt.subplots(num_rows, 2, figsize=(20, 40))

# for i, feature in enumerate(catch22_columns):
#     row = i // 2
#     col = i % 2
#     catch22_df.plot(kind='bar', x='Dataset', y=feature, ax=axes[row, col], legend=False)
#     axes[row, col].set_title(feature)
#     axes[row, col].set_xticklabels([])  # Remove x-axis labels

# plt.suptitle('Catch22 features')
# plt.tight_layout()
# plt.savefig('results/catch22_plot.png')


# ------------------------ Feature Importance ------------------------ #
# VAE Augmentation
logs_df_path = 'results/logs.csv'
logs_df = pd.read_csv(logs_df_path)

logs_df['Max_augmented_VAE_acc'] = logs_df[['vae_augmented_best_acc','vae_augmented_augmented_best_acc','vae_augmented_augmented_augmented_best_acc']].max(axis=1) - logs_df['vae_best_acc']
logs_df['Has_augmented_VAE'] = logs_df['Max_augmented_VAE_acc'] > 0
logs_df_dataset = logs_df['dataset'].unique()
X = catch22_df[catch22_df['Dataset'].isin(logs_df_dataset)].drop(columns=['Dataset'])
X['Num_Classes'] = logs_df['num_classes']
X['Num_Train_Samples'] = logs_df['num_train_samples']
X['Num_Test_Samples'] = logs_df['num_test_samples']
X['Train_Test_Ratio'] = logs_df['num_train_samples'] / logs_df['num_test_samples']
print(X.head())
y = logs_df['Has_augmented_VAE']
print(X.shape, y.shape)
model = RandomForestRegressor(n_estimators=1000, random_state=0, max_depth=10)
model.fit(X, y)
train_acc = model.score(X, y)
# Get feature importances
feature_importances_vae = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)

# Classifier Augmentation
logs_df = pd.read_csv(logs_df_path)

logs_df['Max_augmented_CLASS_acc'] = logs_df[['classifier_augmented_best_acc','classifier_augmented_augmented_best_acc','classifier_augmented_augmented_augmented_best_acc']].max(axis=1) - logs_df['classifier_best_acc']
logs_df['Has_augmented_CLASS'] = logs_df['Max_augmented_CLASS_acc'] > 0
logs_df_dataset = logs_df['dataset'].unique()
X = catch22_df[catch22_df['Dataset'].isin(logs_df_dataset)].drop(columns=['Dataset'])
X['Num_Classes'] = logs_df['num_classes']
X['Num_Train_Samples'] = logs_df['num_train_samples']
X['Num_Test_Samples'] = logs_df['num_test_samples']
X['Train_Test_Ratio'] = logs_df['num_train_samples'] / logs_df['num_test_samples']
y = logs_df['Has_augmented_CLASS']
model = RandomForestRegressor(n_estimators=1000, random_state=0, max_depth=10)
model.fit(X, y)
train_acc = model.score(X, y)
# Get feature importances
feature_importances_class = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)

# Plot feature importances
plt.figure(figsize=(10, 6))
feature_importances_vae.plot(kind='bar', color='blue', alpha=0.5, label='VAE Augmentation', position=0, width=0.4)
feature_importances_class.plot(kind='bar', color='orange', alpha=0.5, label='Classifier Augmentation', position=1, width=0.4)
plt.title('Feature importances for predicting augmentation')
plt.xlabel('Features')
plt.ylabel('Importance')
plt.legend()
plt.tight_layout()
plt.savefig('results/feature_importances.png')
