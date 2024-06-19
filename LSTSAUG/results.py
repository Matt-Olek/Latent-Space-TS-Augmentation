import csv
import os

def add_data_to_csv(filename, 
                    dataset, 
                    type_, 
                    loss_mean, 
                    loss_std, 
                    accuracy_mean, 
                    accuracy_std,
                    f1_mean, 
                    f1_std, 
                    recall_mean, 
                    recall_std, 
                    nb_classes, 
                    train_size, 
                    test_size):
    
    data = [dataset, type_, loss_mean, loss_std, accuracy_mean, accuracy_std,
            f1_mean, f1_std, recall_mean, recall_std, nb_classes, train_size, test_size]

    file_exists = os.path.isfile(filename)

    with open(filename, mode='a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(['dataset', 'type', 'loss_mean', 'loss_std', 'accuracy_mean', 'accuracy_std',
                             'f1_mean', 'f1_std', 'recall_mean', 'recall_std', 'nb_classes', 'train_size', 'test_size'])

        writer.writerow(data)
