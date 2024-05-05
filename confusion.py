import pandas as pd
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

data = pd.read_csv('vit.txt')

def plot_confusion_matrix(data, model, split='test'):
    split_data = data[data['data_split'] == split]

    cm = confusion_matrix(split_data['Y'], split_data['Y_hat'])

    labels = np.array([["True Healthy\n(Correctly not diseased)", "False Diseased\n(Incorrectly diagnosed)"],
                       ["False Healthy\n(Missed diagnosis)", "True Diseased\n(Correctly diagnosed)"]])

    annotated_matrix = np.array([["{}\n{}".format(num, desc) for num, desc in zip(cm_row, label_row)]
                                 for cm_row, label_row in zip(cm, labels)])

    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=annotated_matrix, fmt='', cmap='Blues', xticklabels=['Healthy', 'Diseased'], yticklabels=['Healthy', 'Diseased'])
    plt.xlabel('Predicted Condition')
    plt.ylabel('Actual Condition')
    plt.title(f'Confusion Matrix for {model} {split.capitalize()} Data')
    plt.show()

plot_confusion_matrix(data, 'ViT', 'test')