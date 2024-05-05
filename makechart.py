import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

data = pd.read_excel('data.xlsx', index_col=0)

data = data[['ViT', 'ResNet']]

def plot_metric_with_error(metric, error_metric, title, ylim=None):
    fig, ax = plt.subplots()
    models = data.columns 
    values = data.loc[metric].values 
    errors = data.loc[error_metric].values
    
    errors = np.minimum(errors, values)

    bars = ax.bar(models, values, yerr=errors, capsize=5, color='skyblue', label=metric)
    ax.set_xlabel('Models')
    ax.set_ylabel('Scores')
    ax.set_title(title)
    ax.set_xticks(np.arange(len(models)))  
    ax.set_xticklabels(models, rotation=45)
    ax.legend()

    if ylim:
        ax.set_ylim(ylim)

    plt.tight_layout()
    plt.show()

plot_metric_with_error('test_auc', 'test_auc_std', 'Comparison of Test AUC')
plot_metric_with_error('test_acc', 'test_acc_std', 'Comparison of Test Accuracy')
plot_metric_with_error('test_f1', 'test_f1_std', 'Comparison of Test F1 Score', ylim=(0, 1))