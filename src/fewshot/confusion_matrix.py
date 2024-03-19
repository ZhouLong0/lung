import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np


def create_confusion_matrix(y_true_test, y_preds_test, save_dir):
    idx_to_label = {'P':0, 'H':1, 'Né':2, 'TL':3, 'Fi':4, 'T':5}
    labels =  list(idx_to_label.keys())

    conf_mat_test = confusion_matrix(np.array(y_true_test) ,np.array(y_preds_test), normalize='true') 

    plt.figure(figsize=(12,4))
    plt.subplot(121)   #Test
    sns.heatmap(conf_mat_test, annot=True, fmt='0.3f',xticklabels=labels,yticklabels=labels, cmap='BuPu') 
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title('Confusion matrix - Test')

    conf_mat_test = confusion_matrix(np.array(y_true_test) ,np.array(y_preds_test)) 
    plt.subplot(122)   #Test
    sns.heatmap(conf_mat_test, annot=True, fmt='0.3f',xticklabels=labels,yticklabels=labels, cmap='BuPu') 
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title('Confusion matrix - Test')
    plt.tight_layout()
    plt.savefig(save_dir + f'confusion_matrix.jpeg')
    plt.show()
