
import time
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt


def create_confusion_matrix(legend='',filename='',cf_matrix='None'):
    list_classes = ['AM', 'AN', 'NT', 'RE', 'VE']
    # list_classes = ['AM', 'AN', 'NT', 'VE']
    # list_classes = ['AN', 'NT', 'RE']
    if isinstance(cf_matrix, str) and cf_matrix=='None':
    # create confusin matrix from file average_confusion_matrix.npy
    # Path: confusion_matrix/create_confusion_matrix.py
        cf_matrix = np.load('confusion_matrix/average_confusion_matrix.npy')#[-1:]
        # print("cf_matrix.shape: ", cf_matrix.shape)
        nb_tasks=cf_matrix.size
        cf_matrix = np.sum(cf_matrix, axis=0) 
        legend+=f' ({nb_tasks} tasks) (accuracy from matrix={computed_accuracy})'
    computed_accuracy=sum([cf_matrix[i,i] for i in range(5)])/sum([sum([cf_matrix[i,j] for i in range(5)]) for j in range(5)])
    computed_accuracy=int(computed_accuracy*1000)/1000
    cf_matrix = cf_matrix #/ np.sum(cf_matrix, axis=1)[:, np.newaxis]
    # print("cf_matrix.shape: ", cf_matrix.shape)
    df_cm = pd.DataFrame(cf_matrix, index = [i for i in list_classes], columns = [i for i in list_classes])
    plt.figure(figsize = (12,7))
    sns.heatmap(df_cm, annot=True)
    if filename=='':
        filename=f'confusion_matrix_{int(time.time())}'
    plt.xlabel('PREDICTION')
    plt.ylabel('TRUE VALUE')
    if legend!='':
        plt.title(legend)
    plt.savefig('confusion_matrix/'+filename+'.png')

if __name__ == '__main__':
    create_confusion_matrix()