import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import os
import xml.etree.ElementTree as et
import matplotlib.patches as mpatches
import torch


def display_prediction(compo_querysets, predictions, save_dir, save_name, title):
    """
    Display predictions for a given query set.

    
    """
    # predictions_folder = 'results/predictions/'+args.trainset_name
    # if not os.path.exists(predictions_folder):
    #     os.mkdir(predictions_folder)

    T_COLOR = (255,0,0) 	 # red 
    FI_COLOR = (100, 216, 230) #light coral 	
    NE_COLOR = (138,43,226)  #blue violet
    TL_COLOR= (255,215,0)    #gold 
    H_COLOR = (255,192,203)
    P_COLOR= (50,205,50)    #lime green 

    list_classes = ['P', 'H', 'Né', 'TL', 'Fi', 'T']
    colors_pred = {0:P_COLOR, 1:H_COLOR, 2:NE_COLOR, 3:TL_COLOR, 4:FI_COLOR, 5:T_COLOR}
    idx_to_label = {'P':0, 'H':1, 'Né':2, 'TL':3, 'Fi':4, 'T':5}

    min_i = compo_querysets['min_x']-15  #add 15 as a margin (optional)
    max_i = compo_querysets['max_x']+15
    min_j = compo_querysets['min_y']-15
    max_j = compo_querysets['max_y']+15


    print("Let's display the predictions:")
    # visual_predictions = [np.zeros((max_j-min_j,max_i-min_i)) for _ in range(args.n_ways)]
    global_predictions = np.full((max_j-min_j,max_i-min_i,3),255)
    
    plt.figure(figsize=(10,10))
    #plt.title("Predictions_"+f'_gamma_{args.gamma}_alpha_{alpha}')

    legends={}
    for patch, label in tqdm(predictions.items()):
        patient, slide, _, x, _, y = patch.replace('.jpg', '').split("_")
        x, y = int(x), int(y)

        global_predictions[y-min_j,x-min_i,:] = colors_pred[int(label)]
    
    #plt.legend(handles=[mpatches.Patch(color=colors_pred[i], label=list_classes[i]) for i in range(6)], loc='upper right')
    # Create a list of patches
    patches = [mpatches.Patch(color=tuple(c/255 for c in color), label=list_classes[label]) for label, color in colors_pred.items()]
    plt.legend(handles=patches, loc='upper right')
    plt.title(title)

    plt.imshow(global_predictions)
    plt.savefig(os.path.join(save_dir, save_name))
    