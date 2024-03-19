import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import os
import xml.etree.ElementTree as et
import matplotlib.patches as mpatches
import torch


def display_prediction(compo_querysets, predictions_par_patch, args, alpha, shot, random_number):
    """
    Display predictions for a given query set.

    Args:
        compo_querysets (dict): A dictionary containing the composition of query sets.
        predictions_par_patch (list): A list of tuples representing the predictions for each patch.
#        args (object): An object containing various arguments.
#        alpha (float): The alpha value for the predictions.                                          
#        shot (int): The number of shots for the predictions.
##        number_classes_by_query_set (list): A list of the number of classes in each query set.
###        all_features_support (list): A list of all the features for the support set.
###        all_features_query (list): A list of all the features for the query set.
###        all_patchs_query (list): A list of all the patches for the query set.
###        all_patchs_support (list): A list of all the patches for the support set.
###        all_labels_support (list): A list of all the labels for the support set.
###        all_labels_query (list): A list of all the labels for the query set.
# = needed for the file name
## = used for figures of confidence and num_classes_per_window
### used for function patch_plus_proche (trés intélligente mais on n'a pas besoin maintenant)
    Returns:
        None
    """
    predictions_folder = 'results/predictions/'+args.trainset_name
    if not os.path.exists(predictions_folder):
        os.mkdir(predictions_folder)

    ve_color = (255,69,0) 	 #orange red 
    am_color = (240,128,128) #light coral 	
    re_color = (138,43,226)  #blue violet
    an_color= (255,215,0)    #gold 
    nt_color= (50,205,50)    #lime green 
    list_classes = ['AM', 'AN', 'NT', 'RE', 'VE']
    colors_pred = {'NT':nt_color,'AN':an_color,'AM':am_color,'VE':ve_color,'RE':re_color}
    colors_pred_hex = {'NT':"#32CD32",'AN':"#FFD700",'AM':"#F08080",'VE':"#FF4500",'RE':"#8A2BE2"}
    colors_truth = {'NT':'#008000','NP':'#FFFF00','P':'#A00','AM':'#DC143C','RE':'#4B0082','VE':'#FF8C00','AN':'#FFFF00'}
    min_i = compo_querysets['min_x']-15  #add 15 as a margin (optional)
    max_i = compo_querysets['max_x']+15
    min_j = compo_querysets['min_y']-15
    max_j = compo_querysets['max_y']+15

    ############################################## load annoatations ############################################
    display_poly = True
    lx,ly = {},{}
    path="/home/eliott/Bureau/liver_FS_classification/data"
    xml_file = f'{path}/annotations/{args.trainset_name.split("_")[0]}_Annotations.xml'
    if os.path.isfile(xml_file):
        xroot = et.parse(xml_file).getroot() 
        size = int(512*(1.5**(max(args.patch_sizes))))
        for child in xroot[1][1]:
            poly = child.attrib['name']
            lx[poly] = []
            ly[poly] = []
            for point in child:
                x = (int(point.attrib['x']) / (size//2) ) - min_i - 1  
                y = (int(point.attrib['y'])/ (size//2) ) - min_j - 1 
                lx[poly].append(x)
                ly[poly].append(y)
        if len(lx)==0:
            display_poly = False
    else:
        print(f'No xml file found for {args.trainset_name}',xml_file)
        display_poly = False
                
    if args.s_use_all_train_set:
        s_where = 'alltrainset'
    else:
        s_where = 'justesupport'
    figs_name_params = f'_alpha_{alpha}_nshots_{shot}_covmatrix_{args.covariance_used}_trainset_{args.support_hospital}'
    if args.covariance_used!="sans_S":
        figs_name_params+=f"_sur_{s_where}"
    if args.select_support_nb_elements!=False:
        figs_name_params += f'_select_support_{args.select_support_nb_elements}'
    if args.predire_une_seule_classe:
        figs_name_params += '_predire_une_seule_classe'
    figs_name_params += f'_gamma_{args.gamma}_{random_number}'

    ############################################## create prediction figures & add annotations ############################################
    print("Let's display the predictions:")
    # visual_predictions = [np.zeros((max_j-min_j,max_i-min_i)) for _ in range(args.n_ways)]
    global_predictions = np.full((max_j-min_j,max_i-min_i,3),255)
    confidence_heatmap = np.zeros((max_j-min_j,max_i-min_i))
    plt.figure(figsize=(10,10))
    plt.title("Predictions_"+f'_gamma_{args.gamma}_alpha_{alpha}')
    legends={}
    for (i,j) in tqdm(predictions_par_patch):
        pred = [0 for c in range(args.n_ways)]
        for (i2,j2) in [(i,j),(i,j+1),(i+1,j),(i+1,j+1)]:    # each zone in the slide is covered by 4 patchs (because of overlap)
            if (i2,j2) in predictions_par_patch:             # but some patchs can be empty
                for c in range(args.n_ways):
                    pred[c] += predictions_par_patch[i2,j2][c]
        # for c in range(args.n_ways):
        #     visual_predictions[c][j-min_j,i-min_i]=pred[c]/sum(pred)
        freq_max = max(pred)
        class_pred = pred.index(freq_max)
        global_predictions[j-min_j,i-min_i,:] = colors_pred[list_classes[class_pred]]
        confidence_heatmap[j-min_j,i-min_i] = freq_max/sum(pred)
        legends[list_classes[class_pred]]=colors_pred_hex[list_classes[class_pred]]
    plt.imshow(global_predictions)
    annot_legends=[]
    set_annot_classes=set()
    if display_poly:
        for poly in lx:
            if 'p' not in poly and 'a' not in poly:
                if "NT" in poly:                    classe="NT"
                elif "NP" in poly or "AN" in poly:  classe="AN"
                elif "R" in poly:                   classe="RE"
                elif "AM" in poly:                  classe="AM"
                elif "VE" in poly:                  classe="VE"
                else:                               classe=None
                if classe!=None:
                    plt.plot(lx[poly], ly[poly], label=poly, c=colors_truth[classe])
                    if classe not in set_annot_classes:
                        annot_legends+=plt.plot([], [], label=classe, c=colors_truth[classe])
                        set_annot_classes.add(classe)
    legend1=plt.legend(handles=[mpatches.Patch(color=legends[key], label=key) for key in legends], loc='upper left', bbox_to_anchor=(0.,1),title="Predictions")
    plt.gca().add_artist(legend1)
    plt.legend(handles=annot_legends, loc="upper right", bbox_to_anchor=(1,1.0),title="Ground Truth")
    plt.savefig(f'{predictions_folder}/predsAnnoatations{figs_name_params}.jpeg')
    print('DONE!')