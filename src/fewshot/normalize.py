from histomicstk.preprocessing.color_normalization import reinhard
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import os
from tqdm import tqdm

def norm_reinhard(args):
    """
    Normalizes the images in the dataset using the Reinhard method.

    Args:
        args (Namespace): The command line arguments.
            - root (str): The root directory of the dataset.
            - dataset_path (str): The path to the dataset.
            - dataset_path (str): The path to the dataset.
            - support_hospital (str): The name of the support hospital.
            - trainset_name (str): The name of the training set.
            - patch_sizes (list): The list of patch sizes.
            - split_dir (str): The directory of the split files.

    Returns:
        None
    """

    def ReinhardNorm(img):
        cnorm = {'mu': np.array([8.74108109, -0.12440419,  0.0444982]),
                'sigma': np.array([0.6135447, 0.10989545, 0.0286032])}
        return reinhard(img, target_mu=cnorm['mu'], target_sigma=cnorm['sigma'])

    old_folder_path = args.root + '/' + args.dataset_path + '/'
    new_folder_path = args.root + '/' + args.dataset_path + '_normalized_reinhard/'
    print("old_data_path",old_folder_path)
    print("new_data_path",new_folder_path)
    args.dataset_path=args.dataset_path+'_normalized_reinhard'

    if not os.path.exists(new_folder_path):
        os.mkdir(new_folder_path)

    if len(os.listdir(old_folder_path)) == len(os.listdir(new_folder_path)): # ie si on a déjà normalisé
        return
    
    support_split_file =  os.path.join(args.split_dir, 'support_' + args.support_hospital + '.csv')
    query_split_file = os.path.join(args.split_dir, f'query_{args.trainset_name}.csv')
    print("support_split_file",support_split_file)
    print("query_split_file",query_split_file)
    f_support = open(support_split_file, 'r') 
    f_query = open(query_split_file, 'r') 
    split = [x.strip().split(',') for x in f_support.readlines() if x.strip() != ''] 
    split+= [x.strip().split(',') for x in f_query.readlines() if x.strip() != '']

    for patch_size in args.patch_sizes:
        size = [512, 768, 1152, 1728][patch_size]
        
        print('Normalisation Reinhard des images')
        for (image_name,classe) in tqdm(split):
            filename = f'{image_name}_res{size}x{size}.jpg'
            if os.path.exists(new_folder_path+filename) == False:
                image = plt.imread(old_folder_path+filename)
                normalized_image = ReinhardNorm(image)
                im2save = Image.fromarray(normalized_image)
                im2save.save(new_folder_path+filename)