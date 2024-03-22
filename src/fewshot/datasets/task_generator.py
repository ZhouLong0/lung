import torch
from src.fewshot.utils import Logger
import os
from itertools import cycle
import numpy as np
import csv
from tqdm import tqdm
import os
import pickle
import re

class Tasks_Generator:
    def __init__(self, n_ways, shots, support_features_dict, query_features_dict, query_set_paths, log_file):
        self.n_ways = n_ways
        self.shots = shots
        self.support_features_dict = support_features_dict
        self.query_features_dict = query_features_dict
        self.query_set = query_set_paths
        self.log_file = log_file
        self.logger = Logger(__name__, log_file)

    def get_task(self, data_support, data_query, labels_support, labels_query):
        """
        inputs:
            data_support : torch.tensor of shape [shot * n_ways, channels, H, W]
            data_query : torch.tensor of shape [n_query, channels, H, W]
            labels_support :  torch.tensor of shape [shot * n_ways + n_query]
            labels_query :  torch.tensor of shape [n_query]
        returns :
            task : Dictionnary : x_support : torch.tensor of shape [n_ways * shot, channels, H, W]
                                 x_query : torch.tensor of shape [n_query, channels, H, W]
                                 y_support : torch.tensor of shape [n_ways * shot]
                                 y_query : torch.tensor of shape [n_query]
        """

        unique_labels = torch.unique(labels_support)
        new_labels_support = torch.zeros_like(labels_support)
        new_labels_query = torch.zeros_like(labels_query)
        for j, y in enumerate(unique_labels):
            new_labels_support[labels_support == y] = j
            new_labels_query[labels_query == y] = j
        labels_support = new_labels_support
        labels_query = new_labels_query

        task = {'x_s': data_support, 'y_s': labels_support.long(),
                'x_q': data_query, 'y_q': labels_query.long()}
        return task

    def generate_tasks(self):
        """

        returns :
            merged_task : { x_support : torch.tensor of shape [batch_size, n_ways * shot, channels, H, W]
                            x_query : torch.tensor of shape [batch_size, n_ways * query_shot, channels, H, W]
                            y_support : torch.tensor of shape [batch_size, n_ways * shot]
                            y_query : torch.tensor of shape [batch_size, n_ways * query_shot]
                            train_mean: torch.tensor of shape [feature_dim]}
        """
        
        # generate support set
        x_support =[]
        y_support = []
        import torch

        all_labels = torch.unique(self.support_features_dict['concat_labels'])

        for label in all_labels:
            indices = (self.support_features_dict['concat_labels'].eq(label)).nonzero(as_tuple=True)[0]
        
            random_indices = np.random.choice(indices, self.shots, replace=False)
            for index in random_indices:
                x_support.append(self.support_features_dict['concat_features'][index])
                y_support.append(self.support_features_dict['concat_labels'][index])

        # generate the query set
        x_query = []
        y_query = []

        for patch in self.query_set:
            index = np.where(np.array(self.query_features_dict['concat_patchs']) == patch)[0][0]
            x_query.append(self.query_features_dict['concat_features'][index])
            y_query.append(self.query_features_dict['concat_labels'][index])    

        return {
            'x_s': torch.stack(x_support).unsqueeze(0),
            'y_s': torch.stack(y_support).unsqueeze(0).unsqueeze(-1),
            'x_q': torch.stack(x_query).unsqueeze(0),
            'y_q': torch.stack(y_query).unsqueeze(0).unsqueeze(-1),
        }


def generate_support_set(support_features_dict, support_features_dict_only_augmented,shots):
    """
    Given the support dataset and the number of shots, this function generates the support set for the few-shot learning task.

    Args:
        support_features_dict (dict): A dictionary containing the support dataset features and labels.
        shots (int): The number of shots.

    Returns:
        torch.tensor [1, S, d]: The support set features.
        torch.tensor [1, S, 1]: The support set labels.
    """
    
    np.random.seed(48)
    
    x_support =[]
    y_support = []

    all_labels = torch.unique(support_features_dict['concat_labels'])

    for label in all_labels:
        indices = (support_features_dict['concat_labels'].eq(label)).nonzero(as_tuple=True)[0]

        # enough samples in not augmented dataset
        if len(indices) >= shots:
            random_indices = np.random.choice(indices, shots, replace=False)
            for index in random_indices:
                x_support.append(support_features_dict['concat_features'][index])
                y_support.append(support_features_dict['concat_labels'][index])

        # not enough samples in not augmented dataset
        else:
            # take all the samples from the not augmented dataset
            for index in indices:
                x_support.append(support_features_dict['concat_features'][index])
                y_support.append(support_features_dict['concat_labels'][index])

            # take the remaining samples from the augmented dataset
            augmented_indices = (support_features_dict_only_augmented['concat_labels'].eq(label)).nonzero(as_tuple=True)[0]
            random_indices = np.random.choice(augmented_indices, shots-len(indices), replace=False)
            for index in random_indices:
                x_support.append(support_features_dict_only_augmented['concat_features'][index])
                y_support.append(support_features_dict_only_augmented['concat_labels'][index])

    return torch.stack(x_support).unsqueeze(0), torch.stack(y_support).unsqueeze(0).unsqueeze(-1)


def generate_query_set(query_features_dict, query_set_paths):
    """
    Given the query dataset and the patches of the window, this function generates the query set for the few-shot learning task.

    Args:
        query_features_dict (dict): A dictionary containing the query dataset features and labels.
        query_set_paths (list): The list of patches of the window.

    Returns:
        torch.tensor [1, Q, d]: The query set features.
        torch.tensor [1, Q, 1]: The query set labels.

    """
    x_query = []
    y_query = []

    for patch in query_set_paths:
        index = np.where(np.array(query_features_dict['concat_patchs']) == patch)[0][0]
        x_query.append(query_features_dict['concat_features'][index])
        y_query.append(query_features_dict['concat_labels'][index])    

    return torch.stack(x_query).unsqueeze(0), torch.stack(y_query).unsqueeze(0).unsqueeze(-1),
        



def extract_query_sets_full_slide_prediction(patch_list, window_size, save_path, squares=False, overlapping=False) -> list:
    """
    Extract query sets from a list of patches, where each query set is a window 

    Args:
    - patch_list: list of patches
    - window_size: size of the sliding window
    - save_path: path to save the query sets
    - squares: if True, the query sets are extracted from the polygons, otherwise the query sets are extracted using the sliding window

    Returns:
    - list of query sets (each query set is a list of patches in a window)
    """
    # Load query sets from memory if previously saved ...
    if os.path.isfile(save_path):
        f = open(save_path, "rb")
        extracted_querysets_dic = pickle.load(f)
        print(" ==> Query sets loaded from {}".format(save_path))
        return extracted_querysets_dic

    # ... otherwise just extract them
    else:
        print(" ==> Beginning query sets computing")
        # os.makedirs(save_dir, exist_ok=True)

    #csv_file = os.path.join(args.split_dir, f"query_{trainset_name}.csv")
    #f = open(csv_file)
    #reader = csv.reader(f)
    ## take all the patches file names
    #reader = [x[0] for x in reader]
    query_sets = []

    if squares:
        querysets_by_poly = {}
        # iterate over all the file names
        for row in tqdm(patch_list):
            # take the patient, slide and polygon number from the file name
            patient, slide, num_polygon = row.split("_")[:3]
            if (patient, slide, num_polygon) not in querysets_by_poly:
                querysets_by_poly[patient, slide, num_polygon] = []
            # for each patch of the polygon append the file name to the list
            querysets_by_poly[patient, slide, num_polygon].append(row)

        ## create a list for all the patches inside the same polygon and append it to the result list
        ## each sliding window = 1 polygon
        result = [
            querysets_by_poly[poly] for poly in querysets_by_poly
        ]  ## list of poligons, each polygon is a list of patches

    ## Build the sliding window query sets, take one patch and all the patches within the window_size
    else:
        regex_pattern = r'\d+_[A-Z]_row_\d+_col_\d+\.jpg'
        list_x, list_y = [], []

        patch_list_copy = patch_list.copy()
        patch_list_copy.sort()

        for row in tqdm(patch_list_copy):

            if re.match(regex_pattern, row) is None:
                continue

            patient, slide, _, x, _, y = row.replace('.jpg', '').split("_")

            x, y = int(x), int(y)
            list_x.append(x)
            list_y.append(y)
            query = []
            for i in range(window_size):
                for j in range(window_size):
                    neigh_patch = f"{patient}_{slide}_row_{x+i}_col_{y+j}.jpg"
                    if neigh_patch in patch_list_copy:
                        query.append(neigh_patch)

        # if enough samples, create the window            
            if len(query) > 0.5 * (window_size**2):
                query_sets.append(query.copy())

                # delete the patches that are already in a query set to avoid overlapping
                if not overlapping:
                    for patch in query:
                        patch_list_copy.remove(patch)

        result = {
            "query_sets": query_sets,
            "min_x": min(list_x),
            "max_x": max(list_x),
            "min_y": min(list_y),
            "max_y": max(list_y),
        }
    print(" ==> Saving query sets to {}".format(save_path))
    f = open(save_path, "wb")
    pickle.dump(result, f)
    f.close()
    return result
