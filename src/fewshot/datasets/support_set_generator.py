import torch
import numpy as np


def generate_support_set(support_features_dict, shots):
    """
    Given the support dataset and the number of shots, this function generates the support set for the few-shot learning task.

    Args:
        support_features_dict (dict): A dictionary containing the support dataset features and labels.
        shots (int): The number of shots.

    Returns:
        torch.tensor [1, S, d]: The support set features.
        torch.tensor [1, S, 1]: The support set labels.
    """
    

    x_support =[]
    y_support = []

    all_labels = torch.unique(support_features_dict['concat_labels'])

    for label in all_labels:
        indices = (support_features_dict['concat_labels'].eq(label)).nonzero(as_tuple=True)[0]
        
        random_indices = np.random.choice(indices, shots, replace=False)
        for index in random_indices:
            x_support.append(support_features_dict['concat_features'][index])
            y_support.append(support_features_dict['concat_labels'][index])

    return torch.stack(x_support).unsqueeze(0), torch.stack(y_support).unsqueeze(0).unsqueeze(-1)
            