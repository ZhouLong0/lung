import csv
from tqdm import tqdm
import os
import pickle
import re


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
            if len(query) > 0.3 * (window_size**2):
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
