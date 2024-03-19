import torch
from PIL import Image
import matplotlib.pyplot as plt
import os

def patch_plus_proche(i, j, classe_predite, features_support, features_query, noms_patchs_query, noms_patchs_support, 
                      labels_support, labels_query):
    mon_patch = f"21_B_x_{i}_y_{j}"
    classes = ['AM','AN','NT','RE','VE']
    num_classe_predite = classes.index(classe_predite)
    num_classe_reelle = labels_query[noms_patchs_query.index(mon_patch)]
    if num_classe_predite != num_classe_reelle:
        features_mon_patch = features_query[noms_patchs_query.index(mon_patch)]
        distances_classe_predite = []
        distances_classe_annotee = []
        # for p in support[classe_predite]:
        #     d = (z-w)^T * S * (z-w)
        #     distances.append((d,p))
        for k in range(len(noms_patchs_support)):
            if labels_support[k] == num_classe_predite:
                nom_mon_support = noms_patchs_support[k]
                features_mon_support = features_support[k]
                dist2 = torch.norm(features_mon_patch - features_mon_support)
                distances_classe_predite.append((float(dist2), nom_mon_support))
            if labels_support[k] == num_classe_reelle:
                nom_mon_support = noms_patchs_support[k]
                features_mon_support = features_support[k]
                dist2 = torch.norm(features_mon_patch - features_mon_support)
                distances_classe_annotee.append((float(dist2), nom_mon_support))
        distances_classe_annotee.sort()
        distances_classe_predite.sort()
        patch_proche_classe_annotee = distances_classe_annotee[0][1]
        patch_proche_classe_predite = distances_classe_predite[0][1]
        image_mon_patch = Image.open('data/liver_normalized_reinhard/' + mon_patch + '_res1728x1728.jpg')
        image_patch_proche_classe_annotee = Image.open('data/liver_normalized_reinhard/' + patch_proche_classe_annotee + '_res1728x1728.jpg')
        image_patch_proche_classe_predite = Image.open('data/liver_normalized_reinhard/' + patch_proche_classe_predite + '_res1728x1728.jpg')
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
        fig.set_figwidth(20) 
        fig.suptitle(f"Patch : {mon_patch}, classe prédite : {classe_predite}, classe annotée : {classes[num_classe_reelle]}")
        ax1.tick_params(left=False, right=False, labelleft=False, labelbottom=False, bottom=False)
        ax2.tick_params(left=False, right=False, labelleft=False, labelbottom=False, bottom=False)
        ax3.tick_params(left=False, right=False, labelleft=False, labelbottom=False, bottom=False)
        ax1.imshow(image_mon_patch)
        ax1.set_title(mon_patch)
        ax2.imshow(image_patch_proche_classe_predite)
        ax2.set_title(patch_proche_classe_predite)
        ax3.imshow(image_patch_proche_classe_annotee)
        ax3.set_title(patch_proche_classe_annotee)
        folder = f'proches/rouen/reel_{classes[num_classe_reelle]}_predit_{classe_predite}'
        if not os.path.exists(folder):
            os.makedirs(folder)
        plt.savefig(f'{folder}/patch_plus_proche_de_{mon_patch}.png')
        # plt.show()
        plt.close()
