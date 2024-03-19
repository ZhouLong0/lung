import os

import PIL.Image as Image
import numpy as np
import matplotlib.pyplot as plt
from histomicstk.preprocessing.color_normalization import reinhard

__all__ = ['DatasetFolder']

res = ['512x512','768x768','1152x1152','1728x1728']

existing_classes = ['AM','AN','NT','RE','VE','UNKNOWN']

class DatasetFolder(object):

    def __init__(self, root, split_dir, split_type, transform, patch_size, out_name=False, sampling='same_slice', split_name="", split_hospital=""):
        assert split_type in ['query', 'support']
        if split_type=='support':
            split_file = os.path.join(split_dir, split_type + '_' + split_hospital + '.csv')
        elif split_type=='query':
            split_file = os.path.join(split_dir, f'{split_type}_{split_name}.csv')
        assert os.path.isfile(split_file), split_file
        with open(split_file, 'r') as f:
            split = [x.strip().split(',') for x in f.readlines() if x.strip() != '']
            
        # data, ori_labels = [[x[0] + '_res' + res[0] + '.jpg', x[0] + '_res' + res[1] + '.jpg', x[0] + '_res' + res[2] + '.jpg'] for x in split], [x[1] for x in split]
        data, ori_labels = [x[0] + '_res' + res[patch_size] + '.jpg' for x in split], [x[1] for x in split]
        label_key = np.array(existing_classes)
        label_map = dict(zip(label_key, range(len(label_key))))
        mapped_labels = [label_map[x] for x in ori_labels]

        self.root = root
        self.transform = transform
        self.data = data
        self.labels = mapped_labels
        self.out_name = out_name
        self.length = len(self.data)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        # for i in range(3):
        #     img = Image.open(os.path.join(self.root, self.data[index][i])).convert('RGB')
        #     if self.transform:
        #         img = self.transform(img)
        #     if i == 0:
        #         imgs = img.unsqueeze(0)
        #     else:
        #         imgs = torch.cat((imgs, img.unsqueeze(0)), 0)

        imgs = Image.open(os.path.join(self.root, self.data[index])).convert('RGB')
        if self.transform:
            imgs = self.transform(imgs)

        label = self.labels[index]
        label = int(label)
        if self.out_name:       
            return imgs, label, self.data[index]     # self.data[index]
        else:        
            return imgs, label, index

