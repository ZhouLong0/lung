import torch
import numpy as np
import random

class CategoriesSampler():
    """
            CategorySampler
            inputs:
                label : All labels of dataset
                n_batch : Number of batches to load
                n_ways : Number of classification ways (n_ways)
                s_shot : Support shot
                n_query : Size of query set
                sampling : 'same_slice' : Uniform distribution over query data with the same slice name
                           'sliding_window' : sliding window sampling, to use on full slide data
                           'squares' : create query sets from patchs that come from the same square annotation on the slice
                alpha : Dirichlet's concentration parameter

            returns :
                sampler : CategoriesSampler object that will yield batch when iterated
                When iterated returns : batch
                        data : torch.tensor [n_support + n_query, channel, H, W]
                               [support_data, query_data]
                        labels : torch.tensor [n_support + n_query]
                               [support_labels, query_labels]
    """

    def __init__(self, n_batch, n_ways, s_shot, n_query, sampling):
        self.n_batch = n_batch                      # the number of iterations in the dataloader
        self.s_shot = s_shot
        self.n_query = n_query
        self.sampling = sampling
        self.n_ways = n_ways
        self.slice=None
        
    def create_list_index(self, label_support, label_query, slices_query, all_patchs_query=None, slice_to_take=None, patchs_this_query=None):
        label_support = np.array(label_support)     # all data label
        self.index_support = []                     # the data index of each class
        
        for i in range(max(label_support) + 1):
            ind = np.argwhere(label_support == i).reshape(-1)  # all data index of this class
            ind = torch.from_numpy(ind)
            self.index_support.append(ind)
            
        label_query = np.array(label_query)         # all data label
        self.index_query = []                       # the data index of each class

        if self.sampling == 'same_slice':
            name_slices = np.unique(slices_query)
            for i, slice_name in enumerate(name_slices):
                ind = np.argwhere(slices_query == slice_name).reshape(-1)
                ind = torch.from_numpy(ind)
                self.index_query.append(ind)
            if slice_to_take==None:
                self.slice = np.random.randint(len(name_slices))
            else:
                self.slice = slice_to_take
            # print(f'Slice {self.slice}/{len(name_slices)}')
            # print('slice: ', name_slices[self.slice])
        elif self.sampling in ['sliding_window', 'squares']:
            ind = [i for i in range(len(all_patchs_query)) if (all_patchs_query[i] in patchs_this_query)]
            random.shuffle(ind)
            ind = np.array(ind)
            ind = torch.from_numpy(ind)
            self.index_query.append(ind)
        else:
            for i in range(max(label_support) + 1):
                ind = np.argwhere(label_query == i).reshape(-1)  # all data index of this class
                ind = torch.from_numpy(ind)
                self.index_query.append(ind)
            
        self.list_classes = []
        for i_batch in range(self.n_batch):
            self.list_classes.append(torch.tensor([0, 1, 2, 3, 4]))  # random sample num_class indexs
        
    
class SamplerSupport:
    def __init__(self, cat_samp,shots_par_patients=True,slices_support=None):
        self.name = "SamplerSupport"
        self.list_classes = cat_samp.list_classes
        self.n_batch = cat_samp.n_batch
        self.s_shot = cat_samp.s_shot
        self.index_support = cat_samp.index_support
        self.shots_par_patients=shots_par_patients
        self.slices_support=slices_support
        self.list_patients = ['23', '43', '85', '45', '73','82', '25', '61', '34', '49', '27', '28', '20', '29', '52']

    def __len__(self):
        return self.n_batch

    def __iter__(self):
        if self.shots_par_patients:
            # Select in self.slices_support the slices of the patients in list_patients
            for i_batch in range(self.n_batch):
                support = []
                classes = self.list_classes[i_batch]
                for c in classes:
                    l = self.index_support[c]                       # all data indexs of this class
                    # get the names of the slices corresponding to the indexes
                    names = [self.slices_support[i] for i in l]
                    # get the names of the patients corresponding to the indexes
                    patients = [name.split('_')[0] for name in names]
                    # get the indexes of the patients in list_patients
                    indexes = torch.tensor([i for i in range(len(patients)) if (patients[i] in self.list_patients)])
                    #print("indexes", len(indexes))
                    pos = torch.randperm(len(indexes))[:self.s_shot]                 # select all data
                    #support.append(l[indexes])
                    #print("pos", len(l[indexes[pos]]))
                    support.append(l[indexes[pos]])

                support = torch.cat(support)
                #print('support size', support.shape)
                yield support


        else:
            for i_batch in range(self.n_batch):
                support = []
                classes = self.list_classes[i_batch]
                for c in classes:
                    l = self.index_support[c]                       # all data indexs of this class
                    pos = torch.randperm(len(l))[:self.s_shot]                 # select all data
                    support.append(l[pos])                         # C'EST DE LA QUE VIENT LE RANDOM !
                    #support.append(l[:self.s_shot])
                support = torch.cat(support)
            
                yield support


class SamplerQuery:
    def __init__(self, cat_samp):
        self.name = "SamplerQuery"
        self.list_classes = cat_samp.list_classes
        self.n_batch = cat_samp.n_batch
        self.index_query = cat_samp.index_query
        self.n_query = cat_samp.n_query
        self.sampling = cat_samp.sampling
        self.slice = cat_samp.slice
        self.n_ways = cat_samp.n_ways

    def __len__(self):
        return self.n_batch

    def __iter__(self):
        for i_batch in range(self.n_batch):
            assert self.sampling in ['same_slice', 'sliding_window', 'squares']

            if self.sampling == 'same_slice':
                complete_possible_samples = self.index_query[self.slice]
                pos = torch.randperm(len(complete_possible_samples))[:self.n_query]
                query = complete_possible_samples[pos]
            
            elif self.sampling in ['sliding_window', 'squares']:
                complete_possible_samples = self.index_query[0]
                query = complete_possible_samples 
                
            yield query

