import torch
import torch.nn as nn
import torch.utils.data as data
import torch.utils.model_zoo as model_zoo

import numpy as np


class ActRecDataset(data.Dataset):
    def __init__(self, data, test=False):
        if len(data) == 0:
            raise(RuntimeError("Error, no data!"))
        features=[]
        class_num=[]
        class_name=[]
        for i in range(len(data)):
            features.append(np.reshape(data[i]['features'], 5120))
            if not test:
                class_num.append(data[i]['class_num'])
                class_name.append((data[i]['class_name']))

        self.data = data
        self.class_num = class_num
        self.class_name=class_name
        self.features=features
        #self.loader = loader

    def __getitem__(self, index):
        features = self.features[index]
        target=self.class_num[index]

        return features, target

    def __len__(self):
        return len(self.features)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str