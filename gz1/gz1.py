import torch as t
import pickle
from PIL import Image

class GZ1():
    def __init__(self, train, transform=None):
        self.train = train
        self.transform = transform
  
        img_fname = {True: "gz1/imgs_train", False: "gz1/imgs_test"}[train]
        targets_fname = {True: "gz1/targets_train", False: "gz1/targets_test"}[train]

        with open(img_fname, 'rb') as f:
            self.data = pickle.load(f)
        with open(targets_fname, 'rb') as f:
            self.targets = pickle.load(f)

    def __len__(self):
        return self.targets.shape[0]
        
    def __getitem__(self, idx):
        img, target = self.data[idx], self.targets[idx]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        return img, target
