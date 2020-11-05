import pickle
import numpy as np
import pandas as pd
from PIL import Image


class GZ2():
    def __init__(self, data, targets, transform=None):
        self.transform = transform
        self.data = data
        self.targets = targets

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


def load_consensus_data(dir_path, consensus_quantile, sample_size, train_transform=None, test_transform=None):
    with open(f"{dir_path}/imgs", 'rb') as f:
        data = pickle.load(f)
    with open(f"{dir_path}/targets", 'rb') as f:
        targets = pickle.load(f)

    # permute the data
    perm = np.random.permutation(targets.shape[0])
    data = data[perm]
    targets = targets[perm]

    df = pd.DataFrame(targets)

    # determine majority class and corrected consensus coefficient of majority class
    df['consensus_coefficient'] = (df.max(axis=1) + 1) / (df.sum(axis=1) + 2)
    df['class'] = df.idxmax(axis=1)

    # keep only data points with a sufficiently high consensus coefficient
    df = df[df['consensus_coefficient'] >=
            df.groupby('class')['consensus_coefficient'].transform('quantile', consensus_quantile)]
    # sample down to desired number of data points
    df = df.sample(sample_size)

    data = data[df.index]
    targets = df.iloc[:, :-2].to_numpy()

    # permute again to avoid any bugs related to the consensus filtering
    perm = np.random.permutation(targets.shape[0])
    data = data[perm]
    targets = targets[perm]

    data_train, data_test = np.split(data, 2)
    targets_train, targets_test = np.split(targets, 2)

    return targets.shape[1], \
           GZ2(data_train, targets_train, transform=train_transform), \
           GZ2(data_test, targets_test, transform=test_transform)
