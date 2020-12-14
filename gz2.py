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


def threshold_search(dir_path, target):
    with open(f"{dir_path}/targets", 'rb') as f:
        targets = pickle.load(f)

    df = pd.DataFrame(targets)
    df['consensus_coefficient'] = (df.max(axis=1) + 1) / (df.sum(axis=1) + 2)
    df['class'] = df.idxmax(axis=1)

    min_val = 0
    max_val = 1.0
    while True:
        consensus_quantile = (min_val + max_val) / 2
        red = df[df['consensus_coefficient'] >=
                df.groupby('class')['consensus_coefficient'].transform('quantile', consensus_quantile)]
        num = red.shape[0]
        print(f"{num}: {consensus_quantile}")
        if num < target:
            max_val = consensus_quantile
        else:
            if round(max_val - min_val, 10) == 0:
                return consensus_quantile
            min_val = consensus_quantile


def load_consensus_data(dir_path, consensus_quantile, sample_size, prop_train,
                        train_transform=None, test_transform=None, one_hot=True):
    with open(f"{dir_path}/imgs", 'rb') as f:
        data = pickle.load(f)
    with open(f"{dir_path}/targets", 'rb') as f:
        targets = pickle.load(f)

    # permute the data
    perm = np.random.permutation(targets.shape[0])
    data = data[perm]
    targets = targets[perm]

    df = pd.DataFrame(targets)
    num_classes = len(df.columns)

    # determine majority class and corrected consensus coefficient of majority class
    df['consensus_coefficient'] = (df.max(axis=1) + 1) / (df.sum(axis=1) + 2)
    df['class'] = df.idxmax(axis=1)

    # keep only data points with a sufficiently high consensus coefficient
    df = df[df['consensus_coefficient'] >=
            df.groupby('class')['consensus_coefficient'].transform('quantile', consensus_quantile)]
    # sample down to desired number of data points
    df = df.sample(sample_size)

    data = data[df.index]
    if one_hot:
        targets = np.zeros((len(df.index), num_classes))
        targets[np.arange(len(df.index)), df['class']] = 1
    else:
        targets = df.iloc[:, :num_classes].to_numpy()

    # permute again to avoid any bugs related to the consensus filtering
    perm = np.random.permutation(targets.shape[0])
    data = data[perm]
    targets = targets[perm]

    num_train = round(prop_train * sample_size)
    data_train = data[:num_train]
    data_test = data[num_train:]
    targets_train = targets[:num_train]
    targets_test = targets[num_train:]

    return num_classes, \
           GZ2(data_train, targets_train, transform=train_transform), \
           GZ2(data_test, targets_test, transform=test_transform)
