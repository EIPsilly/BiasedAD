from torch.utils.data import DataLoader, Dataset
# from base.base_dataset import BaseADDataset
import pandas as pd
import numpy as np
from collections import Counter
from sklearn.preprocessing import StandardScaler, MinMaxScaler


class MyDataset(Dataset):
    def __init__(self, x, y, target):
        self.data = x
        self.labels = y
        self.semi_targets = target

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]
        y = self.labels[idx]
        target = self.semi_targets[idx]
        return x, y, target


class SQBDataset():

    def __init__(self, sqb_test_frac):

        if sqb_test_frac is not None:
            sqbdata = np.load(f'./data/SQB/sqb_data_for_BAD_test_{sqb_test_frac}.npz')
        else:
            sqbdata = np.load("./data/SQB/sqb_data_for_BAD.npz")

        x_train = sqbdata["x_train"]
        y_train = sqbdata["y_train"]
        target_y_train = sqbdata["target_y_train"]
        x_test = sqbdata["x_test"]
        y_test = sqbdata["y_test"]
        target_y_test = sqbdata["target_y_test"]

        print("Counter(y_train)", Counter(y_train))
        print("Counter(target_y_train)", Counter(target_y_train))
        print("Counter(y_test)", Counter(y_test))
        print("Counter(target_y_test)", Counter(target_y_test))

        self.train_set = MyDataset(x_train, y_train, target_y_train)

        self.test_set = MyDataset(x_test, y_test, target_y_test)

    def loaders(self, batch_size: int, shuffle_train=True, shuffle_test=False, num_workers: int = 0, drop_last_train = True):

        train_loader = DataLoader(dataset=self.train_set, batch_size=batch_size, shuffle=shuffle_train,
                                  num_workers=num_workers, drop_last= drop_last_train)

        test_loader = DataLoader(dataset=self.test_set, batch_size=batch_size, shuffle=shuffle_test,
                                 num_workers=num_workers, drop_last=False)

        return train_loader, test_loader


if __name__ == "__main__":
    SQBDataset()
