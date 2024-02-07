from torch.utils.data import Subset, Dataset
from collections import Counter
from PIL import Image
from base.torchvision_dataset import TorchvisionDataset
import torchvision.transforms as transforms
import numpy as np
from torchvision.datasets import FashionMNIST
import torch

class MyFashionMNIST(FashionMNIST):
    """
    Torchvision FashionMNIST class with additional targets for the semi-supervised setting and patch of __getitem__
    method to also return the semi-supervised target as well as the index of a data sample.
    """

    def __init__(self, *args, **kwargs):
        super(MyFashionMNIST, self).__init__(*args, **kwargs)

        self.semi_targets = torch.zeros_like(self.targets)

    def __getitem__(self, index):
        """Override the original method of the MyFashionMNIST class.
        Args:
            index (int): Index

        Returns:
            tuple: (image, target, semi_target, index)
        """
        img, target, semi_target = self.data[index], int(self.targets[index]), int(self.semi_targets[index])

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img.numpy(), mode='L')

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, semi_target

class MyDataset(Dataset):
    def __init__(self, x, y, target, transform=None, target_transform=None):
        self.data = x
        self.labels = y
        self.semi_targets = target
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = self.data[idx]
        label = self.labels[idx]
        semi_target = self.semi_targets[idx]

        img = Image.fromarray(img.numpy(), mode='L')

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            label = self.target_transform(label)

        return img, label, semi_target

class FashionMNIST_Dataset(TorchvisionDataset):

    def get_random_sample(self, dataset, selected_class:int, labeled:int, unlabeled:int, seed:int):
        np.random.seed(seed)
        idx = np.argwhere(dataset.targets == selected_class).flatten()
        idx = np.random.choice(idx, labeled + unlabeled, False)
        return idx[:labeled], idx[labeled:]


    def __init__(self, root: str,

                 normal_class: int,
                 unlabeled_normal_number: int,
                 labeled_normal_number: int,
                 test_normal: int,

                 non_target_outlier_class: int,
                 unlabeled_non_target_outlier_number: int,
                 labeled_non_target_outlier_number: int,
                 test_non_target_outlier: int,

                 target_outlier_class: int,
                 unlabeled_target_outlier_number: int,
                 labeled_target_outlier_number: int,
                 test_target_outlier: int,
                 random_seed:int = 0):

        # super().__init__(root)
        self.root = root

        self.random_seed = random_seed

        # FashionMNIST preprocessing: feature scaling to [0, 1]
        transform = transforms.ToTensor()
        target_transform = None
        target_transform = transforms.Lambda(lambda x: int(x == target_outlier_class))
        # target_transform = None
        
        # Get train set
        train_set = MyFashionMNIST(root=self.root, train=True, transform=transform, target_transform=target_transform,
                                   download=True)

        # Create semi-supervised setting
        labeled_normal_idx, unlabeled_normal_idx = self.get_random_sample(train_set, normal_class, labeled_normal_number, unlabeled_normal_number, self.random_seed + 0)
        labeled_non_target_outlier_idx, unlabeled_non_target_outlier_idx = self.get_random_sample(train_set, non_target_outlier_class, labeled_non_target_outlier_number, unlabeled_non_target_outlier_number, self.random_seed + 1)
        labeled_target_outlier_idx, unlabeled_target_outlier_idx = self.get_random_sample(train_set, target_outlier_class, labeled_target_outlier_number, unlabeled_target_outlier_number, self.random_seed + 2)

        print("=========train=========")
        print("labeled_normal_idx:\t",len(labeled_normal_idx), "\tunlabeled_normal_idx:\t",len(unlabeled_normal_idx))
        print("labeled_non_target_outlier_idx:\t",len(labeled_non_target_outlier_idx), "\tunlabeled_non_target_outlier_idx:\t",len(unlabeled_non_target_outlier_idx))
        print("labeled_target_outlier_idx:\t",len(labeled_target_outlier_idx), "\tunlabeled_target_outlier_idx:\t",len(unlabeled_target_outlier_idx))

        idx = np.concatenate((labeled_normal_idx , unlabeled_normal_idx , labeled_non_target_outlier_idx , unlabeled_non_target_outlier_idx , labeled_target_outlier_idx , unlabeled_target_outlier_idx))
        
        # print(len(idx))

        train_set.semi_targets[labeled_normal_idx] = 0
        train_set.semi_targets[labeled_non_target_outlier_idx] = -2
        train_set.semi_targets[labeled_target_outlier_idx] = -1
        
        
        # Subset train_set to semi-supervised setup
        sub_train_set = Subset(train_set, idx)
        
        # Get test set
        test_set = MyFashionMNIST(root=self.root, train=False, transform=transform,
                                       target_transform=target_transform, download=True)
        
        labeled_normal_idx, _ = self.get_random_sample(test_set, normal_class, test_normal, 0, self.random_seed + 3)
        labeled_non_target_outlier_idx, _ = self.get_random_sample(test_set, non_target_outlier_class, test_non_target_outlier, 0, self.random_seed + 4)
        labeled_target_outlier_idx, _ = self.get_random_sample(test_set, target_outlier_class, test_target_outlier, 0, self.random_seed + 5)

        test_set.semi_targets[labeled_normal_idx] = 0
        test_set.semi_targets[labeled_non_target_outlier_idx] = -2
        test_set.semi_targets[labeled_target_outlier_idx] = -1

        print("=========test=========")
        print("labeled_normal_idx:\t",len(labeled_normal_idx))
        print("labeled_non_target_outlier_idx:\t",len(labeled_non_target_outlier_idx))
        print("labeled_target_outlier_idx:\t",len(labeled_target_outlier_idx))

        idx = np.concatenate((labeled_normal_idx, labeled_non_target_outlier_idx, labeled_target_outlier_idx))
        
        sub_test_set = Subset(test_set, idx)

        x_train = sub_train_set.dataset.train_data[sub_train_set.indices]
        y_train = sub_train_set.dataset.targets[sub_train_set.indices]
        target_y_train = sub_train_set.dataset.semi_targets[sub_train_set.indices]

        x_test = sub_test_set.dataset.train_data[sub_test_set.indices]
        y_test = sub_test_set.dataset.targets[sub_test_set.indices]
        target_y_test = sub_test_set.dataset.semi_targets[sub_test_set.indices]

        self.train_set = MyDataset(x_train, y_train, target_y_train, transform=transform, target_transform=target_transform)
        self.test_set = MyDataset(x_test, y_test, target_y_test, transform=transform, target_transform=target_transform)


