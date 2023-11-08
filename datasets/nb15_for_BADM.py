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
        self.sampled = np.zeros_like(y)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]
        y = self.labels[idx]
        target = self.semi_targets[idx]
        sampled = self.sampled[idx]
        return idx, x, y, target, sampled

    def modify_sampled(self, index, new_value):
        self.sampled[index] = new_value

    def clean_sampled(self):
        self.sampled = np.zeros_like(self.sampled)


class nb15_for_BADM():

    def __init__(self, s_non_target = 100, s_target = 100, nb15_non_target_class_num = 4, random_seed = None):

        Labelled_non_target_target = pd.read_csv('./data/nb15/Labelled_data.csv', index_col=0)
        Unlabelled_mix = pd.read_csv('./data/nb15/Unlabelled_data.csv', index_col=0)
        test_mix = pd.read_csv('./data/nb15/test_data.csv', index_col=0)

        Labelled_non_target = Labelled_non_target_target.loc[(Labelled_non_target_target['attack_cat'] == 'Fuzzers') |
                                             (Labelled_non_target_target['attack_cat'] == 'Analysis') |
                                             (Labelled_non_target_target['attack_cat'] == 'Exploits') |
                                             (Labelled_non_target_target['attack_cat'] == 'Reconnaissance')]

        non_target_list = [Labelled_non_target.loc[(Labelled_non_target['attack_cat'] == 'Fuzzers')].sample(s_non_target, random_state = random_seed),
                    Labelled_non_target.loc[(Labelled_non_target['attack_cat'] == 'Analysis')].sample(s_non_target, random_state = random_seed),
                    Labelled_non_target.loc[(Labelled_non_target['attack_cat'] == 'Exploits')].sample(s_non_target, random_state = random_seed),
                    Labelled_non_target.loc[(Labelled_non_target['attack_cat'] == 'Reconnaissance')].sample(s_non_target, random_state = random_seed)]

        sampled_non_target = pd.concat(non_target_list[:nb15_non_target_class_num])

        Labelled_target = Labelled_non_target_target.loc[(Labelled_non_target_target['attack_cat'] == 'DoS') |
                                              (Labelled_non_target_target['attack_cat'] == 'Generic') |
                                              (Labelled_non_target_target['attack_cat'] == 'Backdoor')]
        target_1 = Labelled_target.loc[(Labelled_target['attack_cat'] == 'DoS')].sample(s_target, random_state = random_seed)
        target_2 = Labelled_target.loc[(Labelled_target['attack_cat'] == 'Generic')].sample(s_target, random_state = random_seed)
        target_3 = Labelled_target.loc[(Labelled_target['attack_cat'] == 'Backdoor')].sample(s_target, random_state = random_seed)
        sampled_target = pd.concat([target_1, target_2, target_3])

        selected_Labelled_non_target_target = pd.concat([sampled_target, sampled_non_target])

        # get x_test, y_test, target_y_test
        # label/y_test, target=1, non_target=normal=0
        # target_y_test, target=non_target=normal=1

        test_mix.loc[(test_mix['attack_cat'] == 'Generic') | (test_mix['attack_cat'] == 'DoS') | (test_mix['attack_cat'] == 'Backdoor'), 'label'] = 1
        test_mix.loc[(test_mix['attack_cat'] != 'Generic') & (test_mix['attack_cat'] != 'DoS') & (test_mix['attack_cat'] != 'Backdoor'), 'label'] = 0
        # print('test_mix:', Counter(test_mix.label))

        x_test = test_mix.drop(['attack_cat', 'label'], axis=1).values
        x_test = x_test.astype(np.float32)
        x_test[np.isnan(x_test)] = 0
        y_test = test_mix.loc[:, ['label']].values[:, 0].astype(int)
        target_y_test = np.ones(len(x_test))
        print('y_test:', Counter(y_test))
        print('target_y_test', Counter(target_y_test))

        # train

        # unlabeled
        x_u = Unlabelled_mix.drop(['attack_cat', 'label'], axis=1).values
        x_u = x_u.astype(np.float32)
        x_u[np.isnan(x_u)] = 0
        y_u = np.zeros(len(x_u))
        print('y_u:', Counter(y_u))

        # labeled
        x_l = selected_Labelled_non_target_target
        x_l = x_l.drop(['attack_cat', 'label'], axis=1).values
        x_l = x_l.astype(np.float32)
        x_l[np.isnan(x_l)] = 0
        y_l = selected_Labelled_non_target_target.loc[:, ['label']].values[:, 0].astype(int)
        print('y_l:', Counter(y_l))

        # MinMaxScaler
        scaler = MinMaxScaler()
        scaler = scaler.fit(x_u)
        x_u = scaler.transform(x_u)
        x_l = scaler.transform(x_l)
        x_test = scaler.transform(x_test)

        
        x_train = np.vstack((x_u, x_l))
        target_y_train = np.hstack((y_u, y_l))
        # -2 means non-target, -1 means target
        target_y_train[target_y_train == -1] = -2
        target_y_train[target_y_train == 1] = -1

        y_train = np.hstack((y_u, y_l))
        y_train[y_train == -1] = 0

        print(x_u.shape[1])
        print('y_train', Counter(y_train))
        print('target_y_train', Counter(target_y_train))

        print('y_test', Counter(y_test))
        print('target_y_test', Counter(target_y_test))

        self.train_set = MyDataset(x_train, y_train, target_y_train)

        self.test_set = MyDataset(x_test, y_test, target_y_test)

    def modify_sampled(self, index, new_value):
        self.train_set.modify_sampled(index, new_value)

    def clean_sampled(self):
        self.train_set.clean_sampled()

    def loaders(self, batch_size: int, shuffle_train=True, shuffle_test=False, num_workers: int = 0, drop_last_train = True):

        train_loader = DataLoader(dataset=self.train_set, batch_size=batch_size, shuffle=shuffle_train,
                                  num_workers=num_workers, drop_last= drop_last_train)

        test_loader = DataLoader(dataset=self.test_set, batch_size=batch_size, shuffle=shuffle_test,
                                 num_workers=num_workers, drop_last=False)

        return train_loader, test_loader


if __name__ == "__main__":
    nb15_for_BADM()