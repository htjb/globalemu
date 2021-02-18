import numpy as np
import os
import pandas as pd
from globalemu.cmSim import calc_signal
from globalemu.resample import sampling


class process():
    def __init__(self, num, z, **kwargs):
        print('Preprocessing started...')

        for key, values in kwargs.items():
            if key not in set(
                    ['base_dir', 'data_location', 'xHI', 'logs']):
                raise KeyError("Unexpected keyward argument in process()")

        self.num = num
        self.z = z
        self.base_dir = kwargs.pop('base_dir', 'model_dir/')
        self.data_location = kwargs.pop('data_location', 'data/')
        self.xHI = kwargs.pop('xHI', False)
        self.logs = kwargs.pop('logs', [0, 1, 2])

        if not os.path.exists(self.base_dir):
            os.mkdir(self.base_dir)

        np.savetxt(self.base_dir + 'z.txt', self.z)

        full_train_data = pd.read_csv(
            self.data_location + 'train_data.txt',
            delim_whitespace=True, header=None).values
        full_train_labels = pd.read_csv(
            self.data_location + 'train_labels.txt',
            delim_whitespace=True, header=None).values

        if self.xHI is False:
            np.save(
                self.base_dir + 'AFB_norm_factor.npy',
                full_train_labels[0, -1]*1e-3)
            res = calc_signal(self.z, self.base_dir)

        if self.num == 'full':
            train_data = full_train_data.copy()
            if self.xHI is False:
                train_labels = full_train_labels.copy() - res.deltaT
            else:
                train_labels = full_train_labels.copy()
        else:
            ind = []
            for i in range(len(full_train_labels)):
                index = np.random.randint(0, len(full_train_labels))
                if index not in set(ind):
                    ind.append(index)
                if len(ind) == self.num:
                    break
            ind = np.array(ind)

            train_data, train_labels = [], []
            for i in range(len(full_train_labels)):
                if np.any(ind == i):
                    train_data.append(full_train_data[i, :])
                    if self.xHI is False:
                        train_labels.append(full_train_labels[i] - res.deltaT)
                    else:
                        train_labels.append(full_train_labels[i])
            train_data, train_labels = np.array(train_data), \
                np.array(train_labels)

        log_td = []
        for i in range(train_data.shape[1]):
            if i in set(self.logs):
                for j in range(train_data.shape[0]):
                    if train_data[j, i] == 0:
                        train_data[j, i] = 1e-6
                log_td.append(np.log10(train_data[:, i]))
            else:
                log_td.append(train_data[:, i])
        train_data = np.array(log_td).T

        samples = sampling(
            self.z, self.base_dir, train_labels).samples

        resampled_labels = []
        for i in range(len(train_labels)):
            resampled_labels.append(
                np.interp(samples, self.z, train_labels[i]))
        train_labels = np.array(resampled_labels)

        norm_s = (samples.copy() - samples.min())/(samples.max()-samples.min())

        data_mins = train_data.min(axis=0)
        data_maxs = train_data.max(axis=0)

        norm_train_data = []
        for i in range(train_data.shape[1]):
            norm_train_data.append(
                (train_data[:, i] - data_mins[i])/(data_maxs[i]-data_mins[i]))
        norm_train_data = np.array(norm_train_data).T

        if self.xHI is False:
            labels_stds = train_labels.std()
            norm_train_labels = [
                train_labels[i, :]/labels_stds
                for i in range(train_labels.shape[0])]
            norm_train_labels = np.array(norm_train_labels)

            norm_train_labels = norm_train_labels.flatten()
            np.save(self.base_dir + 'labels_stds.npy', labels_stds)
        else:
            norm_train_labels = train_labels.flatten()

        if self.num != 'full':
            np.savetxt(self.base_dir + 'indices.txt', ind)
        np.savetxt(self.base_dir + 'data_mins.txt', data_mins)
        np.savetxt(self.base_dir + 'data_maxs.txt', data_maxs)

        flattened_train_data = []
        for i in range(len(norm_train_data)):
            for j in range(len(norm_s)):
                flattened_train_data.append(
                    np.hstack([norm_train_data[i, :], norm_s[j]]))
        flattened_train_data = np.array(flattened_train_data)

        train_data, train_label = flattened_train_data, norm_train_labels
        train_dataset = np.hstack([train_data, train_label[:, np.newaxis]])

        np.savetxt(
            self.base_dir + 'train_dataset.csv', train_dataset, delimiter=',')
        np.savetxt(self.base_dir + 'train_data.txt', train_data)
        np.savetxt(self.base_dir + 'train_label.txt', train_label)

        print('...preprocessing done.')
