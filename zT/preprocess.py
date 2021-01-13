import numpy as np
from sklearn.utils import shuffle
import os
from zT.cmSim import calc_signal
from zT.resample import sampling, samplingXHI

class process():
    def __init__(self, num, **kwargs):
        print('Preprocessing started...')
        self.num = num
        self.base_dir = kwargs.pop('base_dir', 'results/')
        self.data_location = kwargs.pop('data_location', 'data/')

        if not os.path.exists(self.base_dir):
            os.mkdir(self.base_dir)

        orig_z = np.linspace(5, 50, 451)

        full_train_data = np.loadtxt(self.data_location + 'train_data.txt')
        full_train_labels = np.loadtxt(self.data_location + 'train_labels.txt')

        np.save(self.base_dir + 'AFB_norm_factor.npy', full_train_labels[0, -1]*1e-3)

        res = calc_signal(orig_z, base_dir=self.base_dir)

        if self.num == 'full':
            train_data = full_train_data.copy()
            train_labels = full_train_labels.copy() - res.deltaT
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
                    train_labels.append(full_train_labels[i]- res.deltaT)
            train_data, train_labels = np.array(train_data), np.array(train_labels)

        log_td = []
        for i in range(train_data.shape[1]):
            if i in set([0, 1]):
                log_td.append(np.log10(train_data[:, i]))
            elif i == 2:
                for j in range(train_data.shape[0]):
                    if train_data[j, i] == 0:
                        train_data[j, i] = 1e-6
                log_td.append(np.log10(train_data[:, i]))
            else:
                log_td.append(train_data[:, i])
        train_data = np.array(log_td).T

        samples = sampling(self.base_dir, data_location=self.data_location).samples
        resampled_labels = []
        for i in range(len(train_labels)):
            resampled_labels.append(np.interp(samples, orig_z, train_labels[i]))
        train_labels = np.array(resampled_labels)

        norm_s = (samples.copy() - samples.min())/(samples.max()-samples.min())

        labels_stds = train_labels.std()

        data_mins = train_data.min(axis=0)
        data_maxs = train_data.max(axis=0)

        norm_train_data = []
        for i in range(train_data.shape[1]):
            norm_train_data.append((train_data[:, i] - data_mins[i])/(data_maxs[i]-data_mins[i]))
        norm_train_data = np.array(norm_train_data).T

        norm_train_labels = []
        for i in range(train_labels.shape[0]):
            norm_train_labels.append(train_labels[i, :]/labels_stds)
        norm_train_labels = np.array(norm_train_labels)

        norm_train_labels = norm_train_labels.flatten()
        print(norm_train_labels.shape)

        if self.num != 'full':
            np.savetxt(self.base_dir + 'indices.txt', ind)
        np.save(self.base_dir + 'labels_stds.npy', labels_stds)
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

        np.savetxt(self.base_dir + 'zT_train_dataset.csv', train_dataset, delimiter=',')
        np.savetxt(self.base_dir + 'zT_train_data.txt', train_data)
        np.savetxt(self.base_dir + 'zT_train_label.txt', train_label)

        print('...preprocessing done.')

class processXHI():
    def __init__(self, num, **kwargs):
        print('Preprocessing started...')
        self.num = num
        self.base_dir = kwargs.pop('base_dir', 'results/')
        self.data_location = kwargs.pop('data_location', 'data/')

        if not os.path.exists(self.base_dir):
            os.mkdir(self.base_dir)

        orig_z = np.hstack([np.arange(5, 15.1, 0.1), np.arange(16, 31, 1)])

        full_train_data = np.loadtxt(self.data_location + 'train_data.txt')
        full_train_labels = np.loadtxt(self.data_location + 'train_labels.txt')

        if self.num == 'full':
            train_data = full_train_data.copy()
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
                    train_labels.append(full_train_labels[i])
            train_data, train_labels = np.array(train_data), np.array(train_labels)

        log_td = []
        for i in range(train_data.shape[1]):
            if i in set([0, 1]):
                log_td.append(np.log10(train_data[:, i]))
            elif i == 2:
                for j in range(train_data.shape[0]):
                    if train_data[j, i] == 0:
                        train_data[j, i] = 1e-6
                log_td.append(np.log10(train_data[:, i]))
            else:
                log_td.append(train_data[:, i])
        train_data = np.array(log_td).T

        samples = samplingXHI(self.base_dir, data_location=self.data_location, plot=True).samples
        resampled_labels = []
        for i in range(len(train_labels)):
            resampled_labels.append(np.interp(samples, orig_z, train_labels[i]))
        train_labels = np.array(resampled_labels)

        norm_z = (samples.copy() - samples.min())/(samples.max()-samples.min())
        print(norm_z.shape)
        #labels_stds = train_labels.std()

        data_mins = train_data.min(axis=0)
        data_maxs = train_data.max(axis=0)

        norm_train_data = []
        for i in range(train_data.shape[1]):
            norm_train_data.append((train_data[:, i] - data_mins[i])/(data_maxs[i]-data_mins[i]))
        norm_train_data = np.array(norm_train_data).T

        norm_train_labels = train_labels.flatten()
        print(norm_train_labels.shape)


        if self.num != 'full':
            np.savetxt(self.base_dir + 'indices.txt', ind)
        np.savetxt(self.base_dir + 'data_mins.txt', data_mins)
        np.savetxt(self.base_dir + 'data_maxs.txt', data_maxs)

        flattened_train_data = []
        for i in range(len(norm_train_data)):
            for j in range(len(norm_z)):
                flattened_train_data.append(
                    np.hstack([norm_train_data[i, :], norm_z[j]]))
        flattened_train_data = np.array(flattened_train_data)

        train_data, train_label = flattened_train_data, norm_train_labels
        train_dataset = np.hstack([train_data, train_label[:, np.newaxis]])

        np.savetxt(self.base_dir + 'zT_train_dataset.csv', train_dataset, delimiter=',')
        np.savetxt(self.base_dir + 'zT_train_data.txt', train_data)
        np.savetxt(self.base_dir + 'zT_train_label.txt', train_label)

        print('...preprocessing done.')
