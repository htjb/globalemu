import numpy as np
from sklearn.utils import shuffle
import os

class process():
    def __init__(self, num, **kwargs):
        print('Preprocessing started...')
        self.num = num
        self.base_dir = kwargs.pop('base_dir', 'results/')

        if not os.path.exists(self.base_dir):
            os.mkdir(self.base_dir)

        def remove_zeros(y):
            ind = []
            y_new = []
            for i in range(y.shape[1]):
                if np.all(y[:, i] == y[0, i]):
                    ind.append(i)
                else:
                    y_new.append(y[:, i])
            y_new = np.array(y_new)
            return ind, y_new.T
        nComp=20

        z = np.arange(5, 50.1, 0.1)
        z = (z - z.mean())/z.std()

        full_train_data = np.loadtxt('21cmGEM_data/Par_train_21cmGEM.txt')
        full_train_labels = np.loadtxt('21cmGEM_data/T21_train_21cmGEM.txt')

        ftd_means = full_train_data.mean(axis=0)
        ftd_stds = full_train_data.std(axis=0)
        # print(ftd_stds, ftd_means)

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

        data_means = train_data.mean(axis=0)
        data_stds = train_data.std(axis=0)

        for i in range(train_data.shape[1]):
            train_data[:, i] = (train_data[:, i] - data_means[i])/data_stds[i]

        ind, train_labels = remove_zeros(train_labels)

        label_means = train_labels.mean(axis=0)
        label_stds = train_labels.std(axis=0)

        for i in range(train_data.shape[1]):
            train_labels[:, i] = (train_labels[:, i] - label_means[i])/label_stds[i]
        np.savetxt(self.base_dir + 'train_labels_preflatten.txt', train_labels)
        train_labels = train_labels.flatten()

        np.savetxt(self.base_dir + 'indices.txt', ind)
        np.savetxt(self.base_dir + 'data_means.txt', data_means)
        np.savetxt(self.base_dir + 'data_stds.txt', data_stds)
        np.savetxt(self.base_dir + 'label_means.txt', label_means)
        np.savetxt(self.base_dir + 'label_stds.txt', label_stds)

        flattened_train_data = []
        for i in range(len(train_data)):
            for j in range(len(z)):
                if j not in set(ind):
                    flattened_train_data.append(np.hstack([train_data[i, :], z[j]]))
        flattened_train_data = np.array(flattened_train_data)

        train_data, train_label = shuffle(flattened_train_data, train_labels, random_state=0)

        train_dataset = np.hstack([train_data, train_labels[:, np.newaxis]])
        np.savetxt(self.base_dir + 'zT_train_dataset.csv', train_dataset, delimiter=',')
        np.savetxt(self.base_dir + 'zT_train_data.txt', train_data)
        np.savetxt(self.base_dir + 'zT_train_label.txt', train_label)

        test_data = np.loadtxt('21cmGEM_data/Par_test_21cmGEM.txt')
        test_labels = np.loadtxt('21cmGEM_data/T21_test_21cmGEM.txt')

        test_data = [(test_data[:, i] - data_means[i])/data_stds[i]
                for i in range(test_data.shape[1])]
        tests = []
        for i in range(test_labels.shape[1]):
            if i in set(ind):
                pass
            else:
                tests.append(test_labels[:, i])
        test_labels = np.array(tests).T

        test_labels = [(test_labels[:, i] - label_means[i])/label_stds[i]
                for i in range(test_labels.shape[1])]
        test_data, test_labels = np.array(test_data).T, np.array(test_labels)
        test_labels = test_labels.flatten()

        flattened_test_data = []
        for i in range(len(test_data)):
            for j in range(len(z)):
                if j not in set(ind):
                    flattened_test_data.append(np.hstack([test_data[i, :], z[j]]))
        flattened_test_data = np.array(flattened_test_data)

        test_data, test_label = shuffle(flattened_test_data, test_labels, random_state=0)

        test_dataset = np.hstack([test_data, test_labels[:, np.newaxis]])
        np.savetxt(self.base_dir + 'zT_test_dataset.csv', test_dataset, delimiter=',')
        np.savetxt(self.base_dir + 'zT_test_data.txt', test_data)
        np.savetxt(self.base_dir + 'zT_test_label.txt', test_label)

        print('...preprocessing done.')
