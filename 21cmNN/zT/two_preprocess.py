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

        z = np.arange(5, 50.1, 0.1)
        z = (z - z.mean())/z.std()

        full_train_data = np.loadtxt('21cmGEM_data/Par_train_21cmGEM.txt')
        full_train_labels = np.loadtxt('21cmGEM_data/T21_train_21cmGEM.txt')

        ind = []
        for i in range(len(full_train_labels)):
            index = np.random.randint(0, len(full_train_labels))
            if index not in set(ind):
                ind.append(index)
            if len(ind) == self.num:
                break
        ind = np.array(ind)

        #import matplotlib.pyplot as plt

        train_data, train_labels = [], []
        for i in range(len(full_train_labels)):
            if np.any(ind == i):
                train_data.append(full_train_data[i, :])
                train_labels.append(full_train_labels[i])
        train_data, train_labels = np.array(train_data), np.array(train_labels)

        #for i in range(train_data.shape[1]):
        #    if i in set([0, 1]):
        #        train_data[:, i] = np.log10(train_data[:, i])

        labels_min = np.abs(train_labels.min())
        print('labels min', labels_min)
        #fig, axes = plt.subplots(3, 1, figsize=(5, 8))

        #for i in range(len(train_labels)):
        #    axes[0].plot(np.arange(5, 50.1, 0.1), train_labels[i])

        data_means = train_data.mean(axis=0)
        data_stds = train_data.std(axis=0)

        #data_abs_max = []
        #for i in range(train_data.shape[1]):
        #    data_abs_max.append(np.abs(train_data[:, i]).max())
        #data_abs_max = np.array(data_abs_max)

        norm_train_data = []
        for i in range(train_data.shape[1]):
            #norm_train_data.append(train_data[:, i]/data_abs_max[i])
            norm_train_data.append((train_data[:, i] - data_means[i])/data_stds[i])
        norm_train_data = np.array(norm_train_data).T

        norm_train_labels = []
        for i in range(train_labels.shape[0]):
            norm_train_labels.append(train_labels[i, :]/labels_min)
        norm_train_labels = np.array(norm_train_labels)

        #for i in range(len(norm_train_labels)):
        #    axes[1].plot(z, norm_train_labels[i])

        norm_train_labels = norm_train_labels.flatten()

        #for i in range(0, len(norm_train_labels), 451):
        #    axes[2].plot(z, norm_train_labels[i:i+451])
        #plt.show()
        #sys.exit(1)

        np.savetxt(self.base_dir + 'indices.txt', ind)
        #np.savetxt(self.base_dir + 'data_abs_max.txt', data_abs_max)
        np.savetxt(self.base_dir + 'data_means.txt', data_means)
        np.savetxt(self.base_dir + 'data_stds.txt', data_stds)
        np.save(self.base_dir + 'label_min.npy', labels_min)

        flattened_train_data = []
        for i in range(len(norm_train_data)):
            for j in range(len(z)):
                flattened_train_data.append(np.hstack([norm_train_data[i, :], z[j]]))
        flattened_train_data = np.array(flattened_train_data)

        #train_data, train_label = shuffle(flattened_train_data, norm_train_labels, random_state=0)
        train_data, train_label = flattened_train_data, norm_train_labels
        train_dataset = np.hstack([train_data, train_label[:, np.newaxis]])
        np.savetxt(self.base_dir + 'zT_train_dataset.csv', train_dataset, delimiter=',')
        np.savetxt(self.base_dir + 'zT_train_data.txt', train_data)
        np.savetxt(self.base_dir + 'zT_train_label.txt', train_label)

        test_data = np.loadtxt('21cmGEM_data/Par_test_21cmGEM.txt')
        test_labels = np.loadtxt('21cmGEM_data/T21_test_21cmGEM.txt')

        #test_data = [(test_data[:, i]/data_abs_max[i])
        #        for i in range(test_data.shape[1])]
        #for i in range(test_data.shape[1]):
        #    if i in set([0, 1]):
        #        test_data[:, i] = np.log10(test_data[:, i])

        test_data = [(test_data[:, i]-data_means[i])/data_stds[i]
                for i in range(test_data.shape[1])]

        norm_test_labels = []
        for i in range(test_labels.shape[0]):
            norm_test_labels.append(test_labels[i, :]/labels_min)

        test_data, test_labels = np.array(test_data).T, np.array(norm_test_labels)
        test_labels = test_labels.flatten()

        flattened_test_data = []
        for i in range(len(test_data)):
            for j in range(len(z)):
                flattened_test_data.append(np.hstack([test_data[i, :], z[j]]))
        flattened_test_data = np.array(flattened_test_data)

        #test_data, test_label = shuffle(flattened_test_data, test_labels, random_state=0)
        test_data, test_label = flattened_test_data, test_labels

        test_dataset = np.hstack([test_data, test_label[:, np.newaxis]])
        np.savetxt(self.base_dir + 'zT_test_dataset.csv', test_dataset, delimiter=',')
        np.savetxt(self.base_dir + 'zT_test_data.txt', test_data)
        np.savetxt(self.base_dir + 'zT_test_label.txt', test_label)

        print('...preprocessing done.')
