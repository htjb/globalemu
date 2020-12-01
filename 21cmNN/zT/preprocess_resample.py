import numpy as np
from sklearn.utils import shuffle
import os
from zT.cmSim import calc_signal
from zT.resample import sampling

class process():
    def __init__(self, num, **kwargs):
        print('Preprocessing started...')
        self.num = num
        self.base_dir = kwargs.pop('base_dir', 'results/')

        if not os.path.exists(self.base_dir):
            os.mkdir(self.base_dir)

        orig_z = np.linspace(5, 50, 451)

        full_train_data = np.loadtxt('Resplit_data/train_data.txt')
        full_train_labels = np.loadtxt('Resplit_data/train_labels.txt')
        np.save(self.base_dir + 'AFB_norm_factor.npy', full_train_labels[0, -1])

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

        """for i in range(len(train_labels)):
            plt.plot(orig_z, train_labels[i, :])
            if i == 0:
                print(train_labels[i, :])
        plt.show()
        sys.exit(1)"""

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

        samples = sampling(self.base_dir).samples
        resampled_labels = []
        for i in range(len(train_labels)):
            resampled_labels.append(np.interp(samples, orig_z, train_labels[i]))
        train_labels = np.array(resampled_labels)

        norm_s = (samples.copy() - samples.min())/(samples.max()-samples.min())
        #ls = np.log10(samples)
        #norm_s = (ls.copy() - ls.min())/(ls.max()-ls.min())

        #labels_min = train_labels.min()
        #labels_max = train_labels.max()
        labels_means = train_labels.mean()
        labels_stds = train_labels.std()

        #fig, axes = plt.subplots(3, 1, figsize=(5, 8))

        #for i in range(len(train_labels)):
        #    axes[0].plot(np.arange(5, 50.1, 0.1), train_labels[i])

        #data_means = train_data.mean(axis=0)
        #data_stds = train_data.std(axis=0)
        data_mins = train_data.min(axis=0)
        data_maxs = train_data.max(axis=0)

        norm_train_data = []
        for i in range(train_data.shape[1]):
            #norm_train_data.append(train_data[:, i]/data_abs_max[i])
            #norm_train_data.append((train_data[:, i] - data_means[i])/data_stds[i])
            norm_train_data.append((train_data[:, i] - data_mins[i])/(data_maxs[i]-data_mins[i]))
        norm_train_data = np.array(norm_train_data).T

        norm_train_labels = []
        for i in range(train_labels.shape[0]):
            norm_train_labels.append((train_labels[i, :]- labels_means)/labels_stds)
            #norm_train_labels.append((train_labels[i, :]- labels_min)/(labels_max-labels_min))
        norm_train_labels = np.array(norm_train_labels)
        #print(norm_train_labels.shape)
        #sys.exit(1)

        #for i in range(len(norm_train_labels)):
        #    axes[1].plot(z, norm_train_labels[i])

        norm_train_labels = norm_train_labels.flatten()
        print(norm_train_labels.shape)
        #sys.exit(1)

        #for i in range(0, len(norm_train_labels), 451):
        #    axes[2].plot(z, norm_train_labels[i:i+451])
        #plt.show()
        #sys.exit(1)
        if self.num != 'full':
            np.savetxt(self.base_dir + 'indices.txt', ind)
        #np.savetxt(self.base_dir + 'data_abs_max.txt', data_abs_max)
        #np.savetxt(self.base_dir + 'data_means.txt', data_means)
        #np.savetxt(self.base_dir + 'data_stds.txt', data_stds)
        #np.save(self.base_dir + 'label_min.npy', labels_min)
        #np.save(self.base_dir + 'label_max.npy', labels_max)
        np.save(self.base_dir + 'labels_means.npy', labels_means)
        np.save(self.base_dir + 'labels_stds.npy', labels_stds)
        np.savetxt(self.base_dir + 'data_mins.txt', data_mins)
        np.savetxt(self.base_dir + 'data_maxs.txt', data_maxs)

        flattened_train_data = []
        for i in range(len(norm_train_data)):
            for j in range(len(norm_s)):
                flattened_train_data.append(np.hstack([norm_train_data[i, :], norm_s[j]]))
        flattened_train_data = np.array(flattened_train_data)

        #train_data, train_label = shuffle(flattened_train_data, norm_train_labels, random_state=0)
        train_data, train_label = flattened_train_data, norm_train_labels
        train_dataset = np.hstack([train_data, train_label[:, np.newaxis]])

        np.savetxt(self.base_dir + 'zT_train_dataset.csv', train_dataset, delimiter=',')
        np.savetxt(self.base_dir + 'zT_train_data.txt', train_data)
        np.savetxt(self.base_dir + 'zT_train_label.txt', train_label)

        print('...preprocessing done.')
