"""

``process()`` is used to preprocess the data in the provided directory
using the techniques outlined in the ``globalemu`` paper. For ``process()``
to work it requires the testing and training data to be saved in the
``data_location`` directory in a specific manner. The "labels" or temperatures
(network outputs) should be saved
as "test_labels.txt"/"train_labels.txt" and the "data" or
astrophysical parameters (network inputs excluding redshift) as
"test_data.txt"/"train_data.txt".

"""

import numpy as np
import os
import pandas as pd
from globalemu.cmSim import calc_signal
from globalemu.resample import sampling


class process():

    r"""

    **Parameters:**

        num: **int**
            | The number of models that will be used to train globalemu. If
                you wish to use the full training data set then set
                ``num = 'full'``.

        z: **np.array**
            | The redshift range that corresponds to the models in the saved
                "test_labels.txt" and "train_labels.txt" e.g. for the
                ``21cmGEM`` data this would be :code:`np.arange(5, 50.1, 0.1)`.

    **kwargs:**

        base_dir: **string / default: 'model_dir/'**
            | The ``base_dir`` is where the preprocessed data and later the
                trained models will be placed. This should be thought of as the
                working directory as it will be needed when training a model
                and making evaluations of trained models.

        data_location: **string / default: 'data/'**
            | As discussed above the ``data_loaction`` is where the data to be
                processed is to be found. It must be accurately provided for
                the code to work and must end in a '/'.

        xHI: **Bool / default: False**
            | If True then ``globalemu`` will act as if it is training a
                neutral fraction history emulator.

        logs: **list / default: [0, 1, 2]**
            | The indices corresponding to the astrophysical parameters in
                "train_data.txt" that need to be logged. The default assumes
                that the first three columns in "train_data.txt" are
                :math:`{f_*}` (star formation efficiency),
                :math:`{V_c}` (minimum virial circular velocity) and
                :math:`{f_x}` (X-ray efficieny).
    """

    def __init__(self, num, z, **kwargs):

        print('Preprocessing started...')

        for key, values in kwargs.items():
            if key not in set(
                    ['base_dir', 'data_location', 'xHI', 'logs']):
                raise KeyError("Unexpected keyward argument in process()")

        self.num = num
        if type(self.num) is not int:
            if self.num != 'full':
                raise TypeError("'num' must be an integer or 'full'.")

        self.z = z
        if type(self.z) not in set([np.ndarray, list]):
            raise TypeError("'z' should be a numpy array or list.")

        self.base_dir = kwargs.pop('base_dir', 'model_dir/')
        self.data_location = kwargs.pop('data_location', 'data/')

        file_kwargs = [self.base_dir, self.data_location]
        file_strings = ['base_dir', 'data_location']
        for i in range(len(file_kwargs)):
            if type(file_kwargs[i]) is not str:
                raise TypeError("'" + file_strings[i] + "' must be a sting.")
            elif file_kwargs[i].endswith('/') is False:
                raise KeyError("'" + file_strings[i] + "' must end with '/'.")

        self.xHI = kwargs.pop('xHI', False)
        if type(self.xHI) is not bool:
            raise TypeError("'xHI' must be a bool.")

        self.logs = kwargs.pop('logs', [0, 1, 2])
        if type(self.logs) is not list:
            raise TypeError("'logs' must be a list.")

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

        sampling_call = sampling(
            self.z, self.base_dir, train_labels)
        samples = sampling_call.samples
        cdf = sampling_call.cdf

        resampled_labels = []
        for i in range(len(train_labels)):
            resampled_labels.append(
                np.interp(samples, self.z, train_labels[i]))
        train_labels = np.array(resampled_labels)

        norm_s = np.interp(samples, self.z, cdf)

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
