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
import pickle
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

        AFB: **Bool / default: None**
            | If True then ``globalemu`` will calculate an astrophysics free
                baseline and subtract this from the training data signals.
                The AFB is specific to the global 21-cm signal and as
                ``globalemu`` is set up to emulate the global signal by
                default this parameter is set to True. If xHI is True then
                AFB is set to False by default.

        std_division: **Bool / default: None**
            | If True then ``globalemu`` will divide the training data by the
                standard deviation across the training data. This is
                recommended when building an emulator to emulate the global
                signal and is set to True by default. If xHI is True then
                std_division is set to False by default.

        resampling: **Bool / default: None**
            | Controls whether or not the signals will be resampled with
                higher sampling at regions of large variation in the training
                data set or not. Set to True by default as this is advised for
                training both neutral fraction and global signal emulators.

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
                    ['base_dir', 'data_location', 'xHI', 'logs', 'AFB',
                     'std_division', 'resampling']):
                raise KeyError("Unexpected keyword argument in process()")

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
        if self.xHI is False:
            self.preprocess_settings = {'AFB': True, 'std_division': True,
                                        'resampling': True}
        else:
            self.preprocess_settings = {'AFB': False, 'std_division': False,
                                        'resampling': True}

        preprocess_settings_keys = ['AFB', 'std_division', 'resampling']
        for key in preprocess_settings_keys:
            if key in kwargs:
                self.preprocess_settings[key] = kwargs[key]

        bool_kwargs = [self.xHI, self.preprocess_settings['AFB'],
                       self.preprocess_settings['std_division'],
                       self.preprocess_settings['resampling']]
        bool_strings = ['xHI', 'AFB', 'std_division', 'resampling']
        for i in range(len(bool_kwargs)):
            if type(bool_kwargs[i]) is not bool:
                raise TypeError(bool_strings[i] + " must be a bool.")

        self.logs = kwargs.pop('logs', [0, 1, 2])
        if type(self.logs) is not list:
            raise TypeError("'logs' must be a list.")

        if not os.path.exists(self.base_dir):
            os.mkdir(self.base_dir)

        file = open(self.base_dir + "preprocess_settings.pkl", "wb")
        pickle.dump(self.preprocess_settings, file)
        file.close()

        np.savetxt(self.base_dir + 'z.txt', self.z)

        def load_data(file):
            return pd.read_csv(
                self.data_location + file,
                delim_whitespace=True, header=None).values

        full_train_data = load_data('train_data.txt')
        full_train_labels = load_data('train_labels.txt')
        full_test_data = load_data('test_data.txt')
        test_labels = load_data('test_labels.txt')

        if self.preprocess_settings['AFB'] is True:
            np.save(
                self.base_dir + 'AFB_norm_factor.npy',
                full_train_labels[0, -1]*1e-3)
            res = calc_signal(self.z, self.base_dir)

        if self.num == 'full':
            train_data = full_train_data.copy()
            if self.preprocess_settings['AFB'] is True:
                train_labels = full_train_labels.copy() - res.deltaT
                test_labels -= res.deltaT
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
                    if self.preprocess_settings['AFB'] is True:
                        train_labels.append(full_train_labels[i] - res.deltaT)
                    else:
                        train_labels.append(full_train_labels[i])
            train_data, train_labels = np.array(train_data), \
                np.array(train_labels)

        log_train_data = []
        for i in range(train_data.shape[1]):
            if i in set(self.logs):
                for j in range(train_data.shape[0]):
                    if train_data[j, i] == 0:
                        train_data[j, i] = 1e-6
                log_train_data.append(np.log10(train_data[:, i]))
            else:
                log_train_data.append(train_data[:, i])
        train_data = np.array(log_train_data).T

        log_test_data = []
        for i in range(full_test_data.shape[1]):
            if i in set(self.logs):
                for j in range(full_test_data.shape[0]):
                    if full_test_data[j, i] == 0:
                        full_test_data[j, i] = 1e-6
                log_test_data.append(np.log10(full_test_data[:, i]))
            else:
                log_test_data.append(full_test_data[:, i])
        test_data = np.array(log_test_data).T

        if self.preprocess_settings['resampling'] is True:
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

            resampled_test_labels = []
            for i in range(len(test_labels)):
                resampled_test_labels.append(
                    np.interp(samples, self.z, test_labels[i]))
            test_labels = np.array(resampled_test_labels)
        else:
            norm_s = (self.z - self.z.min())/(self.z.max() - self.z.min())

        train_data_mins = train_data.min(axis=0)
        train_data_maxs = train_data.max(axis=0)

        test_data_mins = test_data.min(axis=0)
        test_data_maxs = test_data.max(axis=0)

        norm_train_data = []
        for i in range(train_data.shape[1]):
            norm_train_data.append(
                (train_data[:, i] - train_data_mins[i])/ \
                    (train_data_maxs[i]-train_data_mins[i]))
        norm_train_data = np.array(norm_train_data).T

        norm_test_data = []
        for i in range(test_data.shape[1]):
            norm_test_data.append(
                (test_data[:, i] - test_data_mins[i])/ \
                    (test_data_maxs[i]-test_data_mins[i]))
        norm_test_data = np.array(norm_test_data).T

        if self.preprocess_settings['std_division'] is True:
            labels_stds = train_labels.std()
            norm_train_labels = [
                train_labels[i, :]/labels_stds
                for i in range(train_labels.shape[0])]
            norm_train_labels = np.array(norm_train_labels)

            norm_train_labels = norm_train_labels.flatten()
            np.save(self.base_dir + 'labels_stds.npy', labels_stds)

            test_labels_stds = test_labels.std()
            norm_test_labels = [
                test_labels[i, :]/test_labels_stds
                for i in range(test_labels.shape[0])]
            norm_test_labels = np.array(norm_test_labels)

            norm_test_labels = norm_test_labels.flatten()

        else:
            norm_train_labels = train_labels.flatten()
            norm_test_labels = test_labels.flatten()

        if self.num != 'full':
            np.savetxt(self.base_dir + 'indices.txt', ind)
        np.savetxt(self.base_dir + 'data_mins.txt', train_data_mins)
        np.savetxt(self.base_dir + 'data_maxs.txt', train_data_maxs)

        flattened_train_data = []
        for i in range(len(norm_train_data)):
            for j in range(len(norm_s)):
                flattened_train_data.append(
                    np.hstack([norm_train_data[i, :], norm_s[j]]))
        flattened_train_data = np.array(flattened_train_data)

        flattened_test_data = []
        for i in range(len(norm_test_data)):
            for j in range(len(norm_s)):
                flattened_test_data.append(
                    np.hstack([norm_test_data[i, :], norm_s[j]]))
        flattened_test_data = np.array(flattened_test_data)

        train_dataset = np.hstack([flattened_train_data, 
                                   norm_train_labels[:, np.newaxis]])

        np.savetxt(
            self.base_dir + 'train_dataset.csv', train_dataset, delimiter=',')
        np.savetxt(self.base_dir + 'train_data.txt', flattened_train_data)
        np.savetxt(self.base_dir + 'train_label.txt', norm_train_labels)
        np.savetxt(self.base_dir + 'test_data.txt', flattened_test_data)
        np.savetxt(self.base_dir + 'test_label.txt', norm_test_labels)

        print('...preprocessing done.')
