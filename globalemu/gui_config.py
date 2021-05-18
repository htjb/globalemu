import numpy as np
import pandas as pd
import os

class config():
    def __init__(self, base_dir, paramnames, data_dir, logs, xHI=False):
        self.base_dir = base_dir
        self.paramnames = paramnames
        self.logs = logs
        self.xHI = xHI

        test_data = np.loadtxt(data_dir + 'test_data.txt')
        test_labels = np.loadtxt(data_dir + 'test_labels.txt')
        for i in range(test_data.shape[1]):
            if i in logs:
                test_data[:, i] = np.log10(test_data[:, i])

        data_mins = test_data.min(axis=0)
        data_maxs = test_data.max(axis=0)

        full_logs = []
        for i in range(len(data_maxs)):
            if i in set(self.logs):
                full_logs.append(i)
            else:
                full_logs.append('--')

        df = pd.DataFrame({'names': self.paramnames,
                           'mins': data_mins,
                           'maxs': data_maxs,
                           'label_min':
                           [test_labels.min()] + ['']*(len(data_maxs)-1),
                           'label_max':
                           [test_labels.max()] + ['']*(len(data_maxs)-1),
                           'logs': full_logs,
                           'xHI':
                           [self.xHI] + ['']*(len(data_maxs)-1),})

        df.to_csv(base_dir + 'gui_configuration.csv', index=False)
