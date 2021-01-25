import numpy as np
import matplotlib.pyplot as plt
from zT.cmSim import calc_signal

class sampling():
    def __init__(self, base_dir, xHI, **kwargs):
        self.base_dir = base_dir
        self.xHI = xHI
        self.data_location = kwargs.pop('data_location', 'data/')

        if self.xHI is False:
            orig_z = np.linspace(5, 50, 451)
        else:
            orig_z = np.hstack([np.arange(5, 15.1, 0.1), np.arange(16, 31, 1)])

        train_labels = np.loadtxt(self.data_location + 'train_labels.txt')

        if self.xHI is False:
            res = calc_signal(orig_z, base_dir=self.base_dir)
            for i in range(len(train_labels)):
                train_labels[i] -= res.deltaT

        difference = []
        for i in range(train_labels.shape[1]):
            difference.append(train_labels[:, i].max() - train_labels[:, i].min())
        difference = np.array(difference)

        difference = [difference[i]/np.sum(difference) for i in range(len(difference))]

        cdf = np.cumsum(difference)

        x = np.linspace(0, 1, len(orig_z))

        self.samples = np.interp(x, cdf, orig_z)
        np.savetxt(self.base_dir + 'samples.txt', self.samples)
