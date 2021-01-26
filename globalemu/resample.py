import numpy as np
from globalemu.cmSim import calc_signal

class sampling():
    def __init__(self, z, base_dir, xHI, **kwargs):
        self.z = z
        self.base_dir = base_dir
        self.xHI = xHI
        self.data_location = kwargs.pop('data_location', 'data/')

        train_labels = np.loadtxt(self.data_location + 'train_labels.txt')

        if self.xHI is False:
            res = calc_signal(self.z, self.base_dir)
            for i in range(len(train_labels)):
                train_labels[i] -= res.deltaT

        difference = []
        for i in range(train_labels.shape[1]):
            difference.append(train_labels[:, i].max() - train_labels[:, i].min())
        difference = np.array(difference)

        difference = [difference[i]/np.sum(difference) for i in range(len(difference))]

        cdf = np.cumsum(difference)

        x = np.linspace(0, 1, len(self.z))

        self.samples = np.interp(x, cdf, self.z)
        np.savetxt(self.base_dir + 'samples.txt', self.samples)
