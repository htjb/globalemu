import numpy as np


class sampling():
    def __init__(self, z, base_dir, train_labels):
        self.z = z
        self.base_dir = base_dir
        self.train_labels = train_labels

        difference = []
        for i in range(self.train_labels.shape[1]):
            difference.append(
                self.train_labels[:, i].max() - self.train_labels[:, i].min())
        difference = np.array(difference)

        difference = [
            difference[i]/np.sum(difference) for i in range(len(difference))]

        self.cdf = np.cumsum(difference)

        x = np.linspace(0, 1, len(self.z))

        self.samples = np.interp(x, self.cdf, self.z)
        np.savetxt(self.base_dir + 'samples.txt', self.samples)
        np.savetxt(self.base_dir + 'cdf.txt', self.cdf)
