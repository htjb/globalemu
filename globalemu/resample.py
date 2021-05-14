import numpy as np


class sampling():

    r"""

    This function is used to resample the input training signals so that
    the regions of high variance in the training data are emphasised during
    training. The function finds the regions of high variance and
    builds a probability distribution from which to produce new redshift
    samples. See MNRAS preprint for details at
    https://arxiv.org/abs/2104.04336.

    **Parameters:**

    z: **list or np.array**
        | The original redshift data points.

    base_dir: **string**
        | The location of the trained model files. The samples and cdf are
            needed during evaluation of the network and consequently they
            are saved when this class is called.

    train_labels: **array**
        | The training global 21-cm or neutral fraction signals to resample
            at the new redshifts.

    """
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
