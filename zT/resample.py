import numpy as np
import matplotlib.pyplot as plt
from zT.cmSim import calc_signal

class sampling():
    def __init__(self, base_dir, **kwargs):
        self.base_dir = base_dir
        self.plot = kwargs.pop('plot', False)
        self.data_location = kwargs.pop('data_location', 'data/')

        orig_z = np.linspace(5, 50, 451)

        train_labels = np.loadtxt(self.data_location + 'train_labels.txt')

        res = calc_signal(orig_z, base_dir=self.base_dir)

        for i in range(len(train_labels)):
            train_labels[i] -= res.deltaT

        difference = []
        for i in range(train_labels.shape[1]):
            difference.append(train_labels[:, i].max() - train_labels[:, i].min())
        difference = np.array(difference)

        if self.plot is True:
            plt.plot(orig_z, difference)
            plt.xlabel('z')
            plt.ylabel(r'$\sigma$')
            plt.savefig(self.base_dir + 'difference.pdf')
            plt.close()

        difference = [difference[i]/np.sum(difference) for i in range(len(difference))]

        if self.plot is True:
            plt.plot(orig_z, difference)
            plt.xlabel('z')
            plt.ylabel(r'$\sigma$')
            plt.savefig(self.base_dir + 'difference_as_pdf.pdf')
            plt.close()

        cdf = np.cumsum(difference)

        if self.plot is True:
            plt.plot(orig_z, cdf)
            plt.xlabel('z')
            plt.ylabel('CDF')
            plt.savefig(self.base_dir + 'cdf.pdf')
            plt.close()

        x = np.linspace(0, 1, len(orig_z))

        self.samples = np.interp(x, cdf, orig_z)
        np.savetxt(self.base_dir + 'samples.txt', self.samples)

        tl = []
        for i in range(len(train_labels)):
            tl.append(np.interp(self.samples, orig_z, train_labels[i]))
        tl = np.array(tl)

        if self.plot is True:
            fig, axes = plt.subplots(2, 2, figsize=(10, 10))
            axes[0, 0].plot(orig_z, train_labels[0])
            axes[0, 0].plot(self.samples, tl[0], marker='.')
            axes[0, 1].plot(orig_z, train_labels[2700])
            axes[0, 1].plot(self.samples, tl[2700], marker='.')
            axes[1, 0].plot(orig_z, train_labels[26000])
            axes[1, 0].plot(self.samples, tl[26000], marker='.')
            axes[1, 1].plot(orig_z, train_labels[10000])
            axes[1, 1].plot(self.samples, tl[10000], marker='.')
            fig.add_subplot(111, frame_on=False)
            plt.tick_params(bottom=False, left=False, labelcolor='none')
            plt.xlabel('z')
            plt.ylabel(r'$\delta T$ [mk]')
            plt.tight_layout()
            plt.subplots_adjust(hspace=0, wspace=0)
            plt.savefig(self.base_dir + 'example_signals.pdf')
            plt.close()

            plt.figure()
            plt.plot(orig_z, cdf)
            plt.plot(self.samples, x, ls='', marker='.')
            plt.plot(self.samples, [-0.1]*len(self.samples),  ls='', marker='.')
            plt.xlabel('z', fontsize=12)
            plt.ylabel('CDF', fontsize=12)
            plt.savefig(self.base_dir + 'icdf.pdf')
            plt.close()

class samplingXHI():
    def __init__(self, base_dir, **kwargs):
        self.base_dir = base_dir
        self.plot = kwargs.pop('plot', False)
        self.data_location = kwargs.pop('data_location', 'data/')

        orig_z = np.hstack([np.arange(5, 15.1, 0.1), np.arange(16, 31, 1)])

        train_labels = np.loadtxt(self.data_location + 'train_labels.txt')

        difference = []
        for i in range(train_labels.shape[1]):
            difference.append(train_labels[:, i].max() - train_labels[:, i].min())
        difference=np.array(difference)

        if self.plot is True:
            plt.plot(orig_z, difference)
            plt.xlabel('z')
            plt.ylabel(r'$\sigma$')
            plt.savefig(self.base_dir + 'difference.pdf')
            plt.close()

        difference = [difference[i]/np.sum(difference) for i in range(len(difference))]

        if self.plot is True:
            plt.plot(orig_z, difference)
            plt.xlabel('z')
            plt.ylabel(r'$\sigma$')
            plt.savefig(self.base_dir + 'difference_as_pdf.pdf')
            plt.close()

        cdf = np.cumsum(difference)

        if self.plot is True:
            plt.plot(orig_z, cdf)
            plt.xlabel('z')
            plt.ylabel('CDF')
            plt.savefig(self.base_dir + 'cdf.pdf')
            plt.close()

        x = np.linspace(0, 1, len(orig_z))

        self.samples = np.interp(x, cdf, orig_z)
        np.savetxt(self.base_dir + 'samples.txt', self.samples)

        tl = []
        for i in range(len(train_labels)):
            tl.append(np.interp(self.samples, orig_z, train_labels[i]))
        tl = np.array(tl)

        if self.plot is True:
            fig, axes = plt.subplots(2, 2, figsize=(10, 10))
            axes[0, 0].plot(orig_z, train_labels[0])
            axes[0, 0].plot(self.samples, tl[0], marker='.')
            axes[0, 1].plot(orig_z, train_labels[2700])
            axes[0, 1].plot(self.samples, tl[2700], marker='.')
            axes[1, 0].plot(orig_z, train_labels[10046])
            axes[1, 0].plot(self.samples, tl[10046], marker='.')
            axes[1, 1].plot(orig_z, train_labels[9000])
            axes[1, 1].plot(self.samples, tl[9000], marker='.')
            fig.add_subplot(111, frame_on=False)
            plt.tick_params(bottom=False, left=False, labelcolor='none')
            plt.xlabel('z')
            plt.ylabel(r'$\delta T$ [mk]')
            plt.tight_layout()
            plt.subplots_adjust(hspace=0, wspace=0)
            plt.savefig(self.base_dir + 'example_signals.pdf')
            plt.close()

            plt.figure()
            plt.plot(orig_z, cdf)
            plt.plot(self.samples, x, ls='', marker='.')
            plt.plot(self.samples, [-0.1]*len(self.samples),  ls='', marker='.')
            plt.xlabel('z', fontsize=12)
            plt.ylabel('CDF', fontsize=12)
            plt.savefig(self.base_dir + 'icdf.pdf')
            plt.close()
