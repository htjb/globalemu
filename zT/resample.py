import numpy as np
import matplotlib.pyplot as plt
from zT.cmSim import calc_signal

class sampling():
    def __init__(self, base_dir, **kwargs):
        self.base_dir = base_dir
        self.plot = kwargs.pop('plot', False)

        orig_z = np.linspace(5, 50, 451)

        train_labels = np.loadtxt('Resplit_data/train_labels.txt')

        res = calc_signal(orig_z, base_dir=self.base_dir)

        for i in range(len(train_labels)):
            train_labels[i] -= res.deltaT

        stds = []
        for i in range(train_labels.shape[1]):
            stds.append(train_labels[:, i].max() - train_labels[:, i].min())
            #stds.append(train_labels[:, i].std())
        stds=np.array(stds)

        if self.plot is True:
            plt.plot(orig_z, stds)
            plt.xlabel('z')
            plt.ylabel(r'$\sigma$')
            plt.savefig(self.base_dir + 'stds.pdf')
            plt.close()

        stds = [stds[i]/np.sum(stds) for i in range(len(stds))]

        if self.plot is True:
            plt.plot(orig_z, stds)
            plt.xlabel('z')
            plt.ylabel(r'$\sigma$')
            plt.savefig(self.base_dir + 'stds_as_pdf.pdf')
            plt.close()

        cdf = []
        for i in range(len(stds)):
            cdf.append(np.sum(stds[:i]))
        cdf = np.array(cdf)

        if self.plot is True:
            plt.plot(orig_z, cdf)
            plt.xlabel('z')
            plt.ylabel('CDF')
            plt.savefig(self.base_dir + 'cdf.pdf')
            plt.close()

        norm_z = (orig_z - orig_z.min())/(orig_z.max() - orig_z.min())

        samples = []
        for r in norm_z:
            if r == 1:
                samples.append(orig_z.max())
            else:
                samples.append(orig_z[np.argwhere(cdf == min(cdf[(cdf - r) >= 0]))][0][0])
        self.samples = np.array(samples)
        np.savetxt(self.base_dir + 'samples.txt', self.samples)

        tl = []
        for i in range(len(train_labels)):
            tl.append(np.interp(samples, orig_z, train_labels[i]))
        tl = np.array(tl)

        if self.plot is True:
            fig, axes = plt.subplots(2, 2, figsize=(10, 10))
            axes[0, 0].plot(orig_z, train_labels[0])
            axes[0, 0].plot(samples, tl[0], marker='.')
            axes[0, 1].plot(orig_z, train_labels[2700])
            axes[0, 1].plot(samples, tl[2700], marker='.')
            axes[1, 0].plot(orig_z, train_labels[26000])
            axes[1, 0].plot(samples, tl[26000], marker='.')
            axes[1, 1].plot(orig_z, train_labels[10000])
            axes[1, 1].plot(samples, tl[10000], marker='.')
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
            plt.plot(samples, norm_z, ls='', marker='.')
            plt.plot(samples, [-0.1]*len(samples),  ls='', marker='.')
            plt.xlabel('z', fontsize=12)
            plt.ylabel('CDF', fontsize=12)
            plt.savefig(self.base_dir + 'icdf.pdf')
            plt.close()
