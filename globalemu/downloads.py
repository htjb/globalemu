"""
``download()`` can be used to download the released trained models
for both the global signal and neutral fraction history emulators.
"""

import requests
import os


class download():

    r"""

    **Parameters:**

        xHI: **Bool / default: False**
            | Setting this equal to ``True`` will cause the method ``model()``
                to download the released neutal fraction history model rather
                than the released global signal network.

    """

    def __init__(self, xHI=False):
        self.xHI = xHI
        if type(self.xHI) is not bool:
            raise TypeError("'xHI' must be a bool.")

    def model(self):

        if self.xHI is False:
            base_dir = 'T_release/'
        else:
            base_dir = 'xHI_release/'

        os.mkdir(base_dir)

        files = [
            'model.h5', 'data_mins.txt', 'data_maxs.txt', 'samples.txt',
            'cdf.txt', 'z.txt', 'kwargs.txt',
            'AFB_norm_factor.npy', 'labels_stds.npy', 'AFB.txt']

        if self.xHI is False:
            base_url = \
                'https://raw.githubusercontent.com/' + \
                'htjb/globalemu/master/T_release/'
        else:
            base_url = \
                'https://raw.githubusercontent.com/' + \
                'htjb/globalemu/master/xHI_release/'

        for i in range(len(files)):
            if i > 6 and self.xHI is True:
                break
            r = requests.get(base_url + files[i])
            open(base_dir + files[i], 'wb').write(r.content)

        return base_dir

    def kappa(self):
        base_url = \
            'https://raw.githubusercontent.com/' + \
            'htjb/globalemu/master/kappa_HH.txt'

        r = requests.get(base_url)
        open('kappa_HH.txt', 'wb').write(r.content)
