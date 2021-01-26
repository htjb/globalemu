import requests
import os
import numpy as np

class download():
    def __init__(self, xHI):
        self.xHI = xHI

    def model(self):

        if self.xHI is False:
            base_dir = 'best_T/'
        else:
            base_dir = 'best_xHI/'

        os.mkdir(base_dir)

        files = ['model.h5', 'data_mins.txt', 'data_maxs.txt', 'samples.txt',
            'AFB_norm_factor.npy', 'label_stds.npy']

        if self.xHI is False:
            base_url = \
                'https://raw.githubusercontent.com/' + \
                'htjb/emulator/master/best_T/'
        else:
            base_url = \
                'https://raw.githubusercontent.com/' + \
                'htjb/emulator/master/best_xHI/'

        for i in range(len(files)):
            if i > 3 and self.xHI is True:
                break
            r = requests.get(base_url + files[i])
            open(base_dir + files[i], 'wb').write(r.content)

        return base_dir

    def kappa(self):
        files = ['kappa_HH.txt']

        base_url = \
            'https://raw.githubusercontent.com/' + \
            'htjb/emulator/master/'

        r = requests.get(base_url + files)
        open(files, 'wb').write(r.content)