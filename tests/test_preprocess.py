import numpy as np
from globalemu.preprocess import process
import requests, zipfile, io
import os
import pytest
import pandas as pd

def test_process_nn():
    r = requests.get('https://people.ast.cam.ac.uk/~afialkov/21cmGEM_data.zip')
    z = zipfile.ZipFile(io.BytesIO(r.content))
    loc = 'data_download/'
    z.extractall('data_download/')
    os.rename(loc + 'Par_test_21cmGEM.txt', loc + 'test_data.txt')
    os.rename(loc + 'Par_train_21cmGEM.txt', loc + 'train_data.txt')
    os.rename(loc + 'T21_test_21cmGEM.txt', loc + 'test_labels.txt')
    os.rename(loc + 'T21_train_21cmGEM.txt', loc + 'train_labels.txt')
    z = np.arange(5, 50.1, 0.1)

    process(10, z, data_location='data_download/')

    files = ['AFB_norm_factor.npy', 'AFB.txt', 'cdf.txt', 'data_maxs.txt',
             'data_mins.txt', 'indices.txt', 'labels_stds.npy', 'samples.txt',
             'train_data.txt', 'train_dataset.csv', 'train_label.txt', 'z.txt']

    for i in range(len(files)):
        assert(os.path.exists('model_dir/' + files[i]) is True)

    full_train_data = pd.read_csv(
        'model_dir/train_dataset.csv', header=None).values

    for i in range(full_train_data.shape[1]):
        if i < full_train_data.shape[1] - 1:
            assert(full_train_data[:, i].min() == 0)
            assert(full_train_data[:, i].max() == 1)
