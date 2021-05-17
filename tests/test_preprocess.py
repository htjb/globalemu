import numpy as np
from globalemu.preprocess import process
import requests, zipfile, io
import os
import pytest
import pandas as pd
import shutil

def download_21cmGEM_data():
    data_dir = '21cmGEM_data/'
    if not os.path.exists(data_dir):
      os.mkdir(data_dir)

    files = ['Par_test_21cmGEM.txt',
             'Par_train_21cmGEM.txt',
             'T21_test_21cmGEM.txt',
             'T21_train_21cmGEM.txt']
    saves = ['test_data.txt',
             'train_data.txt',
             'test_labels.txt',
             'train_labels.txt']

    for i in range(len(files)):
      url = 'https://zenodo.org/record/4541500/files/' + files[i]
      with open(data_dir + saves[i], 'wb') as f:
          f.write(requests.get(url).content)

def test_process_nn():
    download_21cmGEM_data()
    z = np.arange(5, 50.1, 0.1)

    process(10, z, data_location='21cmGEM_data/')

    files = ['AFB_norm_factor.npy', 'AFB.txt', 'cdf.txt', 'data_maxs.txt',
             'data_mins.txt', 'indices.txt', 'labels_stds.npy', 'samples.txt',
             'train_data.txt', 'train_dataset.csv', 'train_label.txt', 'z.txt']

    for i in range(len(files)):
        assert(os.path.exists('model_dir/' + files[i]) is True)

    full_train_data = pd.read_csv(
        'model_dir/train_dataset.csv', header=None).values

    for i in range(full_train_data.shape[1]):
        if i < full_train_data.shape[1] - 1:
            assert(np.isclose(full_train_data[:, i].min(),
                   0, rtol=1e-1, atol=1e-1))
            assert(np.isclose(full_train_data[:, i].max(),
                   1, rtol=1e-1, atol=1e-1))

    with pytest.raises(TypeError):
        process(10.2, z, data_location='21cmGEM_data/')
    with pytest.raises(TypeError):
        process(10, 10, data_location='21cmGEM_data/')
    with pytest.raises(KeyError):
        process(10, z, data_location='data_download')
    with pytest.raises(TypeError):
        process(10, z, data_location='21cmGEM_data/', base_dir=10)
    with pytest.raises(TypeError):
        process(10, z, data_location='21cmGEM_data/', xHI=10)
    with pytest.raises(TypeError):
        process(10, z, data_location='21cmGEM_data/', logs=True)

    if os.path.exists('21cmGEM_data/'):
        shutil.rmtree('21cmGEM_data/')
