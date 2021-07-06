import numpy as np
from globalemu.preprocess import process
import requests
import os
import pytest
import pandas as pd
import shutil


def test_preprocess():
    z = np.arange(5, 50.1, 0.1)

    process(10, z, data_location='21cmGEM_data/')
    process(10, z, data_location='21cmGEM_data/', xHI=True)
    process('full', z, data_location='21cmGEM_data/')
    process('full', z, data_location='21cmGEM_data/', AFB=False)
    process(10, z, data_location='21cmGEM_data/', resampling=False)

    files = ['AFB_norm_factor.npy', 'AFB.txt', 'cdf.txt', 'data_maxs.txt',
             'data_mins.txt', 'indices.txt', 'labels_stds.npy', 'samples.txt',
             'train_data.txt', 'train_dataset.csv', 'train_label.txt', 'z.txt',
             'preprocess_settings.pkl']

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
        process(10, z, data_location='21cmGEM_data/', logs=True)
    with pytest.raises(TypeError):
        process(10, z, data_location='21cmGEM_data/', AFB=10)
    with pytest.raises(TypeError):
        process(10, z, data_location='21cmGEM_data/', resampling=10)
    with pytest.raises(TypeError):
        process(10, z, data_location='21cmGEM_data/', resampling=10)

    dir = ['21cmGEM_data/', 'model_dir/']
    for i in range(len(dir)):
        if os.path.exists(dir[i]):
            shutil.rmtree(dir[i])
