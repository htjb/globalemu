import numpy as np
from globalemu.preprocess import process
from globalemu.network import nn
import requests
import os
import shutil
import pytest


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
    nn(batch_size=451, layer_sizes=[8], epochs=10)

    # results of below will not make sense as it is being run on the
    # global signal data but it will test the code (xHI data not public)
    process(10, z, data_location='21cmGEM_data/', xHI=True)
    nn(batch_size=451, layer_sizes=[8], epochs=5, xHI=True)

    # test early_stop code
    nn(batch_size=451, layer_sizes=[], epochs=20, early_stop=True)

    with pytest.raises(KeyError):
        process(10, z, datalocation='21cmGEM_data/')

    with pytest.raises(KeyError):
        nn(batch_size=451, layersizes=[8], epochs=10)

    with pytest.raises(TypeError):
        nn(batch_size='foo')
    with pytest.raises(TypeError):
        nn(activation=10)
    with pytest.raises(TypeError):
        nn(epochs=False)
    with pytest.raises(TypeError):
        nn(lr='bar')
    with pytest.raises(TypeError):
        nn(dropout=True)
    with pytest.raises(TypeError):
        nn(input_shape='foo')
    with pytest.raises(TypeError):
        nn(output_shape='foobar')
    with pytest.raises(TypeError):
        nn(layer_sizes=10)
    with pytest.raises(TypeError):
        nn(base_dir=50)
    with pytest.raises(KeyError):
        nn(base_dir='dir')
    with pytest.raises(TypeError):
        nn(early_stop='foo')
    with pytest.raises(TypeError):
        nn(early_stop_lim=False)
    with pytest.raises(TypeError):
        nn(xHI='false')
    with pytest.raises(TypeError):
        nn(resume=10)

    process(10, z, data_location='21cmGEM_data/', base_dir='base_dir/')
    nn(batch_size=451, layer_sizes=[], random_seed=10,
       base_dir='base_dir/')

    dir = ['21cmGEM_data/', 'model_dir/', 'base_dir/']
    for i in range(len(dir)):
        if os.path.exists(dir[i]):
            shutil.rmtree(dir[i])
