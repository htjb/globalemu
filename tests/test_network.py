import numpy as np
from globalemu.preprocess import process
from globalemu.network import nn
import requests, zipfile, io
import os
import pytest

def test_process_nn():
    r = requests.get('https://people.ast.cam.ac.uk/~afialkov/21cmGEM_data.zip')
    zip = zipfile.ZipFile(io.BytesIO(r.content))
    loc = 'data_download/'
    zip.extractall('data_download/')
    os.rename(loc + 'Par_test_21cmGEM.txt', loc + 'test_data.txt')
    os.rename(loc + 'Par_train_21cmGEM.txt', loc + 'train_data.txt')
    os.rename(loc + 'T21_test_21cmGEM.txt', loc + 'test_labels.txt')
    os.rename(loc + 'T21_train_21cmGEM.txt', loc + 'train_labels.txt')
    z = np.arange(5, 50.1, 0.1)

    process(10, z, data_location='data_download/')
    nn(batch_size=451, layer_sizes=[8], epochs=10)

    # results of below will not make sense as it is being run on the
    # global signal data but it will test the code (xHI data not public)
    process(10, z, data_location='data_download/', xHI=True)
    nn(batch_size=451, layer_sizes=[8], epochs=5, xHI=True)

    # test early_stop code
    nn(batch_size=451, layer_sizes=[], epochs=20, early_stop=True)

    with pytest.raises(KeyError):
        process(10, z, datalocation='data_download/')

    with pytest.raises(KeyError):
        nn(batch_size=451, layersizes=[8], epochs=10)

    with pytest.raises(TypeError):
        res = nn(batch_size='foo')
    with pytest.raises(TypeError):
        res = nn(activation=10)
    with pytest.raises(TypeError):
        res = nn(epochs=False)
    with pytest.raises(TypeError):
        res = nn(lr='bar')
    with pytest.raises(TypeError):
        res = nn(dropout=True)
    with pytest.raises(TypeError):
        res = nn(input_shape='foo')
    with pytest.raises(TypeError):
        res = nn(output_shape='foobar')
    with pytest.raises(TypeError):
        res = nn(layer_sizes=10)
    with pytest.raises(TypeError):
        res = nn(base_dir=50)
    with pytest.raises(KeyError):
        res = nn(base_dir='dir')
    with pytest.raises(TypeError):
        res = nn(early_stop='foo')
    with pytest.raises(TypeError):
        res = nn(early_stop_lim=False)
    with pytest.raises(TypeError):
        res = nn(xHI='false')
    with pytest.raises(TypeError):
        res = nn(resume=10)
