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
