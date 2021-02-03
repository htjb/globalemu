import numpy as np
from globalemu.preprocess import process
from globalemu.network import nn
import pytest
import requests, zipfile, io
import os

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
    nn(451, [8], 'tanh', 5, 1e-3)
    # results of below will not make sense as it is being run on the
    # global signal data but it will test the code (xHI data not public)
    process(10, z, data_location='data_download/', xHI=True)
    nn(451, [8], 'tanh', 5, 1e-3, xHI=True)
