import numpy as np
import matplotlib.pyplot as plt
from globalemu.preprocess import process
import pytest
import requests, zipfile, io
import os

def test_process():
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
