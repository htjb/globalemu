from globalemu.downloads import download
import os
import pytest
import requests

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

def test_existing_dir():
    if os.path.exists('kappa_HH.txt'):
        os.remove('kappa_HH.txt')

    download().kappa()

    with pytest.raises(TypeError):
        download(xHI=2).kappa()

    # for use in later tests...
    download_21cmGEM_data()
