import pandas as pd
from globalemu.gui_config import config
from globalemu.downloads import download
import requests
import shutil
import os
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


def test_config():
    if os.path.exists('T_release/'):
        shutil.rmtree('T_release/')
    if os.path.exists('xHI_release/'):
        shutil.rmtree('xHI_release/')

    download().model()
    download(xHI=True).model()

    download_21cmGEM_data()

    paramnames = [r'$\log(f_*)$', r'$\log(V_c)$', r'$\log(f_X)$',
                  r'$\nu_\mathrm{min}$', r'$\tau$', r'$\alpha$',
                  r'$R_\mathrm{mfp}$']

    # Providing this with global signal data as neutral fraction data is
    # not publicly available. Will not effect efficacy of the test.
    config('xHI_release/', paramnames, '21cmGEM_data/',
           logs=[0, 1, 2], xHI=True)

    assert(os.path.exists('xHI_release/gui_configuration.csv') is True)

    res = pd.read_csv('xHI_release/gui_configuration.csv')
    assert(res['xHI'][0] is True)
    logs = res['logs'].tolist()
    logs = [int(x) for x in logs if x != '--']
    assert(logs == [0, 1, 2])

    paramnames = [r'$\log(f_*)$', r'$\log(V_c)$', r'$\log(f_X)$',
                  r'$\tau$', r'$\alpha$', r'$\nu_\mathrm{min}$',
                  r'$R_\mathrm{mfp}$']

    config('T_release/', paramnames, '21cmGEM_data/')

    assert(os.path.exists('T_release/gui_configuration.csv') is True)

    res = pd.read_csv('T_release/gui_configuration.csv')
    assert(res['xHI'][0] is False)

    with pytest.raises(KeyError):
        config('T_release', paramnames, '21cmGEM_data/')
    with pytest.raises(TypeError):
        config(10, paramnames, '21cmGEM_data/')
    with pytest.raises(KeyError):
        config('T_release/', paramnames, '21cmGEM_data')
    with pytest.raises(TypeError):
        config('T_release/', paramnames, '21cmGEM_data/', xHI=4)
    with pytest.raises(TypeError):
        config('T_release/', paramnames, '21cmGEM_data/', logs='banana')
    with pytest.raises(TypeError):
        config('T_release/', 5, '21cmGEM_data/')

    with pytest.raises(KeyError):
        config('T_release/', paramnames, '21cmGEM_data/', color='C0')

    shutil.rmtree('21cmGEM_data/')
