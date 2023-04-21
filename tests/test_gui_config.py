import pandas as pd
from globalemu.gui_config import config
from globalemu.downloads import download
import shutil
import os
import pytest


def test_config():
    if os.path.exists('T_release/'):
        shutil.rmtree('T_release/')
    if os.path.exists('xHI_release/'):
        shutil.rmtree('xHI_release/')

    download().model()
    download(xHI=True).model()

    paramnames = [r'$\log(f_*)$', r'$\log(V_c)$', r'$\log(f_X)$',
                  r'$\nu_\mathrm{min}$', r'$\tau$', r'$\alpha$',
                  r'$R_\mathrm{mfp}$']

    # Providing this with global signal data as neutral fraction data is
    # not publicly available. Will not effect efficacy of the test.
    config('xHI_release/', paramnames, '21cmGEM_data/',
           logs=[0, 1, 2])

    assert(os.path.exists('xHI_release/gui_configuration.csv') is True)

    res = pd.read_csv('xHI_release/gui_configuration.csv')
    logs = res['logs'].tolist()
    logs = [int(x) for x in logs if x != '--']
    assert(logs == [0, 1, 2])

    paramnames = [r'$\log(f_*)$', r'$\log(V_c)$', r'$\log(f_X)$',
                  r'$\tau$', r'$\alpha$', r'$\nu_\mathrm{min}$',
                  r'$R_\mathrm{mfp}$']

    config('T_release/', paramnames, '21cmGEM_data/')

    assert(os.path.exists('T_release/gui_configuration.csv') is True)

    with pytest.raises(KeyError):
        config('T_release', paramnames, '21cmGEM_data/')
    with pytest.raises(TypeError):
        config(10, paramnames, '21cmGEM_data/')
    with pytest.raises(KeyError):
        config('T_release/', paramnames, '21cmGEM_data')
    with pytest.raises(TypeError):
        config('T_release/', paramnames, '21cmGEM_data/', logs='banana')
    with pytest.raises(TypeError):
        config('T_release/', 5, '21cmGEM_data/')

    with pytest.raises(KeyError):
        config('T_release/', paramnames, '21cmGEM_data/', color='C0')
