import numpy as np
from globalemu.preprocess import process
from globalemu.network import nn
import requests
import os
import shutil
import pytest


def test_process_nn():
    z = np.arange(5, 50.1, 0.1)

    process(10, z, data_location='21cmGEM_data/')
    nn(batch_size=451, layer_sizes=[8], epochs=10)

    # results of below will not make sense as it is being run on the
    # global signal data but it will test the code (xHI data not public)
    process(10, z, data_location='21cmGEM_data/', xHI=True)
    nn(batch_size=451, layer_sizes=[8], epochs=5, xHI=True)

    nn(batch_size=451, layer_sizes=[8], epochs=5, output_activation='linear')

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
    with pytest.raises(TypeError):
        nn(output_activation=2)

    process(10, z, data_location='21cmGEM_data/', base_dir='base_dir/')
    nn(batch_size=451, layer_sizes=[], random_seed=10,
       base_dir='base_dir/')

    dir = ['model_dir/', 'base_dir/']
    for i in range(len(dir)):
        if os.path.exists(dir[i]):
            shutil.rmtree(dir[i])
