import numpy as np
from globalemu.eval import evaluate
from globalemu.downloads import download
import os
import shutil
import pytest

params = [0.25, 30, 2, 0.056, 1.3, 2, 30]
originalz = np.linspace(10, 20, 100)


def test_existing_dir():
    if os.path.exists('T_release/'):
        shutil.rmtree('T_release/')
    if os.path.exists('xHI_release/'):
        shutil.rmtree('xHI_release/')

    download().model()
    download(xHI=True).model()

    predictor = evaluate(z=originalz, base_dir='T_release/')
    signal, z = predictor(params)

    assert(len(signal.shape) == 1)

    params_array = []
    for i in range(10):
        params_array.append(np.random.uniform(0, 1, 7))
    params_array = np.array(params_array)

    data_max = np.loadtxt('T_release/data_maxs.txt')
    data_min = np.loadtxt('T_release/data_mins.txt')

    for i in range(params_array.shape[1]):
        params_array[:, i] = params_array[:, i] * \
            (data_max[i] - data_min[i]) + data_min[i]

    signal, z = predictor(params_array)

    assert(len(signal.shape) == 2)

    predictor = evaluate(z=10, base_dir='T_release/')
    signal, z = predictor(params)
    predictor = evaluate(z=originalz, base_dir='xHI_release/')
    signal, z = predictor(params)

    with pytest.raises(KeyError):
        evaluate(z=originalz, basedir='T_release/')

    with pytest.raises(TypeError):
        predictor = evaluate(z=originalz, base_dir='T_release/')
        predictor(10)
    with pytest.raises(TypeError):
        predictor = evaluate(z=originalz, base_dir=100)
    with pytest.raises(TypeError):
        predictor = evaluate(z=originalz, base_dir='T_release/', logs=10)
    with pytest.raises(TypeError):
        predictor = evaluate(z='foo', base_dir='T_release/')
    with pytest.raises(TypeError):
        predictor = evaluate(z=originalz, base_dir='T_release/',
                             gc='false')
