import numpy as np
from globalemu.eval import evaluate
from globalemu.downloads import download
import os
import shutil
import pytest

params = [0.25, 30, 2, 0.056, 1.3, 2, 30]
z = np.arange(10, 20, 100)


def test_existing_dir():
    #if os.path.exists('T_release/'):
    #    shutil.rmtree('T_release/')
    #if os.path.exists('xHI_release/'):
    #    shutil.rmtree('xHI_release/')

    #download().model()
    #download(True).model()

    res = evaluate(params, z=z, base_dir='T_release/')
    res = evaluate(params, z=z, base_dir='xHI_release/', xHI=True)
    res = evaluate(params, z=10, base_dir='T_release/')

    with pytest.raises(KeyError):
        res = evaluate(params, z=z, basedir='T_release/')

    with pytest.raises(TypeError):
        res = evaluate(10, z=z, base_dir='T_release/')
    with pytest.raises(TypeError):
        res = evaluate(params, z=z, base_dir=100)
    with pytest.raises(TypeError):
        res = evaluate(params, z=z, base_dir='T_release/', logs=10)
    with pytest.raises(TypeError):
        res = evaluate(params, z='foo', base_dir='T_release/')
    with pytest.raises(TypeError):
        res = evaluate(params, z=z, base_dir='T_release/',
                       gc='false')
    with pytest.raises(TypeError):
        res = evaluate(params, z=z, base_dir='T_release/', xHI='bar')
