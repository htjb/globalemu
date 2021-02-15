import numpy as np
from globalemu.eval import evaluate
from globalemu.downloads import download
import os

params = [0.25, 30, 2, 0.056, 1.3, 2, 30]
z = np.arange(10, 20, 100)


def test_existing_dir():
    if os.path.exists('T_release/'):
        os.remove('T_release/')
    if os.path.exists('xHI_release/'):
        os.remove('xHI_release/')

    download(False).model()
    download(True).model()

    res = evaluate(params, z=z, base_dir='T_release/')
    res = evaluate(params, z=z, base_dir='xHI_release/', xHI=True)
    res = evaluate(params, z=10, base_dir='T_release/')
