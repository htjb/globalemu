import numpy as np
import matplotlib.pyplot as plt
from globalemu.eval import evaluate
from globalemu.downloads import download
import pytest

params = [0.25, 30, 2, 0.056, 1.3, 2, 30]
z = np.arange(10, 20, 100)

def test_existing_dir():
    #download(False).model()
    #download(True).model()
    res = evaluate(params, z=z, base_dir='best_T/')
    #res = evaluate(params, z=z, base_dir='best_xHI/', xHI=True)
    res = evaluate(params, z=10, base_dir='best_T/')
