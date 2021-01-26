import numpy as np
import matplotlib.pyplot as plt
from zT.eval import evaluate
import pytest

params = [0.25, 30, 2, 0.056, 1.3, 2, 30]
z = np.arange(10, 20, 100)

def test_existing_default_dir():
    res = evaluate(params, z=z)

def test_existing_dir():
    res = evaluate(params, z=z, base_dir='xHI_250121/', xHI=True)

#def test_download_model():
#    res = evaluate(params, z=z, base_dir='rrrr/')
