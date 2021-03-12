import numpy as np
from globalemu.downloads import download
import os
import pytest

def test_existing_dir():
    if os.path.exists('kappa_HH.txt'):
        os.remove('kappa_HH.txt')

    download().kappa()

    with pytest.raises(TypeError):
        download(xHI=2).kappa()
