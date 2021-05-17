from globalemu.cmSim import calc_signal
import numpy as np
import os
import shutil


def test_calc_signal():
    if os.path.exists('kappa_HH.txt'):
        os.remove('kappa_HH.txt')

    if not os.path.exists('model_dir/'):
        os.mkdir('model_dir/')

    os.system("cp T_release/AFB_norm_factor.npy model_dir/")

    _ = calc_signal(np.arange(5, 50, 100), 'model_dir/')

    shutil.rmtree('model_dir/')
