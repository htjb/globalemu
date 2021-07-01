import numpy as np
from globalemu.eval import evaluate
from globalemu.plotter import signal_plot
from globalemu.downloads import download
import os
import shutil
import pytest
import requests

params = [0.25, 30, 2, 0.056, 1.3, 2, 30]
z = np.arange(10, 20, 100)

def test_existing_dir():
    if os.path.exists('T_release/'):
        shutil.rmtree('T_release/')
    if os.path.exists('xHI_release/'):
        shutil.rmtree('xHI_release/')

    download().model()
    download(xHI=True).model()

    predictor = evaluate(base_dir='T_release/')

    parameters = np.loadtxt('21cmGEM_data/test_data.txt')
    labels = np.loadtxt('21cmGEM_data/test_labels.txt')

    signal_plot(parameters, labels, 'rmse',
                predictor, 'T_release/', figsizex=3.15,
                figsizey=6, loss_label='RMSE = {:.4f} [mK]')

    assert(os.path.exists('T_release/eval_plot.pdf') is True)

    with pytest.raises(KeyError):
        signal_plot(parameters, labels, 'rmse',
                    predictor, 'T_release/', figsize_x=3.15)

    with pytest.raises(TypeError):
        signal_plot(parameters, labels, 'rmse',
                    predictor, 'T_release')
    with pytest.raises(TypeError):
        signal_plot(1, labels, 'rmse',
                    predictor, 'T_release/')
    with pytest.raises(TypeError):
        signal_plot(parameters, labels, 'banana',
                    predictor, 'T_release/')
    with pytest.raises(TypeError):
        signal_plot(parameters, 6, 'rmse',
                    predictor, 'T_release/')
    with pytest.raises(TypeError):
        signal_plot(parameters, labels, 'rmse',
                    'foobar', 'T_release/')
    with pytest.raises(TypeError):
        signal_plot(parameters, labels, 'rmse',
                    predictor, 'T_release/', rtol='banana')
    with pytest.raises(TypeError):
        signal_plot(parameters, labels, 'rmse',
                    predictor, 'T_release/', atol='banana')
    with pytest.raises(TypeError):
        signal_plot(parameters, labels, 'rmse',
                    predictor, 'T_release/', figsizex='banana')
    with pytest.raises(TypeError):
        signal_plot(parameters, labels, 'rmse',
                    predictor, 'T_release/', figsizey='banana')
    with pytest.raises(TypeError):
        signal_plot(parameters, labels, 'rmse',
                    predictor, 'T_release/', loss_label=60)
    with pytest.raises(TypeError):
        signal_plot(parameters, labels, 'rmse',
                    predictor, 'T_release/', xHI=60)

    def loss_func(labels, signals):
        return np.mean(np.square(labels - signals))*1000

    signal_plot(parameters, labels, loss_func,
                predictor, 'T_release/')

    assert(os.path.exists('T_release/eval_plot.pdf') is True)

    signal_plot(parameters, labels, 'mse',
                predictor, 'T_release/')

    signal_plot(parameters, labels, 'GEMLoss',
                predictor, 'T_release/')
