import numpy as np
from globalemu.preprocess import process
from globalemu.network import nn
from tensorflow.keras import backend as K
import os
import shutil
import pytest


def test_process_nn():

    def custom_loss(y, y_, x):
        return K.mean(K.abs(y - y_))

    z = np.arange(5, 50.1, 0.1)

    process(10, z, data_location="21cmGEM_data/")
    nn(batch_size=451, layer_sizes=[8], epochs=10, loss_function=custom_loss)

    # results of below will not make sense as it is being run on the
    # global signal data but it will test the code (xHI data not public)
    process(10, z, data_location="21cmGEM_data/", xHI=True)
    nn(batch_size=451, layer_sizes=[8], epochs=5, xHI=True)

    nn(batch_size=451, layer_sizes=[8], epochs=5, output_activation="linear")

    # test early_stop code
    nn(batch_size=451, layer_sizes=[], epochs=20, early_stop=True)

    process(10, z, data_location="21cmGEM_data/", base_dir="base_dir/")
    nn(
        batch_size=451,
        layer_sizes=[],
        random_seed=10,
        epochs=30,
        base_dir="base_dir/",
        early_stop=True,
    )

    dir = ["model_dir/", "base_dir/"]
    for i in range(len(dir)):
        if os.path.exists(dir[i]):
            shutil.rmtree(dir[i])


def test_process_nn_keyword_errors():
    z = np.arange(5, 50.1, 0.1)
    with pytest.raises(KeyError):
        process(10, z, datalocation="21cmGEM_data/")

    with pytest.raises(KeyError):
        nn(batch_size=451, layersizes=[8], epochs=10)


@pytest.mark.parametrize(
    "keyword, value",
    [
        ("batch_size", "foo"),
        ("activation", 10),
        ("epochs", False),
        ("lr", "bar"),
        ("dropout", True),
        ("input_shape", "foo"),
        ("output_shape", "foobar"),
        ("layer_sizes", 10),
        ("base_dir", 50),
        ("early_stop", "foo"),
        ("xHI", "false"),
        ("resume", 10),
        ("output_activation", 2),
        ("loss_function", "foobar"),
    ],
)
def test_process_nn_type_errors(keyword, value):
    z = np.arange(5, 50.1, 0.1)

    process(10, z, data_location="21cmGEM_data/")
    with pytest.raises((TypeError, ValueError)):
        nn(**{keyword: value})
