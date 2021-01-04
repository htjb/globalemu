import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
import numpy as np
from tensorflow.keras import backend as K
import time
from zT.models import network_models
import os
import pandas as pd

class nn():
    def __init__(
        self, batch_size, layer_sizes, activation, drop_val, epochs, lr,
        input_shape, output_shape, **kwargs):
        self.batch_size = batch_size #136
        self.layer_sizes =layer_sizes
        self.activation = activation
        self.drop_val = drop_val
        self.epochs = epochs
        self.lr = lr
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.base_dir = kwargs.pop('base_dir', 'results/')

        if not os.path.exists(self.base_dir):
            os.mkdir(self.base_dir)

        train_dataset_fp = '/home/harry/Documents/emulator/21cmNN/' + self.base_dir + 'zT_train_dataset.csv'

        column_names = ['fstar', 'Vc', 'fx', 'tau', 'alpha', 'nu_min', 'Rmfp', 'z', 'T']
        #column_names = ['a', 'b', 'x', 'T']
        feature_names = column_names[:-1]
        label_names = column_names[-1]

        raw_dataset = pd.read_csv(train_dataset_fp, names=column_names)
        dataset = raw_dataset.copy()
        train_dataset=dataset

        train_features = train_dataset.copy()
        train_labels = train_features.pop('T')

        model = network_models().basic_model(
            self.input_shape, self.output_shape,
            self.layer_sizes, self.activation, self.drop_val)

        optimizer = keras.optimizers.Adam(learning_rate=self.lr)
        model.compile(optimizer=optimizer, loss='mse')

        history = model.fit(
            train_features, train_labels,
            epochs = 100,
            verbose=1,
            validation_split = 0.2
        )

        model.save(self.base_dir + 'zT_model')

        hist = pd.DataFrame(history.history)
        #print(hist.tail())
        #sys.exit(1)

        plt.figure()
        plt.ylabel('Loss', fontsize=14)
        plt.plot(hist['loss'])
        plt.xlabel('Epoch', fontsize=14)
        plt.savefig(self.base_dir + 'accurcay_loss.pdf')
        plt.show()
