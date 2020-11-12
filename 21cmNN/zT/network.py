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
        self.BN = kwargs.pop('BN', 'True')

        if not os.path.exists(self.base_dir):
            os.mkdir(self.base_dir)

        train_dataset_fp = '/home/harry/Documents/emulator/21cmNN/' + self.base_dir + 'zT_train_dataset.csv'

        column_names = ['fstar', 'Vc', 'fx', 'tau', 'alpha', 'nu_min', 'Rmfp', 'z', 'T']
        #column_names = ['a', 'b', 'x', 'T']
        feature_names = column_names[:-1]
        label_names = column_names[-1]

        train_dataset = tf.data.experimental.make_csv_dataset(
        	train_dataset_fp,
        	batch_size,
        	column_names = column_names,
        	label_name = label_names,
        	num_epochs=1)
            #shuffle=False)

        def pack_features_vector(features, labels):
            features = tf.stack(list(features.values()), axis=1)
            return features, labels

        train_dataset = train_dataset.map(pack_features_vector)
        #features, labels = next(iter(train_dataset))
        #print(features, labels)
        #print(train_dataset)

        if self.BN is True:
            model = network_models().basic_model_norm(
                self.input_shape, self.output_shape,
                self.layer_sizes, self.activation, self.drop_val)
        else:
            model = network_models().basic_model(
                self.input_shape, self.output_shape,
                self.layer_sizes, self.activation, self.drop_val)

        #print(model.summary())

        def root_mean_squared_error(y, y_):
        	return K.sqrt(K.mean(K.square(y - y_)))/ \
                K.max(K.abs(y))

        def mean_squared_error(y, y_):
            return K.mean(K.square(y - y_))

        #def chi(y, y_):
        #    return K.sum(K.square(y - y_))

        def loss(model, x, y, training):
            y_ = tf.transpose(model(x, training=training))[0]
            return mean_squared_error(y, y_), root_mean_squared_error(y, y_)
            #return chi(y, y_), mean_squared_error(y, y_), root_mean_squared_error(y, y_)

        def grad(model, inputs, targets):
        	with tf.GradientTape() as tape:
        		loss_value, rmse = loss(model, inputs, targets, training=True)
        	return loss_value, rmse, tape.gradient(loss_value, model.trainable_variables)

        optimizer = keras.optimizers.Adam(learning_rate=self.lr)

        train_loss_results = []
        train_rmse_results = []
        num_epochs = self.epochs
        for epoch in range(num_epochs):
            s = time.time()
            epoch_loss_avg = tf.keras.metrics.Mean()
            epoch_rmse_avg = tf.keras.metrics.Mean()

            for x, y in train_dataset:
                loss_values, rmse, grads = grad(model, x, y)
                optimizer.apply_gradients(zip(grads, model.trainable_variables))
                epoch_loss_avg.update_state(loss_values)
                epoch_rmse_avg.update_state(rmse)

            train_loss_results.append(epoch_loss_avg.result())
            train_rmse_results.append(epoch_rmse_avg.result())
            e = time.time()
            print('Epoch: {:03d}, Loss: {:.3f}, RMSE: {:.3f}, Time: {:.3f}'.format(epoch,
            epoch_loss_avg.result(), epoch_rmse_avg.result(), e-s))
            if len(train_loss_results) > 10:
                if np.isclose(
                        train_loss_results[-10], train_loss_results[-1],
                        1e-4, 1e-4):
                    print('Early Stop')
                    model.save(self.base_dir + 'zT_model')
                    break
            if epoch % 10 == 0:
                model.save(self.base_dir + 'zT_model')

        model.save(self.base_dir + 'zT_model')

        plt.figure()
        plt.ylabel('Loss', fontsize=14)
        plt.plot(train_loss_results)
        plt.xlabel('Epoch', fontsize=14)
        plt.savefig(self.base_dir + 'accurcay_loss.pdf')
        plt.show()
