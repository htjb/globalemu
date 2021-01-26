import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
import numpy as np
from tensorflow.keras import backend as K
import time
from zT.models import network_models
import os
from zT.losses import loss_functions

class nn():
    def __init__(
        self, batch_size, layer_sizes, activation, drop_val, epochs, lr,
        **kwargs):
        self.batch_size = batch_size
        self.layer_sizes =layer_sizes
        self.activation = activation
        self.drop_val = drop_val
        self.epochs = epochs
        self.lr = lr
        self.input_shape = kwargs.pop('input_shape', 8)
        self.output_shape = kwargs.pop('output_shape', 1)
        self.base_dir = kwargs.pop('base_dir', 'model_dir/')
        self.early_stop = kwargs.pop('early_stop', False)
        self.xHI = kwargs.pop('xHI', False)

        if not os.path.exists(self.base_dir):
            os.mkdir(self.base_dir)

        pwd = os.getcwd()
        train_dataset_fp = pwd + '/' + self.base_dir + 'train_dataset.csv'

        column_names = [
            'p' + str(i)
            for i in range(self.input_shape + self.output_shape)]
        feature_names = column_names[:-1]
        label_names = column_names[-1]

        train_dataset = tf.data.experimental.make_csv_dataset(
        	train_dataset_fp,
        	batch_size,
        	column_names = column_names,
        	label_name = label_names,
        	num_epochs=1)

        def pack_features_vector(features, labels):
            features = tf.stack(list(features.values()), axis=1)
            return features, labels

        train_dataset = train_dataset.map(pack_features_vector)

        if self.xHI is False:
            model = network_models().basic_model(
                self.input_shape, self.output_shape,
                self.layer_sizes, self.activation, self.drop_val,
                'linear')
        else:
            model = network_models().basic_model(
                self.input_shape, self.output_shape,
                self.layer_sizes, self.activation, self.drop_val,
                'relu')

        def loss(model, x, y, training):
            y_ = tf.transpose(model(x, training=training))[0]
            lf = loss_functions(y, y_)
            return lf.mse(), lf.rmse()

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

            print(
                'Epoch: {:03d}, Loss: {:.5f}, RMSE: {:.5f}, Time: {:.3f}'.format(epoch,
                epoch_loss_avg.result(),
                epoch_rmse_avg.result(), e-s))

            if self.early_stop is True:
                if len(train_loss_results) > 10:
                    if np.isclose(
                            train_loss_results[-10], train_loss_results[-1],
                            1e-4, 1e-4):
                        print('Early Stop')
                        model.save(self.base_dir + 'model.h5')
                        break
            if epoch % 10 == 0:
                model.save(self.base_dir + 'model.h5')
                np.savetxt(self.base_dir + 'loss_history.txt', train_loss_results)

        model.save(self.base_dir + 'model.h5')
        np.savetxt(self.base_dir + 'loss_history.txt', train_loss_results)
