import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
import numpy as np
from tensorflow.keras import backend as K
import time
from zT.models import network_models
import os
import pandas as pd
from zT.losses import loss_functions

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
        self.BN = kwargs.pop('BN', False)
        self.reg = kwargs.pop('reg', None)
        self.weights = kwargs.pop('weights', False)

        if not os.path.exists(self.base_dir):
            os.mkdir(self.base_dir)

        pwd = os.getcwd()
        train_dataset_fp = pwd + '/' + self.base_dir + 'zT_train_dataset.csv'

        if self.weights is True:
            column_names = ['fstar', 'Vc', 'fx', 'tau', 'alpha', 'nu_min', 'Rmfp', 'z', 'w', 'T']
        else:
            column_names = ['fstar', 'Vc', 'fx', 'tau', 'alpha', 'nu_min', 'Rmfp', 'z', 'T']
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

        if self.BN is True:
            model = network_models().basic_model_norm(
                self.input_shape, self.output_shape,
                self.layer_sizes, self.activation, self.drop_val)
        else:
            if self.reg == 'l2':
                model = network_models().basic_model_L2(
                    self.input_shape, self.output_shape,
                    self.layer_sizes, self.activation, self.drop_val)
            else:
                model = network_models().basic_model(
                    self.input_shape, self.output_shape,
                    self.layer_sizes, self.activation, self.drop_val)


        def loss(model, x, y, training):
            y_ = tf.transpose(model(x, training=training))[0]
            lf = loss_functions(y, y_)
            return lf.mse(), lf.rmse()

        def grad(model, inputs, targets):
            with tf.GradientTape() as tape:
                loss_value, rmse = loss(model, inputs, targets, training=True)
            return loss_value, rmse, tape.gradient(loss_value, model.trainable_variables)

        def loss_weights(model, x, y, w, training):
            y_ = tf.transpose(model(x, training=training))[0]
            lf = loss_functions(y, y_)
            return lf.wmse(w), lf.mse(), lf.rmse()

        def grad_weights(model, inputs, targets, w):
        	with tf.GradientTape() as tape:
        		loss_value, mse, rmse = loss_weights(model, inputs, targets, w, training=True)
        	return loss_value, mse, rmse, tape.gradient(loss_value, model.trainable_variables)

        optimizer = keras.optimizers.Adam(learning_rate=self.lr)

        train_loss_results = []
        if self.weights is True:
            train_mse_results = []
        train_rmse_results = []
        num_epochs = self.epochs
        for epoch in range(num_epochs):
            s = time.time()
            epoch_loss_avg = tf.keras.metrics.Mean()
            epoch_rmse_avg = tf.keras.metrics.Mean()
            if self.weights is True:
                epoch_mse_avg = tf.keras.metrics.Mean()

            for x, y in train_dataset:
                if self.weights is True:
                    loss_values, mse, rmse, grads = grad_weights(
                        model, x[:, :-1], y, x[:, -1])
                else:
                    loss_values, rmse, grads = grad(model, x, y)
                optimizer.apply_gradients(zip(grads, model.trainable_variables))
                epoch_loss_avg.update_state(loss_values)
                if self.weights is True:
                    epoch_mse_avg.update_state(mse)
                epoch_rmse_avg.update_state(rmse)

            train_loss_results.append(epoch_loss_avg.result())
            if self.weights is True:
                train_mse_results.append(epoch_mse_avg.result())
            train_rmse_results.append(epoch_rmse_avg.result())
            e = time.time()

            if self.weights is True:
                print(
                    'Epoch: {:03d}, Loss: {:.6f}, MSE: {:.4f}, RMSE: {:.4f}, Time: {:.3f}'.format(
                    epoch, epoch_loss_avg.result(), epoch_mse_avg.result(),
                    epoch_rmse_avg.result(), e-s))
            else:
                print(
                    'Epoch: {:03d}, Loss: {:.4f}, RMSE: {:.4f}, Time: {:.3f}'.format(epoch,
                    epoch_loss_avg.result(),
                    epoch_rmse_avg.result(), e-s))

            if len(train_loss_results) > 10:
                if np.isclose(
                        train_loss_results[-10], train_loss_results[-1],
                        1e-4, 1e-4):
                    print('Early Stop')
                    model.save(self.base_dir + 'zT_model.h5')
                    break
            if epoch % 10 == 0:
                model.save(self.base_dir + 'zT_model.h5')

        model.save(self.base_dir + 'zT_model.h5')

        plt.figure()
        plt.ylabel('Loss', fontsize=14)
        plt.plot(train_loss_results)
        plt.xlabel('Epoch', fontsize=14)
        plt.savefig(self.base_dir + 'accurcay_loss.pdf')
        plt.show()
