import tensorflow as tf
from tensorflow import keras
import numpy as np
import time
import os
from globalemu.models import network_models
from globalemu.losses import loss_functions


class nn():
    def __init__(self, **kwargs):

        for key, values in kwargs.items():
            if key not in set(
                    ['batch_size', 'activation', 'epochs',
                        'lr', 'dropout', 'input_shape',
                        'output_shape', 'layer_sizes', 'base_dir',
                        'early_stop', 'xHI', 'resume']):
                raise KeyError("Unexpected keyward argument in nn()")

        self.batch_size = kwargs.pop('batch_size', 100)
        self.activation = kwargs.pop('activation', 'tanh')
        self.epochs = kwargs.pop('epochs', 10)
        self.lr = kwargs.pop('lr', 1e-3)
        self.drop_val = kwargs.pop('dropout', 0)
        self.input_shape = kwargs.pop('input_shape', 8)
        self.output_shape = kwargs.pop('output_shape', 1)
        self.layer_sizes = kwargs.pop(
            'layer_sizes', [self.input_shape, self.input_shape])
        self.base_dir = kwargs.pop('base_dir', 'model_dir/')
        self.early_stop = kwargs.pop('early_stop', False)
        self.xHI = kwargs.pop('xHI', False)
        self.resume = kwargs.pop('resume', False)

        if not os.path.exists(self.base_dir):
            os.mkdir(self.base_dir)

        pwd = os.getcwd()
        train_dataset_fp = pwd + '/' + self.base_dir + 'train_dataset.csv'

        column_names = [
            'p' + str(i)
            for i in range(self.input_shape + self.output_shape)]
        label_names = column_names[-1]

        train_dataset = tf.data.experimental.make_csv_dataset(
            train_dataset_fp,
            self.batch_size,
            column_names=column_names,
            label_name=label_names,
            num_epochs=1)

        def pack_features_vector(features, labels):
            return tf.stack(list(features.values()), axis=1), labels

        train_dataset = train_dataset.map(pack_features_vector)

        if self.resume is True:
            model = keras.models.load_model(
                self.base_dir + 'model.h5',
                compile=False)
        elif self.xHI is False:
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
            return loss_value, rmse, tape.gradient(
                loss_value, model.trainable_variables)

        optimizer = keras.optimizers.Adam(learning_rate=self.lr)

        if self.resume is True:
            train_loss_results = list(
                np.loadtxt(self.base_dir + 'loss_history.txt'))
        else:
            train_loss_results = []
        train_rmse_results = []
        num_epochs = self.epochs
        for epoch in range(num_epochs):
            s = time.time()
            epoch_loss_avg = tf.keras.metrics.Mean()
            epoch_rmse_avg = tf.keras.metrics.Mean()

            for x, y in train_dataset:
                loss_values, rmse, grads = grad(model, x, y)
                optimizer.apply_gradients(
                    zip(grads, model.trainable_variables))
                epoch_loss_avg.update_state(loss_values)
                epoch_rmse_avg.update_state(rmse)

            train_loss_results.append(epoch_loss_avg.result())
            train_rmse_results.append(epoch_rmse_avg.result())
            e = time.time()

            print(
                'Epoch: {:03d}, Loss: {:.5f}, RMSE: {:.5f}, Time: {:.3f}'
                .format(
                    epoch, epoch_loss_avg.result(),
                    epoch_rmse_avg.result(), e-s))

            if self.early_stop is True:
                if len(train_loss_results) > 10:
                    if np.isclose(
                            train_loss_results[-10], train_loss_results[-1],
                            1e-4, 1e-4):
                        print('Early Stop')
                        model.save(self.base_dir + 'model.h5')
                        break
            if (epoch + 1) % 10 == 0:
                model.save(self.base_dir + 'model.h5')
                np.savetxt(
                    self.base_dir + 'loss_history.txt', train_loss_results)

        model.save(self.base_dir + 'model.h5')
        np.savetxt(self.base_dir + 'loss_history.txt', train_loss_results)
