"""

``nn()`` is used to train an instance of ``globalemu`` on the preprocessed
data in ``base_dir``. All of the parameters for ``nn()`` are kwargs and
a number of them can be left at their default values however you will
need to set the ``base_dir`` and possibly ``epochs`` and ``xHI`` (see below and
the tutorial for details).

"""

import tensorflow as tf
from tensorflow import keras
import numpy as np
import time
import os
from globalemu.models import network_models
from globalemu.losses import loss_functions


class nn():

    r"""

    **kwargs:**

        batch_size: **int / default: 100**
            | The batch size used by ``tensorflow`` when performing training.
                Corresponds to the number of samples propagated before the
                networks hyperparameters are updated. Keep the value ~100 as
                this will help with memory management and training speed.

        epochs: **int / default: 10**
            | The number of epochs to train the network on. An epoch
                corresponds to training on x batches where x is sufficiently
                large for every sample to have influenced an update of the
                network hyperparameters.

        activation: **string / default: 'tanh'**
            | The type of activation function used in the neural networks
                hidden layers. The activation function effects the way that the
                network learns and updates its hyperparameters. The defualt
                is a commonly used activation for regression neural networks.

        lr: **float / default: 0.001**
            | The learning rate acts as a "step size" in the optimization and
                its value can effect the quality of the emulation. Typical
                values fall in the range 0.001-0.1.

        dropout: **float / default: 0**
            | The dropout for the neural network training. ``globalemu`` is
                designed so that you shouldn't need dropout to prevent
                overfitting but we leave it as an option.

        input_shape: **int / default: 8**
            | The number of input parameters (astrophysical parameters
                plus redshift) for the neural network. The default accounts
                for 7 astrophysical
                parameters and a single redshift input.

        output_shape: **int / default: 1**
            | The number of ouputs (temperature) from the neural network.
                This shouldn't need changing.

        layer_sizes: **list / default: [input_shape, input_shape]**
            | The number of hidden layers and the number of nodes in each
                layer. For example ``layer_sizes=[8, 8]`` will create
                two hidden layers both with 8 nodes (this is the default).

        base_dir: **string / default: 'model_dir/'**
            | This should be the same as the ``base_dir`` used when
                preprocessing. It contains the data that the network will
                work with and is the directory in which the trained model will
                be saved in.

        early_stop: **Bool / default: False**
            | If ``early_stop`` is set too ``True`` then the network will stop
                learning if the loss has not changed within
                the last twenty epochs.

        xHI: **Bool / default: False**
            | If True then ``globalemu`` will act as if it is training a
                neutral fraction history emulator.

        output_activation: **string / default: 'linear'**
            | Determines the output activation function for the network.
                Modifying this
                is useful if the emulator output is required to be positive or
                negative etc. If xHI is True then the output activation is
                set to 'relu' else the function is 'linear'. See the tensorflow
                documentation for more details on the types of activation
                functions available.

        loss_function: **Callable/ default: None**
            | By default the code uses an MSE loss however users are able to
                pass their own loss functions when training the neural
                network. These should be functions that take in the true labels
                (temperatures) and the predicted labels and return some measure
                of loss. Care needs to be taken to ensure that the correct loss
                function is supplied when resuming the training of
                a previous run as ``globalemu`` will not check this. In order
                for the loss function to work it must be built
                using the tensorflow.keras backend. An example would be

                .. code:: python

                    from tensorflow.keras import backend as K

                    def custom_loss(true_labels, predicted_labels,
                            netowrk_inputs):
                        return K.mean(K.abs(true_labels - predicted_labels))

                The function must take in as arguments the `true_labels`,
                the `predicted_labels` and the `network_inputs`.

        resume: **Bool / default: False**
            | If set to ``True`` then ``globalemu`` will look in the
                ``base_dir`` for a trained model and ``loss_history.txt``
                file (which contains the loss recorded at each epoch) and
                load these in to continue training. If ``resume`` is ``True``
                then you need to make sure all of the kwargs are set the
                with the same values that they had in the initial training
                for a consistent run.
                There will be a human readable file in ``base_dir`` called
                "kwargs.txt" detailing
                the values of the kwargs that were provided for the
                initial training run. Anything missing from this file will
                of had its default value. This file will not be overwritten
                if ``resume=True``.

        random_seed: **int or float / default: None**
            | This kwarg sets the random seed used by tensorflow with the
                function ``tf.random.set_seed(random_seed)``. It should
                be used if you want to have reproducible results but note
                that it may cause an 'out of memory' error if training on
                large amounts of data
                (see https://github.com/tensorflow/tensorflow/issues/37252).

    """
    def __init__(self, **kwargs):

        for key, values in kwargs.items():
            if key not in set(
                    ['batch_size', 'activation', 'epochs',
                        'lr', 'dropout', 'input_shape',
                        'output_shape', 'layer_sizes', 'base_dir',
                        'early_stop', 'xHI', 'resume',
                        'random_seed', 'output_activation',
                        'loss_function']):
                raise KeyError("Unexpected keyword argument in nn()")

        self.resume = kwargs.pop('resume', False)
        self.base_dir = kwargs.pop('base_dir', 'model_dir/')
        if type(self.base_dir) is not str:
            raise TypeError("'base_dir' must be a sting.")
        elif self.base_dir.endswith('/') is False:
            raise KeyError("'base_dir' must end with '/'.")

        if self.resume is not True:
            with open(self.base_dir + 'kwargs.txt', 'w') as f:
                for key, values in kwargs.items():
                    f.write(str(key) + ': ' + str(values) + '\n')
                f.close()

        self.batch_size = kwargs.pop('batch_size', 100)
        self.activation = kwargs.pop('activation', 'tanh')
        if type(self.activation) is not str:
            raise TypeError("'activation' must be a string.")
        self.epochs = kwargs.pop('epochs', 10)
        self.lr = kwargs.pop('lr', 1e-3)
        self.drop_val = kwargs.pop('dropout', 0)
        self.input_shape = kwargs.pop('input_shape', 8)
        self.output_shape = kwargs.pop('output_shape', 1)
        self.layer_sizes = kwargs.pop(
            'layer_sizes', [self.input_shape, self.input_shape])
        if type(self.layer_sizes) is not list:
            raise TypeError("'layer_sizes' must be a list.")
        self.early_stop = kwargs.pop('early_stop', False)
        self.xHI = kwargs.pop('xHI', False)
        self.random_seed = kwargs.pop('random_seed', None)

        boolean_kwargs = [self.resume, self.early_stop, self.xHI]
        boolean_strings = ['resume', 'early_stop', 'xHI']
        for i in range(len(boolean_kwargs)):
            if type(boolean_kwargs[i]) is not bool:
                raise TypeError("'" + boolean_strings[i] + "' must be a bool.")

        int_kwargs = [self.batch_size, self.epochs, self.input_shape,
                      self.output_shape]
        int_strings = ['batch_size', 'epochs', 'input_shape',
                       'output_shape']
        for i in range(len(int_kwargs)):
            if type(int_kwargs[i]) is not int:
                raise TypeError("'" + int_strings[i] + "' must be a int.")

        float_kwargs = [self.lr, self.drop_val,
                        self.random_seed]
        float_strings = ['lr', 'dropout', 'random_seed']
        for i in range(len(float_kwargs)):
            if float_kwargs[i] is not None:
                if type(float_kwargs[i]) not in set([float, int]):
                    raise TypeError("'" + float_strings[i] +
                                    "' must be a float.")

        loss_function = kwargs.pop('loss_function', None)
        if loss_function is not None:
            if not callable(loss_function):
                raise TypeError('loss_function should be a callable.')

        if self.random_seed is not None:
            tf.random.set_seed(self.random_seed)

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

        test_data = np.loadtxt(self.base_dir + 'test_data.txt')
        test_labels = np.loadtxt(self.base_dir + 'test_label.txt')

        def pack_features_vector(features, labels):
            return tf.stack(list(features.values()), axis=1), labels

        train_dataset = train_dataset.map(pack_features_vector)

        self.output_activation = kwargs.pop('output_activation', 'linear')
        if self.xHI is True:
            self.output_activation = 'relu'

        if self.resume is True:
            model = keras.models.load_model(
                self.base_dir + 'model.h5',
                compile=False)
        else:
            model = network_models().basic_model(
                self.input_shape, self.output_shape,
                self.layer_sizes, self.activation, self.drop_val,
                self.output_activation)

        def loss(model, x, y, training):
            y_ = tf.transpose(model(x, training=training))[0]
            lf = loss_functions(y, y_)
            if loss_function is None:
                return lf.mse(), lf.rmse()
            else:
                return loss_function(y, y_, x), lf.rmse()

        def grad(model, inputs, targets):
            with tf.GradientTape() as tape:
                loss_value, rmse = loss(model, inputs, targets, training=True)
            return loss_value, rmse, tape.gradient(
                loss_value, model.trainable_variables)

        optimizer = keras.optimizers.Adam(learning_rate=self.lr)

        if self.resume is True:
            train_loss_results = list(
                np.loadtxt(self.base_dir + 'loss_history.txt'))
            test_loss_results = list(
                np.loadtxt(self.base_dir + 'test_loss_history.txt'))
        else:
            train_loss_results = []
            test_loss_results = []
        train_rmse_results = []
        num_epochs = self.epochs
        c = 0
        minimum_model = None
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

            test_loss, _ = loss(model, test_data, test_labels, training=False)
            test_loss_results.append(test_loss)

            print(
                'Epoch: {:03d}, Loss: {:.5f}, Test Loss: {:.5f},'
                .format(epoch, epoch_loss_avg.result(), test_loss_results[-1])
                + 'RMSE: {:.5f}, Time: {:.3f}'
                .format(epoch_rmse_avg.result(), e-s), flush=True)

            if self.early_stop:
                c += 1
                if epoch == 0:
                    minimum_loss = test_loss_results[-1]
                    minimum_epoch = epoch
                    minimum_model = None
                else:
                    if test_loss_results[-1] < minimum_loss:
                        minimum_loss = test_loss_results[-1]
                        minimum_epoch = epoch
                        minimum_model = model
                        c = 0
                if minimum_model:
                    if c == round((self.epochs/100)*2):
                        print('Early stopped. Minimum at = ' +
                              str(minimum_epoch) +
                              ' Epochs used = ' + str(epoch))
                        break

            if (epoch + 1) % 10 == 0:
                model.save(self.base_dir + 'model.h5')
                np.savetxt(
                    self.base_dir + 'loss_history.txt', train_loss_results)
                np.savetxt(
                    self.base_dir + 'test_loss_history.txt', test_loss_results)

        if minimum_model:
            minimum_model.save(self.base_dir + 'model.h5')
        else:
            model.save(self.base_dir + 'model.h5')
        np.savetxt(self.base_dir + 'loss_history.txt', train_loss_results)
        np.savetxt(self.base_dir + 'test_loss_history.txt', test_loss_results)
