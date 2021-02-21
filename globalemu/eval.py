import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras import backend as K
import gc


class evaluate():
    def __init__(self, parameters, **kwargs):

        for key, values in kwargs.items():
            if key not in set(
                    ['xHI', 'base_dir', 'model', 'logs', 'gc', 'z']):
                raise KeyError("Unexpected keyward argument in evaluate()")

        self.params = parameters
        self.xHI = kwargs.pop('xHI', False)
        self.base_dir = kwargs.pop('base_dir', 'model_dir/')
        self.model = kwargs.pop('model', None)
        self.logs = kwargs.pop('logs', [0, 1, 2])
        self.garbage_collection = kwargs.pop('gc', False)

        if self.xHI is False:
            self.AFB = np.loadtxt(self.base_dir + 'AFB.txt')
            self.label_stds = np.load(self.base_dir + 'labels_stds.npy')

        self.original_z = np.loadtxt(self.base_dir + 'z.txt')
        self.z = kwargs.pop('z', self.original_z)
        self.data_mins = np.loadtxt(self.base_dir + 'data_mins.txt')
        self.data_maxs = np.loadtxt(self.base_dir + 'data_maxs.txt')
        self.cdf = np.loadtxt(self.base_dir + 'cdf.txt')

        self.signal, self.z_out = self.result()

    def result(self):

        if self.model is None:
            model = keras.models.load_model(
                self.base_dir + 'model.h5',
                compile=False)
        else:
            model = self.model

        params = []
        for i in range(len(self.params)):
            if i in set(self.logs):
                if self.params[i] == 0:
                    self.params[i] = 1e-6
                params.append(np.log10(self.params[i]))
            else:
                params.append(self.params[i])

        normalised_params = [
            (params[i] - self.data_mins[i]) /
            (self.data_maxs[i] - self.data_mins[i])
            for i in range(len(params))]

        norm_z = np.interp(self.z, self.original_z, self.cdf)

        if isinstance(norm_z, np.ndarray):
            x = [
                np.hstack([normalised_params, norm_z[j]])
                for j in range(len(norm_z))]
            tensor = tf.convert_to_tensor(x, dtype=tf.float32)
            result = model(tensor, training=False).numpy()
            evaluation = result.T[0]
            if self.garbage_collection is True:
                K.clear_session()
                gc.collect()
        else:
            x = np.hstack([normalised_params, norm_z]).astype(np.float32)
            result = model(x[np.newaxis, :], training=False).numpy()
            evaluation = result[0][0]

        if self.xHI is False:
            if isinstance(evaluation, np.ndarray):
                evaluation = [
                    evaluation[i]*self.label_stds
                    for i in range(evaluation.shape[0])]
            else:
                evaluation *= self.label_stds

            evaluation += np.interp(self.z, self.original_z, self.AFB)

        return evaluation, self.z
