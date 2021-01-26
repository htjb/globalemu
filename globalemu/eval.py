import tensorflow as tf
import numpy as np
from tensorflow import keras
from globalemu.cmSim import calc_signal
from tensorflow.keras import backend as K
import gc
import os

class evaluate():
    def __init__(self, parameters, **kwargs):
        self.params = parameters
        self.xHI = kwargs.pop('xHI', False)
        self.base_dir = kwargs.pop('base_dir', 'model_dir/')

        if self.xHI is False:
            self.z = kwargs.pop('z', np.linspace(5, 50, 451))
        else:
            self.z = kwargs.pop('z', np.hstack([np.arange(5, 15.1, 0.1),
                np.arange(16, 31, 1)]))

        self.signal, self.z_out = self.result()

    def result(self):

        model = keras.models.load_model(
            self.base_dir + 'model.h5',
            compile=False)

        data_mins = np.loadtxt(self.base_dir + 'data_mins.txt')
        data_maxs = np.loadtxt(self.base_dir + 'data_maxs.txt')
        samples = np.loadtxt(self.base_dir + 'samples.txt')

        params = []
        for i in range(len(self.params)):
            if i in set([0, 1]):
                params.append(np.log10(self.params[i]))
            elif i == 2:
                if self.params[i] == 0:
                    self.params[i] = 1e-6
                params.append(np.log10(self.params[i]))
            else: params.append(self.params[i])

        normalised_params = [
            (params[i] - data_mins[i])/(data_maxs[i] - data_mins[i])
            for i in range(len(params))]
        norm_z = (self.z - samples.min())/(samples.max()-samples.min())

        if isinstance(norm_z, np.ndarray):
            evaluation = []
            x = []
            for j in range(len(norm_z)):
                x.append(np.hstack([normalised_params, norm_z[j]]))
            x = np.array(x)
            tensor = tf.convert_to_tensor(x, dtype=tf.float32)
            result = model.predict(tensor)
            evaluation.append(result.T[0])
            K.clear_session()
            gc.collect()
            evaluation = np.array(evaluation)[0]
        else:
            x = np.hstack([normalised_params, norm_z]).astype(np.float32)
            result = model.predict_on_batch(x[np.newaxis, :])
            evaluation = result[0][0]

        if self.xHI is False:
            label_stds = np.load(self.base_dir + 'labels_stds.npy')
            if isinstance(evaluation, np.ndarray):
                for i in range(evaluation.shape[0]):
                    evaluation[i] = evaluation[i]*label_stds
            else:
                evaluation *= label_stds

            res = calc_signal(self.z, self.base_dir)
            evaluation += res.deltaT

        return evaluation, self.z
