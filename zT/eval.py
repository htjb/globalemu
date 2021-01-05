import tensorflow as tf
import numpy as np
from tensorflow import keras
from zT.cmSim import calc_signal
from tensorflow.keras import backend as K
import gc

class prediction():
    def __init__(self, parameters, **kwargs):
        self.params = parameters
        self.z = kwargs.pop('z', np.linspace(5, 50, 451))
        self.base_dir = kwargs.pop('base_dir', 'results/')
        self.model = kwargs.pop('model', 'load')

        self.signal, self.z_out = self.result()

    def result(self):
        if self.model == 'load':
            model = keras.models.load_model(self.base_dir + 'zT_model.h5', compile=False)
        else:
            model = self.model

        data_mins = np.loadtxt(self.base_dir + 'data_mins.txt')
        data_maxs = np.loadtxt(self.base_dir + 'data_maxs.txt')
        label_stds = np.load(self.base_dir + 'labels_stds.npy')
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
            predicted_spectra = []
            x = []
            for j in range(len(norm_z)):
                x.append(np.hstack([normalised_params, norm_z[j]]))
            x = np.array(x)
            tensor = tf.convert_to_tensor(x, dtype=tf.float32)
            temp = model.predict(tensor)
            predicted_spectra.append(temp.T[0])
            K.clear_session()
            gc.collect()
            predicted_spectra = np.array(predicted_spectra)[0]
        else:
            x = np.hstack([normalised_params, norm_z]).astype(np.float32)
            temp = model.predict_on_batch(x[np.newaxis, :])
            predicted_spectra = temp[0][0]

        if isinstance(predicted_spectra, np.ndarray):
            for i in range(predicted_spectra.shape[0]):
                predicted_spectra[i] = predicted_spectra[i]*label_stds
        else:
            predicted_spectra *= label_stds

        res = calc_signal(self.z, base_dir=self.base_dir)
        predicted_spectra += res.deltaT

        return predicted_spectra, self.z
