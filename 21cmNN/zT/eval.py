import tensorflow as tf
import numpy as np
from tensorflow import keras
from zT.cmSim import calc_signal
from tensorflow.keras import backend as K
import gc

class prediction():
    def __init__(self, parameters, **kwargs):
        self.params = parameters
        self.orig_z = np.linspace(5, 50, 451)
        self.z = kwargs.pop('z', np.linspace(5, 50.1, 451))
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
        label_means = np.load(self.base_dir + 'labels_means.npy')
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
        #print('params', params)

        #normalised_params = [
        #    (params[i] - data_means[i])/data_stds[i]
        #    for i in range(len(params))]
        normalised_params = [
            (params[i] - data_mins[i])/(data_maxs[i] - data_mins[i])
            for i in range(len(params))]
        #print('norm', normalised_params)
        norm_z = (samples.copy() - samples.min())/(samples.max()-samples.min())
        #ls = np.log10(samples)
        #norm_z = (ls.copy() - ls.min())/(ls.max()-ls.min())

        if isinstance(norm_z, np.ndarray):
            """predicted_spectra = []
            for j in range(len(norm_z)):
                x = np.hstack([normalised_params, norm_z[j]])#.astype(np.float32)
                tensor = tf.convert_to_tensor(x[np.newaxis, :], dtype=tf.float32)
                temp = model.predict_on_batch(tensor)#, training=False)
                predicted_spectra.append(temp[0][0])#.numpy())
                K.clear_session()
                gc.collect()
            predicted_spectra = np.array(predicted_spectra)"""

            predicted_spectra = []
            x = []
            for j in range(len(norm_z)):
                x.append(np.hstack([normalised_params, norm_z[j]]))#.astype(np.float32)
            x = np.array(x)
            tensor = tf.convert_to_tensor(x, dtype=tf.float32)
            temp = model.predict(tensor)#, training=False)
            predicted_spectra.append(temp.T[0])#.numpy())
            K.clear_session()
            gc.collect()
            predicted_spectra = np.array(predicted_spectra)[0]
        else:
            x = np.hstack([normalised_params, norm_z]).astype(np.float32)
            temp = model.predict_on_batch(x[np.newaxis, :])#, training=False)
            predicted_spectra = temp[0][0]#.numpy()
        #print('predicted spectra made')
        #print(predicted_spectra)

        if isinstance(predicted_spectra, np.ndarray):
            for i in range(predicted_spectra.shape[0]):
                predicted_spectra[i] = predicted_spectra[i]*label_stds +label_means
                #predicted_spectra[i] = predicted_spectra[i]*(label_max - label_min) + label_min
        else:
            #predicted_spectra *= (label_max - label_min)
            #predicted_spectra += label_min
            predicted_spectra *= label_stds
            predicted_spectra += label_means
        #print(predicted_spectra)

        res = calc_signal(self.z, base_dir=self.base_dir)
        predicted_spectra += res.deltaT
        predicted_spec = np.interp(self.orig_z, self.z, predicted_spectra)


        return predicted_spec, self.orig_z
