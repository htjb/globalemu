import tensorflow as tf
import numpy as np
from tensorflow import keras
from zT.cmSim import calc_signal

class prediction():
    def __init__(self, parameters, **kwargs):
        self.params = parameters
        self.orig_z = np.arange(5, 50.1, 0.1)
        self.z = kwargs.pop('z', np.arange(5, 50.1, 0.1))
        self.base_dir = kwargs.pop('base_dir', 'results/')

        self.signal, self.z_out = self.result()

    def result(self):
        model = keras.models.load_model(self.base_dir + 'zT_model', compile=False)

        #data_means = np.loadtxt(self.base_dir + 'data_means.txt')
        #data_stds = np.loadtxt(self.base_dir + 'data_stds.txt')
        data_mins = np.loadtxt(self.base_dir + 'data_mins.txt')
        data_maxs = np.loadtxt(self.base_dir + 'data_maxs.txt')
        label_means = np.load(self.base_dir + 'labels_means.npy')
        label_stds = np.load(self.base_dir + 'labels_stds.npy')
        #label_min = np.load(self.base_dir + 'label_min.npy')
        #label_max = np.load(self.base_dir + 'label_max.npy')
        samples = np.loadtxt('samples.txt')

        params = []
        for i in range(len(self.params)):
            if i in set([0, 1]):
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
            predicted_spectra = []
            for j in range(len(norm_z)):
                x = np.hstack([normalised_params, norm_z[j]]).astype(np.float32)
                temp = model.predict_on_batch(x[np.newaxis, :])#, training=False)
                predicted_spectra.append(temp[0][0])#.numpy())
            predicted_spectra = np.array(predicted_spectra)
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

        res = calc_signal(self.z, reionization='unity')
        predicted_spectra += res.deltaT*1e3
        predicted_spec = np.interp(self.orig_z, self.z, predicted_spectra)

        """uni_z, zero_z = [], []
        for i in range(len(self.orig_z)):
            if self.orig_z[i] <= self.z.max():
                uni_z.append(self.orig_z[i])
            else:
                zero_z.append(self.orig_z[i])
        uni_z, zero_z = np.array(uni_z), np.array(zero_z)

        predicted_spec = np.hstack([
            np.interp(uni_z, self.z, predicted_spectra), [0]*len(zero_z)])

        res = calc_signal(self.orig_z, reionization='unity')
        predicted_spec += res.deltaT*1e3"""

        return predicted_spec, self.orig_z
