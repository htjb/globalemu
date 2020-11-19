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
        #log_samples = np.log10(samples)
        #norm_z = (np.log10(self.z.copy()) - log_samples.min())/(log_samples.max() - log_samples.min()) # Using the resampled mean and std
        #norm_z = (self.z.copy() - samples.mean())/(samples.std())
        #norm_z = (np.log10(self.z.copy()) - log_samples.mean())/log_samples.std()
        norm_z = (samples.copy() - samples.min())/(samples.max()-samples.min())

        if isinstance(norm_z, np.ndarray):
            predicted_spectra = []
            for j in range(len(norm_z)):
                x = np.hstack([normalised_params, norm_z[j]])
                temp = model(x[np.newaxis, :], training=False)
                predicted_spectra.append(temp[0][0].numpy())
            predicted_spectra = np.array(predicted_spectra)
        else:
            x = np.hstack([normalised_params, norm_z])
            temp = model(x[np.newaxis, :], training=False)
            predicted_spectra = temp[0][0].numpy()
        #print('predicted spectra made')
        #print(predicted_spectra)

        if isinstance(predicted_spectra, np.ndarray):
            for i in range(predicted_spectra.shape[0]):
                predicted_spectra[i] = predicted_spectra[i]*label_stds +label_means
        else:
            predicted_spectra *= label_stds
            predicted_spectra += label_means
        #print(predicted_spectra)

        res = calc_signal(self.z, reionization='unity')
        predicted_spectra += res.deltaT*1e3
        predicted_spec = np.interp(self.orig_z, self.z, predicted_spectra)

        return predicted_spec, self.orig_z
