import tensorflow as tf
import numpy as np
from tensorflow import keras

class prediction():
    def __init__(self, params, **kwargs):
        self.params = params
        self.orig_z = np.arange(5, 50.1, 0.1)
        self.z = kwargs.pop('z', np.arange(5, 50.1, 0.1))
        self.base_dir = kwargs.pop('base_dir', 'results/')

        self.signal, self.z = self.result()

    def result(self):
        model = keras.models.load_model(self.base_dir + 'zT_model')

        #data_abs_max = np.loadtxt(self.base_dir + 'data_abs_max.txt')
        data_means = np.loadtxt(self.base_dir + 'data_means.txt')
        data_stds = np.loadtxt(self.base_dir + 'data_stds.txt')
        label_min = np.load(self.base_dir + 'label_min.npy')
        normalised_params = [
            (self.params[i]-data_means[i])/data_stds[i]
            for i in range(len(self.params))]

        #do log!!!!

        print(label_min)
        z = (self.z - self.orig_z.mean())/self.orig_z.std()

        predicted_spectra = []
        for j in range(len(z)):
            x = np.hstack([normalised_params, z[j]])
            x = x[np.newaxis, :]
            temp = model(x, training=False)
            #print(temp)
            predicted_spectra.append(temp[0][0].numpy())
        predicted_spectra = np.array(predicted_spectra)
        print('predicted spectra made')
        #print(predicted_spectra)
        print(predicted_spectra.shape)

        for i in range(predicted_spectra.shape[0]):
            predicted_spectra[i] = predicted_spectra[i]*label_min
            z[i] = z[i]*self.orig_z.std() + self.orig_z.mean()
        #print(predicted_spectra)

        return predicted_spectra, z
