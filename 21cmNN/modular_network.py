import numpy as np
from zT.two_preprocess import process
from zT.network import nn
from zT.eval import prediction

#layer_size = [2, 4, 4, 4]
#base_dir = '2-4-4-4/'
layer_size = [128, 64, 64, 128]
base_dir = '128-64-64-128/'
process(300, base_dir=base_dir)

# batchsize, layersize, activation, dropout, epochs, learning rate, kwargs
nn(100, layer_size, 'relu', 0.05, 80, 5e-7, base_dir=base_dir)

test_data = np.loadtxt('21cmGEM_data/Par_test_21cmGEM.txt')
test_labels = np.loadtxt('21cmGEM_data/T21_test_21cmGEM.txt')

#train_data = np.loadtxt('21cmGEM_data/Par_train_21cmGEM.txt')
#train_labels = np.loadtxt('21cmGEM_data/T21_train_21cmGEM.txt')

res = prediction(test_data[1500], base_dir=base_dir)
#res = prediction(train_data[7305], base_dir=base_dir)
predicted_spectrum = res.signal
z = res.z
import matplotlib.pyplot as plt

orig_z = np.arange(5, 50.1, 0.1)

plt.plot(orig_z, test_labels[1500])
#plt.plot(orig_z, train_labels[7305])
plt.plot(z, predicted_spectrum)
plt.hlines(test_labels[1500].mean(), orig_z.min(), orig_z.max())
plt.xlabel('z')
plt.ylabel(r'$\delta T$ [mk]')
plt.tight_layout()
plt.savefig(base_dir + 'prediction_example.pdf')
plt.show()
