import numpy as np
from zT.preprocess import process
from zT.network import nn
from zT.eval import prediction
import matplotlib.pyplot as plt

layer_size = [8, 8]
base_dir = '8-8_improved/'
#layer_size = [128, 64, 64, 128]
#base_dir = '128-64-64-128/'
process(3000, base_dir=base_dir)

# batchsize, layersize, activation, dropout, epochs, learning rate, kwargs
nn(
    451, layer_size, 'tanh', 0.0,
    500, 5e-3, 8, 1, base_dir=base_dir)#, BN=False)

orig_z = np.arange(5, 50.1, 0.1)

test_data = np.loadtxt('21cmGEM_data/Par_test_21cmGEM.txt')
test_labels = np.loadtxt('21cmGEM_data/T21_test_21cmGEM.txt')

ids = np.loadtxt('testing/indices.txt')

for i in range(len(ids)):
    ids[i] = int(ids[i])

inds = np.random.randint(0, len(ids), 4)

train_data = np.loadtxt('21cmGEM_data/Par_train_21cmGEM.txt')
train_labels = np.loadtxt('21cmGEM_data/T21_train_21cmGEM.txt')

signals = []
for i in range(len(inds)):
    res = prediction(train_data[int(ids[inds[i]])], base_dir=base_dir)
    signals.append(res.signal)
signals = np.array(signals)
z = res.z
fig, axes = plt.subplots(2, 2, sharex=True, sharey=True)
axes[0, 0].plot(orig_z, train_labels[int(ids[inds[0]])])
axes[0, 0].plot(z, signals[0])
axes[0, 1].plot(orig_z, train_labels[int(ids[inds[1]])])
axes[0, 1].plot(z, signals[1])
axes[1, 0].plot(orig_z, train_labels[int(ids[inds[2]])])
axes[1, 0].plot(z, signals[2])
axes[1, 1].plot(orig_z, train_labels[int(ids[inds[3]])])
axes[1, 1].plot(z, signals[3])
fig.add_subplot(111, frame_on=False)
plt.tick_params(bottom=False, left=False, labelcolor='none')
plt.xlabel('z')
plt.ylabel(r'$\delta T$ [mk]')
plt.tight_layout()
plt.subplots_adjust(hspace=0, wspace=0)
plt.savefig(base_dir + 'train_prediction_example.pdf')
plt.show()
#sys.exit(1)

"""for i in range(len(train_data)):
    if np.any(i == ids):
        print(train_labels[i, 450])
        plt.plot(i, train_labels[i, 0], marker='.', c='k')
        plt.plot(i+ids.max(), train_labels[i, 50], marker='.', c='r')
        plt.plot(i+(2*ids.max()), train_labels[i, 450], marker='.', c='r')
plt.show()"""

#f = np.loadtxt('optimum_f.txt')

#res = prediction(f, test_data[15], base_dir=base_dir)
ind = np.random.randint(0, len(test_data), 4)
sigs = []
for i in range(len(ind)):
    res = prediction(test_data[ind[i]], base_dir=base_dir)
    print(res.signal.min())
    #res = prediction(f, train_data[int(ids[100])], base_dir=base_dir)
    #predicted_spectrum = res.signal
    sigs.append(res.signal)
sigs = np.array(sigs)
print(sigs.shape)
z = res.z

fig, axes = plt.subplots(2, 2, sharex=True, sharey=True)
axes[0, 0].plot(orig_z, test_labels[ind[0]])
axes[0, 0].plot(z, sigs[0])
axes[0, 1].plot(orig_z, test_labels[ind[1]])
axes[0, 1].plot(z, sigs[1])
axes[1, 0].plot(orig_z, test_labels[ind[2]])
axes[1, 0].plot(z, sigs[2])
axes[1, 1].plot(orig_z, test_labels[ind[3]])
axes[1, 1].plot(z, sigs[3])
fig.add_subplot(111, frame_on=False)
plt.tick_params(bottom=False, left=False, labelcolor='none')
plt.xlabel('z')
plt.ylabel(r'$\delta T$ [mk]')
plt.tight_layout()
plt.subplots_adjust(hspace=0, wspace=0)
plt.savefig(base_dir + 'test_prediction_example.pdf')
plt.show()
