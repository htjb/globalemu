import numpy as np
from zT.preprocess import process
from zT.network import nn
from zT.eval import prediction
import matplotlib.pyplot as plt
from zT.losses import loss_functions

layer_size = [8, 16, 8]
base_dir = 'testing/'
data_location = 'Resplit_data/'

num = 3000
process(num, base_dir=base_dir, data_location=data_location)

# batchsize, layersize, activation, dropout, epochs, learning rate, kwargs
nn(
    451, layer_size, 'tanh', 0.0,
    10, 1e-3, base_dir=base_dir)

orig_z = np.linspace(5, 50, 451)

test_data = np.loadtxt(data_location + 'test_data.txt')
test_labels = np.loadtxt(data_location + 'test_labels.txt')
#test_data = np.loadtxt(data_location + 'insample_test_data.txt')
#test_labels = np.loadtxt(data_location + 'insample_test_labels.txt')

train_data = np.loadtxt(data_location + 'train_data.txt')
train_labels = np.loadtxt(data_location + 'train_labels.txt')

samples = np.loadtxt(base_dir + 'samples.txt')

if num != 'full':
    ids = np.loadtxt(base_dir + 'indices.txt')

    for i in range(len(ids)):
        ids[i] = int(ids[i])

    inds = np.random.randint(0, len(ids), 4)

    signals, rmse = [], []
    for i in range(len(inds)):
        res = prediction(train_data[int(ids[inds[i]])], base_dir=base_dir, z=samples)
        signals.append(res.signal)
        rmse.append(loss_functions(train_labels[int(ids[inds[i]])], res.signal).rmse())
    signals = np.array(signals)
    rmse = np.array(rmse)
    z = res.z_out
    fig, axes = plt.subplots(2, 2, sharex=True, sharey=True)
    axes[0, 0].plot(orig_z, train_labels[int(ids[inds[0]])])
    axes[0, 0].plot(z, signals[0], label='rmse = {:.3f} mK'.format(rmse[0]))
    axes[0, 0].legend()
    axes[0, 1].plot(orig_z, train_labels[int(ids[inds[1]])])
    axes[0, 1].plot(z, signals[1], label='rmse = {:.3f} mK'.format(rmse[1]))
    axes[0, 1].legend()
    axes[1, 0].plot(orig_z, train_labels[int(ids[inds[2]])])
    axes[1, 0].plot(z, signals[2], label='rmse = {:.3f} mK'.format(rmse[2]))
    axes[1, 0].legend()
    axes[1, 1].plot(orig_z, train_labels[int(ids[inds[3]])])
    axes[1, 1].plot(z, signals[3], label='rmse = {:.3f} mK'.format(rmse[3]))
    axes[1, 1].legend()
    fig.add_subplot(111, frame_on=False)
    plt.tick_params(bottom=False, left=False, labelcolor='none')
    plt.xlabel('z')
    plt.ylabel(r'$\delta T$ [mk]')
    plt.tight_layout()
    plt.subplots_adjust(hspace=0, wspace=0)
    plt.savefig(base_dir + 'train_prediction_example.pdf')
    plt.show()
    #sys.exit(1)
else:
    inds = np.random.randint(0, len(ids), 4)

    signals, rmse = [], []
    for i in range(len(inds)):
        res = prediction(train_data[int(inds[i])], base_dir=base_dir, z=samples)
        signals.append(res.signal)
        rmse.append(loss_functions(train_labels[int(inds[i])], res.signal).rmse())
    signals = np.array(signals)
    rmse = np.array(rmse)
    z = res.z_out
    fig, axes = plt.subplots(2, 2, sharex=True, sharey=True)
    axes[0, 0].plot(orig_z, train_labels[int(inds[0])])
    axes[0, 0].plot(z, signals[0], label='rmse = {:.3f} mK'.format(rmse[0]))
    axes[0, 0].legend()
    axes[0, 1].plot(orig_z, train_labels[int(inds[1])])
    axes[0, 1].plot(z, signals[1], label='rmse = {:.3f} mK'.format(rmse[1]))
    axes[0, 1].legend()
    axes[1, 0].plot(orig_z, train_labels[int(inds[2])])
    axes[1, 0].plot(z, signals[2], label='rmse = {:.3f} mK'.format(rmse[2]))
    axes[1, 0].legend()
    axes[1, 1].plot(orig_z, train_labels[int(inds[3])])
    axes[1, 1].plot(z, signals[3], label='rmse = {:.3f} mK'.format(rmse[3]))
    axes[1, 1].legend()
    fig.add_subplot(111, frame_on=False)
    plt.tick_params(bottom=False, left=False, labelcolor='none')
    plt.xlabel('z')
    plt.ylabel(r'$\delta T$ [mk]')
    plt.tight_layout()
    plt.subplots_adjust(hspace=0, wspace=0)
    plt.savefig(base_dir + 'train_prediction_example.pdf')
    plt.show()

ind = np.random.randint(0, len(test_data), 4)
sigs, rmse = [], []
for i in range(len(ind)):
    res = prediction(test_data[ind[i]], base_dir=base_dir, z=samples)
    print(res.signal.min())
    rmse.append(loss_functions(test_labels[ind[i]], res.signal).rmse())
    sigs.append(res.signal)
sigs = np.array(sigs)
rmse = np.array(rmse)
print(sigs.shape)
z = res.z_out

fig, axes = plt.subplots(2, 2, sharex=True, sharey=True)
axes[0, 0].plot(orig_z, test_labels[ind[0]])
axes[0, 0].plot(z, sigs[0], label='rmse = {:.3f} mK'.format(rmse[0]))
axes[0, 0].legend()
axes[0, 1].plot(orig_z, test_labels[ind[1]])
axes[0, 1].plot(z, sigs[1], label='rmse = {:.3f} mK'.format(rmse[1]))
axes[0, 1].legend()
axes[1, 0].plot(orig_z, test_labels[ind[2]])
axes[1, 0].plot(z, sigs[2], label='rmse = {:.3f} mK'.format(rmse[2]))
axes[1, 0].legend()
axes[1, 1].plot(orig_z, test_labels[ind[3]])
axes[1, 1].plot(z, sigs[3], label='rmse = {:.3f} mK'.format(rmse[3]))
axes[1, 1].legend()
fig.add_subplot(111, frame_on=False)
plt.tick_params(bottom=False, left=False, labelcolor='none')
plt.xlabel('z')
plt.ylabel(r'$\delta T$ [mk]')
plt.tight_layout()
plt.subplots_adjust(hspace=0, wspace=0)
plt.savefig(base_dir + 'test_prediction_example.pdf')
plt.show()
