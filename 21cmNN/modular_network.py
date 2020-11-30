import numpy as np
#from zT.preprocess import process
from zT.preprocess_resample import process
from zT.network import nn
from zT.eval import prediction
import matplotlib.pyplot as plt
from zT.losses import loss_functions

layer_size = [8, 16, 8]
#layer_size = [2, 4, 4, 4]
base_dir = '8-16-8_normAFB_logfx/'
#layer_size = [128, 64, 64, 128]
#base_dir = '128-64-64-128/'
num = 3000
#process(num, base_dir=base_dir)

# batchsize, layersize, activation, dropout, epochs, learning rate, kwargs
#nn(
#    451, layer_size, 'tanh', 0.0,
#    50, 1e-3, 8, 1, base_dir=base_dir)#, BN=False)

orig_z = np.linspace(5, 50, 451)

test_data = np.loadtxt('Resplit_data/test_data.txt')
test_labels = np.loadtxt('Resplit_data/test_labels.txt')

train_data = np.loadtxt('Resplit_data/train_data.txt')
train_labels = np.loadtxt('Resplit_data/train_labels.txt')

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
