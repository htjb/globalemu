import numpy as np

class weight_calc():
    def __init__(self, train_data):
        self.train_data = train_data

        weights = []
        for j in range(self.train_data.shape[1]):
            hist, edges = np.histogram(self.train_data[:, j], bins=50)
            data_means = []
            for p in range(len(edges)):
                if p > 0:
                    data_means.append((edges[p] - edges[p-1])/2 + edges[p-1])
            data_means = np.array(data_means)
            count = 0
            for f in range(len(hist)):
                if hist[f] != 0:
                    count += 1
            avg_hist = np.sum(hist)/count
            w = []
            for i in range(len(self.train_data[:, j])):
                h, e = np.histogram(
                    self.train_data[i, j], bins=50,
                    range=(
                    self.train_data[:, j].min(), self.train_data[:, j].max()))
                val_hist = hist[np.where(h == 1)][0]
                mean = data_means[np.where(h == 1)][0]
                w.append(avg_hist/(val_hist*mean))
            w = np.array(w)
            weights.append(w/np.sum(w))
        weights = np.array(weights).T
        w = []
        for i in range(weights.shape[0]):
            w.append(np.sum(weights[i, :]))
        w = np.array(w)
        self.w = w/np.sum(w)
