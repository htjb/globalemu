import numpy as np

class weight_calc():
    def __init__(self, train_data):
        self.train_data = train_data

        weights = []
        for j in range(self.train_data.shape[1]):
            hist, edges = np.histogram(self.train_data[:, j], bins=50)
            avg_hist = hist.mean()
            w = []
            for i in range(len(self.train_data[:, j])):
                h, e = np.histogram(
                    self.train_data[i, j], bins=50,
                    range=(
                    self.train_data[:, j].min(), self.train_data[:, j].max()))
                val_hist = hist[np.where(h == 1)][0]
                if val_hist == 0:
                    print(val_hist)
                w.append(avg_hist/val_hist)
            w = np.array(w)
            weights.append(w)
        weights = np.array(weights).T
        w = []
        for i in range(weights.shape[0]):
            w.append(np.sum(weights[i, :]))
        w = np.array(w)
        self.w = w/np.sum(w)
