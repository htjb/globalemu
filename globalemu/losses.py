from tensorflow.keras import backend as K


class loss_functions():
    def __init__(self, y, y_):
        self.y = y
        self.y_ = y_

    def rmse(self):
        return K.sqrt(K.mean(K.square(self.y - self.y_)))

    def mse(self):
        return K.mean(K.square(self.y - self.y_))

    def GEMLoss(self):
        return (
            K.sqrt(K.mean(K.square(self.y - self.y_))) /
            K.max(K.abs(self.y)))
