import tensorflow as tf
from tensorflow import keras
import numpy as np
from tensorflow.keras import backend as K

class loss_functions():
    def __init__(self, y, y_):
        self.y = y
        self.y_ = y_

    def rmse(self):
    	return K.sqrt(K.mean(K.square(self.y - self.y_)))

    def mse(self):
        return K.mean(K.square(self.y - self.y_))

    def wmse(self, w):
        return K.sum((self.y - self.y_)**2*w)

    def mre(self):
        #print(np.abs((self.y.min()-self.y_.min())/self.y.min()))
        # 21cmGEM paper has relative error of abosrbtion trough temp not mean
        # over whole spectrum
        return K.mean(K.abs((self.y - self.y_)/self.y))

    def GEMLoss(self):
        return (K.sqrt(K.mean(K.square(self.y - self.y_)))/K.max(K.abs(self.y)))
